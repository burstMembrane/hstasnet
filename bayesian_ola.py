import logging
import time
import warnings

import numpy as np
import torch
import torchaudio
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from tqdm import tqdm

warnings.filterwarnings("ignore")
from batch_ola import get_args, load_model_from_package, overlap_add_separation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def optimisation_objective(
    chunk, batch_size, model, mix, sample_rate, device, num_sources
):
    """Objective function for Bayesian optimisation."""
    try:

        logger.info(f"Running with chunk_length={chunk}, batch_size={batch_size}")
        torch.cuda.synchronize()  # Ensure all previous operations are completed
        start = time.perf_counter()

        overlap_add_separation(
            model=model,
            mix=mix,
            sample_rate=sample_rate,
            chunk=chunk,
            overlap=0.1,  # Keep overlap fixed
            device=device,
            num_sources=num_sources,
            batch_size=batch_size,
        )

        torch.cuda.synchronize()  # Ensure all GPU operations are completed
        end = time.perf_counter()
        elapsed_time = end - start
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(
                f"CUDA OOM encountered with chunk_length={chunk}, batch_size={batch_size}."
            )
            torch.cuda.empty_cache()  # Free memory
            elapsed_time = 1000.0  # Penalize configuration
        else:
            logger.error(f"Error during processing: {e}")
            elapsed_time = 1000.0  # Penalize configuration
    return elapsed_time


def bayesian_optimisation(model, mix, sample_rate, device, num_sources):
    # Define the search space
    space = [
        Integer(1.0, 5.0, name="chunk"),  # Chunk size in seconds
        Integer(20, 64, name="batch_size"),  # Batch size
    ]

    @use_named_args(space)
    def objective(
        chunk,
        batch_size,
    ):
        return optimisation_objective(
            chunk, batch_size, model, mix, sample_rate, device, num_sources
        )

    # Run Bayesian optimisation
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=100,  # Number of optimisation iterations
        random_state=42,
    )
    return result


def main():
    args = get_args()
    if args.model_path:
        model_package = torch.load(args.model_path, map_location=args.device)
        model = load_model_from_package(model_package)
        model = model.to(args.device)
    mix, sr = torchaudio.load(args.input_audio)
    mix = mix.to(args.device)
    mix = mix.unsqueeze(0)

    # Run Bayesian optimisation
    logger.info(
        "Starting Bayesian optimisation for chunk, batch size, and thread configuration."
    )
    result = bayesian_optimisation(
        model=model,
        mix=mix,
        sample_rate=sr,
        device=args.device,
        num_sources=args.num_sources,
    )

    optimal_chunk, optimal_batch_size = result.x
    logger.info(
        f"Optimal parameters found: chunk_length={optimal_chunk}, batch_size={optimal_batch_size}, "
    )
    logger.info(f"Minimum processing time: {result.fun:.2f} seconds")


if __name__ == "__main__":
    main()
