import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Fade
from tqdm import tqdm

# Add project root to Python path
project_root = Path.cwd()
sys.path.append(str(project_root))
sys.path.append(str(project_root / "hstasnet"))

import argparse
import logging
import time

from hstasnet.hstasnet import HSTasNet
from src.states import load_model_from_package

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def overlap_add_separation(
    model, mix, sample_rate, chunk, overlap, device=None, num_sources=4
):
    start = time.perf_counter()
    chunk_len = int(sample_rate * chunk)
    overlap_len = int(sample_rate * overlap)
    step_size = chunk_len - overlap_len

    batch_size, num_channels, total_len = mix.shape
    output = torch.zeros(
        batch_size, num_sources, num_channels, total_len, device=device
    )
    fade = Fade(overlap_len, overlap_len, "linear")(
        torch.ones(1, chunk_len, device=device)
    )
    fade = fade.unsqueeze(0).unsqueeze(0)[..., :chunk_len]

    # 1) Gather all chunks
    chunks = []
    chunk_idxs = []
    for start_pos in range(0, total_len, step_size):
        end_pos = min(start_pos + chunk_len, total_len)
        chunk_raw = mix[:, :, start_pos:end_pos]
        if (end_pos - start_pos) < chunk_len:
            chunk_raw = F.pad(chunk_raw, (0, chunk_len - (end_pos - start_pos)))

        chunks.append(chunk_raw)
        chunk_idxs.append((start_pos, end_pos))

    # 2) Single batched forward pass
    chunks_tensor = torch.cat(chunks, dim=0).to(
        device
    )  # shape [n_chunks*batch_size, channels, chunk_len]
    with torch.no_grad():
        processed_batch = model(chunks_tensor)

    # 3) Reassemble
    n = 0
    for start_pos, end_pos in chunk_idxs:
        current_len = end_pos - start_pos

        # pad the processed chunk if needed
        if processed_batch.size(-1) < chunk_len:
            processed_batch = F.pad(
                processed_batch, (0, chunk_len - processed_batch.size(-1))
            )
        processed_chunk = processed_batch[n : n + batch_size] * fade
        output[:, :, :, start_pos:end_pos] += processed_chunk[:, :, :, :current_len]
        n += batch_size

    end = time.perf_counter()
    logger.info(f"Processing took {end - start:.2f} seconds")
    return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=Path, required=True, help="Path to the model package"
    )
    parser.add_argument(
        "--input_audio", type=Path, required=True, help="Path to the input audio sample"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on"
    )

    parser.add_argument(
        "--output_dir", type=Path, default="output", help="Output directory"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.1, help="Overlap length in seconds"
    )
    parser.add_argument(
        "--chunk_length", type=float, default=2.0, help="chunk length in seconds"
    )
    parser.add_argument(
        "--num_sources", type=int, default=4, help="Number of sources to separate"
    )

    return parser.parse_args()


def main():
    args = get_args()
    if args.model_path:
        model_package = torch.load(args.model_path, map_location=args.device)
        model = load_model_from_package(model_package)
        model = model.to(args.device)
    mix, sr = torchaudio.load(args.input_audio)
    mix = mix.to(args.device)
    mix = mix.unsqueeze(0)

    output = overlap_add_separation(
        model=model,
        mix=mix,
        sample_rate=sr,
        chunk=args.chunk_length,
        overlap=args.overlap,
        device=args.device,
        num_sources=args.num_sources,
    )
    # save the output to output_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # save with instruments [bass, drums, other, vocals]
    for i, source in enumerate(["bass", "drums", "other", "vocals"]):
        output_path = output_dir / f"{source}.wav"
        sf.write(output_path, output[0, i].cpu().numpy().T, sr)
        logger.info(f"Saved {source} to {output_path}")


if __name__ == "__main__":
    main()
