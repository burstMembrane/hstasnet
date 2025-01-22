import torch
import torch.nn.functional as F
from torchaudio.transforms import Fade
from tqdm import tqdm
import argparse
from pathlib import Path
import soundfile as sf
import sys
# Add project root to Python path
project_root = Path.cwd()
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'hstasnet'))

from hstasnet.hstasnet import HSTasNet
from src.states import  load_model_from_package
import time
import argparse

def overlap_add_separation(
    model, mix, sample_rate, segment, overlap, device=None, num_sources=4
):
    device = device or mix.device
    mix = mix.to(device)
    chunk_len = int(sample_rate * segment)
    overlap_len = int(sample_rate * overlap)
    step_size = chunk_len - overlap_len
    batch_size, num_channels, total_len = mix.shape
    output = torch.zeros(
        batch_size, num_sources, num_channels, total_len, device=device
    )
    fade = (
        Fade(overlap_len, overlap_len, "linear")(
            torch.ones(1, chunk_len, device=device)
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    for start in tqdm(range(0, total_len, step_size)):
        end = min(start + chunk_len, total_len)
        current_chunk_len = end - start
        # Extract and pad the chunk
        chunk = mix[:, :, start:end]
        if current_chunk_len < chunk_len:
            chunk = F.pad(chunk, (0, chunk_len - current_chunk_len))
        with torch.no_grad():
            processed_chunk = model(chunk)
        if processed_chunk.size(-1) < chunk_len:
            processed_chunk = F.pad(
                processed_chunk, (0, chunk_len - processed_chunk.size(-1))
            )
        # Apply the fade and handle the last chunk
        processed_chunk = processed_chunk * fade[..., :chunk_len]
        output[:, :, :, start:end] += processed_chunk[:, :, :, :current_chunk_len]
    return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=Path, required=True, help="Path to the model package")
    parser.add_argument('--input_audio', type=Path, required=True, help="Path to the input audio sample")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")

    parser.add_argument("--output_dir", type=Path, default="output", help="Output directory")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap length in seconds")
    parser.add_argument("--segment", type=float, default=20.0, help="Segment length in seconds")
    parser.add_argument("--num_sources", type=int, default=4, help="Number of sources to separate")

    return  parser.parse_args()
def main():
    args = get_args()
    if args.model_path:
        model_package = torch.load(args.model_path, map_location=args.device)
        model = load_model_from_package(model_package)
    mix, sample_rate = sf.read(args.input_audio)
    mix = torch.tensor(mix).unsqueeze(0)
    output = overlap_add_separation(
        model, mix, sample_rate, args.segment, args.overlap, args.device, args.num_sources
    )
    print(output.shape)

if __name__ == "__main__":
    main()