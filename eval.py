
import sys
import torch
import torchaudio
from pathlib import Path
import numpy as np
from asteroid.metrics import get_metrics
# Add project root to Python path
project_root = Path.cwd()
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'hstasnet'))

from hstasnet.hstasnet import HSTasNet
from src.states import  load_model_from_package
import time
import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HSTasnetEvaluator:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_package = torch.load(model_path, map_location=device)
        self.model = load_model_from_package(self.model_package)
        self.model = self.model.to(device)
        self.model.eval()

    def save_metrics(self, results, output_dir="output"):
        """Save metrics to a CSV file."""
        # Convert metrics to DataFrame
        metrics = {
                "track_name": results["track_name"],
                "sample_rate": results["sample_rate"],
                "input_sdr": results["input_sdr"],
                "input_sir": results["input_sir"],
                "input_sar": results["input_sar"],
                "sdr": results["sdr"],
                "sir": results["sir"],
                "sar": results["sar"],
                # perf stats
                "avg_inference_time": results["perf_stats"]["summary"]["avg_inference_time"],
                "realtime_factor": results["perf_stats"]["summary"]["realtime_factor"],
                "gpu_memory": results["perf_stats"]["segments"][0]["gpu_memory"],
                "total_time": results["perf_stats"]["segments"][0]["total_time"],
                "prep_time": results["perf_stats"]["segments"][0]["prep_time"],
                "inference_time": results["perf_stats"]["segments"][0]["inference_time"]
            }
        df = pd.DataFrame([metrics])
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        csv_path = output_path / "metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Metrics saved to {csv_path}")


    def save_audio(self, results, track, output_dir="output"):
        """Save audio results to an output directory."""
        output_path = Path(output_dir) / results['track_name']
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n=== Saving Results for Track: {results['track_name']} ===")
        sr = results['sample_rate']
        
        # Save mixture
        mixture_path = Path(track) / "mixture.wav"
        if mixture_path.exists():
            logger.info("Saving mixture...")
            mixture, _ = torchaudio.load(mixture_path)
            torchaudio.save(output_path / "mixture.wav", mixture, sr)
        
        # Save ground truth and predictions
        for stem in results['predictions'].keys():
            logger.info(f"Saving {stem}...")
            # Ground truth
            if stem in results['ground_truth']:
                gt_audio = results['ground_truth'][stem]
                gt_file = output_path / f"{stem}_gt.wav"
                torchaudio.save(gt_file, gt_audio, sr)
            # Prediction
            pred_audio = results['predictions'][stem]
            pred_file = output_path / f"{stem}.wav"
            torchaudio.save(pred_file, pred_audio, sr)
        logger.info(f"Results saved to {output_path}")



    def load_and_validate_audio(self, audio_path, segment_length):
        """Load and validate audio format"""
        audio, sr = torchaudio.load(audio_path)
        
        # Handle channels
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() > 2:
            audio = audio[:2]
            
        # Handle length
        if audio.shape[1] < segment_length:
            pad_length = segment_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_length))
        elif audio.shape[1] > segment_length:
            audio = audio[:, :segment_length]

        # Move to device
        audio = audio.to(self.device)
        return audio, sr
    

    def calculate_metrics(self, mixture,  result):
        """Calculate SDR for a single track result"""
        metrics = {}
        # Get predictions and ground truth
        pred_sources = np.array([
            result['predictions'][stem].squeeze().numpy()
            for stem in ['drums', 'bass', 'other', 'vocals']
        ])
        true_sources = np.array([
            result['ground_truth'][stem].squeeze().numpy()
            for stem in ['drums', 'bass', 'other', 'vocals']
        ])

        # Ensure both sources have the same shape
        min_length = min(pred_sources.shape[-1], true_sources.shape[-1])
        pred_sources = np.array([ps[..., :min_length] for ps in pred_sources])
        true_sources = np.array([ts[..., :min_length] for ts in true_sources])

        # collapse mixture to mono
        mixture = mixture.mean(dim=1)

        # collapse sources to mono
        pred_sources = pred_sources.mean(axis=1)
        true_sources = true_sources.mean(axis=1)
        # pad the mixture to the same length as the sources
        mixture = torch.nn.functional.pad(mixture, (0, true_sources.shape[-1] - mixture.shape[-1]))
        mixture = mixture.cpu().numpy()
  
        # Calculate SDR
        metrics_dict = get_metrics(mix=mixture, clean=true_sources, estimate=pred_sources, sample_rate=result['sample_rate'], metrics_list=[ 'sdr', 'sir', 'sar'])

        # Store metrics
        metrics = {
            'file': result['track_name'],
            **metrics_dict
        }
        return metrics

    def process_track(self, track_path, stem_names= ['bass', 'drums', 'other', 'vocals'], segment_length_in_s=20.0):
        track_path = Path(track_path)
        sr = 44100
        segment_length = int(sr * segment_length_in_s)
        fade_length = int(sr * 1.0)
        perf_stats = {'segments': [], 'summary': {}}
        mixture, sr = self.load_and_validate_audio(track_path / "mixture.wav", segment_length)
        start_time = time.perf_counter()
        
        # Time preprocessing
        prep_start = time.perf_counter()
        fade_transform = torchaudio.transforms.Fade(
            fade_in_len=fade_length,
            fade_out_len=fade_length,
            fade_shape='linear'
        )
        mixture = fade_transform(mixture).to(self.device)
        mixture = mixture.unsqueeze(0)
        prep_time = time.perf_counter() - prep_start
        
        # Time inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_start = time.perf_counter()
        # do naive inference: pass our whole mixture through the model
        with torch.no_grad():
            predictions = self.model(mixture)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inference_time = time.perf_counter() - inference_start
        
        # Record performance stats
        perf_stats['segments'].append({
            'prep_time': prep_time,
            'inference_time': inference_time,
            'total_time': time.perf_counter() - start_time,
            'gpu_memory': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        })
        
        # Calculate summary
        perf_stats['summary'] = {
            'avg_inference_time': inference_time,
            'realtime_factor': inference_time / segment_length_in_s
            
        }
        
        # Add performance stats to existing results structure
        results = {
            "track_name": track_path.name,
            "predictions": {
                stem: predictions[0, i].detach().cpu()
                for i, stem in enumerate(stem_names)
            },
            "sample_rate": sr,
            "ground_truth": {},
            "perf_stats": perf_stats
        }
        # Load ground truth stems
        for stem in stem_names:
            audio, _ = self.load_and_validate_audio(track_path / f"{stem}.wav", segment_length)
            results["ground_truth"][stem] = fade_transform(audio).cpu()
        # # Calculate metrics
        metrics = self.calculate_metrics(mixture, results)
        # add the metrics to the results
        results = {**results, **metrics}
        return results
        
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=Path, required=True, help="Path to the model package")
    parser.add_argument('--input_dir', type=Path, required=True, help="Path to the input audio sample")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    parser.add_argument("--segment_length", type=float, default=20.0, help="Segment length in seconds")
    parser.add_argument("--output_dir", type=Path, default="output", help="Output directory")
    args = parser.parse_args()
    evaluator = HSTasnetEvaluator(args.model_path, device=args.device)
    logger.info(f"Processing track { args.input_dir}")
    result = evaluator.process_track( args.input_dir, segment_length_in_s=args.segment_length)
    evaluator.save_audio(result, args.output_dir)
    evaluator.save_metrics(result, args.output_dir)



if __name__ == '__main__':
    main()