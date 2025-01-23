import argparse
import logging
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def display_mel_grid(
    audio_file_paths,
    num_cols=2,
    num_rows=None,
    output_path="multi_mel.png",
    display=True,
    segment_length=None,
    start_time=None,
    hop_length=1024,
    n_fft=4096,
):
    num_files = len(audio_file_paths)
    num_rows = (num_files + 1) // num_cols
    audio_paths = [Path(audio_file) for audio_file in audio_file_paths]
    if any(not audio_path.exists() for audio_path in audio_paths):
        raise FileNotFoundError("One or more audio files do not exist.")
    if any(audio_path.suffix != ".wav" for audio_path in audio_paths):
        raise ValueError("Only .wav files are supported.")

    # warn if number of files is not even
    if num_files % num_cols != 0:
        logger.info(
            f"Warning: Number of files ({num_files}) is not divisible by number of columns ({num_cols})."
        )

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, audio_file in enumerate(audio_file_paths):
        y, sr = librosa.load(audio_file, sr=None)
        if start_time is not None:
            y = y[int(start_time * sr) :]
        if segment_length is not None:
            y = y[: int(segment_length * sr)]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=hop_length, fmax=20000, n_fft=n_fft
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        ax = axes[i]
        librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            x_axis="h",
            y_axis="mel",
            n_fft=n_fft,
            ax=ax,
            cmap="viridis",
            hop_length=hop_length,
            fmax=sr / 1.5,  # Show up to the Nyquist frequency
        )
        ax.set_title(audio_file.stem)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel Frequency")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    if display:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_files",
        type=Path,
        required=True,
        nargs="+",
        help="Paths to the audio files to display",
    )
    parser.add_argument(
        "--num_cols",
        type=int,
        default=2,
        help="Number of columns in the grid",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=None,
        help="Number of rows in the grid",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="multi_mel.png",
        help="Path to save the output grid",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the grid after saving",
    )
    parser.add_argument(
        "--sort", action="store_true", help="Sort the audio files alphabetically"
    )
    parser.add_argument(
        "--segment_length",
        type=float,
        default=None,
        help="Truncate audio to this length in seconds",
    )
    parser.add_argument(
        "--start_time",
        type=float,
        default=None,
        help="Start time in seconds to truncate audio",
    )

    args = parser.parse_args()
    if args.sort:
        args.audio_files.sort()

    display_mel_grid(
        audio_file_paths=args.audio_files,
        num_cols=args.num_cols,
        num_rows=args.num_rows,
        output_path=args.output_path,
        display=args.display,
        segment_length=args.segment_length,
        start_time=args.start_time,
    )


if __name__ == "__main__":
    main()
