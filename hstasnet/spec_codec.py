import torch  # Core PyTorch
import torch.nn as nn  # Neural network modules
import torch.nn.functional as ff  # Functional ops (e.g. activation)
import torchaudio.transforms as tt  # Audio-specific transforms (STFT, inverse-STFT)

class SpecEncoder(nn.Module):
    def __init__(self,
                 n_win: int = 1024,
                 n_hop: int = 512,
                 n_fft: int = 1024,
                 window: str = 'hamming',
                 device=torch.device('cpu')):
        """
        Spectrogram-based encoder for converting time-domain signals into
        magnitude and phase representations (Section 3.2 in the paper).
        Args:
          n_win  : Window size for each STFT frame
          n_hop  : Hop size (overlap offset) for STFT
          n_fft  : Number of FFT points, typically >= n_win
          window : Type of window function (hamming, hann, etc.)
          device : PyTorch device to place tensors on
        """
        super().__init__()
        self.n_win = n_win
        self.n_hop = n_hop
        self.n_fft = n_fft

        # Select the appropriate window function based on 'window' argument
        if window == 'bartlett':
            window_fn = torch.bartlett_window
        elif window == 'blackman':
            window_fn = torch.blackman_window
        elif window == 'hamming':
            window_fn = torch.hamming_window
        elif window == 'hann':
            window_fn = torch.hann_window
        elif window == 'kaiser':
            window_fn = torch.kaiser_window
        else:
            raise Exception(f"Invalid window type for STFT : '{window}'.")

        # Create a Spectrogram transform from torchaudio, which outputs
        # a complex tensor (real + imag) when power=None
        # normalized=True divides by sum of window elements to scale the data
        # onesided=True gives only [0..n_fft//2] positive frequencies
        # center=False avoids time-padding
        self.transform = tt.Spectrogram(
            n_fft=n_fft,
            hop_length=n_hop,
            power=None,
            win_length=n_win,
            window_fn=window_fn,
            normalized=True,
            onesided=True,
            center=False
        ).to(device)

    def forward(self, waveform):
        """
        Forward pass: Takes a batch of waveforms [B, L] and returns:
          spec_magn: [B, T, F] magnitude
          spec_angl: [B, T, F] phase angles in radians
        """
        # 1) Compute the complex spectrogram using torchaudio
        spec = self.transform(waveform)   # Shape: [B, F, T]

        # 2) Separate out the magnitude
        spec_magn = torch.abs(spec)       # [B, F, T]
        # Rearrange to [B, T, F] for consistency with the time path
        spec_magn = spec_magn.permute(0, 2, 1)

        # 3) Separate out the phase angle
        spec_angl = torch.angle(spec)     # [B, F, T]
        spec_angl = spec_angl.permute(0, 2, 1)  # [B, T, F]

        return spec_magn, spec_angl

class SpecDecoder(nn.Module):
    def __init__(self,
                 n_win: int = 1024,
                 n_hop: int = 512,
                 n_fft: int = 1024,
                 window: str = 'hamming',
                 device=torch.device('cpu')):
        """
        Spectrogram-based decoder for reconstructing time-domain waveforms
        from magnitude and phase (Section 3.2 in the paper).
        Args:
          n_win  : Window size for each inverse-STFT frame
          n_hop  : Hop size for inverse-STFT
          n_fft  : Number of FFT points
          window : Window function type
          device : PyTorch device for computations
        """
        super().__init__()
        self.n_win = n_win
        self.n_hop = n_hop
        self.n_fft = n_fft

        # Window selection for the inverse STFT
        if window == 'bartlett':
            window_fn = torch.bartlett_window
        elif window == 'blackman':
            window_fn = torch.blackman_window
        elif window == 'hamming':
            window_fn = torch.hamming_window
        elif window == 'hann':
            window_fn = torch.hann_window
        elif window == 'kaiser':
            window_fn = torch.kaiser_window
        else:
            raise Exception(f"Invalid window type for STFT : '{window}'.")

        # InverseSpectrogram from torchaudio reverts a complex spectrogram
        # back to time-domain audio
        self.transform = tt.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=n_hop,
            win_length=n_win,
            window_fn=window_fn,
            normalized=True,
            onesided=True,
            center=False
        ).to(device)

    def forward(self, spec_magn, spec_angl, waveform_length=None):
        """
        Forward pass:
          spec_magn: [B, T, F] magnitude
          spec_angl: [B, T, F] phase angles
          waveform_length: optional int specifying final length in samples
        Returns:
          waveform: [B, L] the reconstructed waveform
        """
        # 1) Convert magnitude and angle back to real & imaginary parts
        spec_real = spec_magn * torch.cos(spec_angl)  # [B, T, F]
        spec_imag = spec_magn * torch.sin(spec_angl)  # [B, T, F]
        spec = torch.complex(spec_real, spec_imag)    # Combine into a complex tensor

        # 2) Rearrange dimensions for inverse STFT [B, F, T]
        spec = spec.permute(0, 2, 1)

        # 3) Perform the inverse spectrogram to recover time-domain signal
        waveform = self.transform(spec, length=waveform_length)  # [B, L]

        return waveform

if __name__ == '__main__':
    B, C, L = 10, 2, 500000
    x = torch.randn(B, L)  # [10, 500000]
    print(f'Input shape: {x.size()}')

    # Instantiate the spectrogram encoder
    encoder = SpecEncoder(window='hamming')

    # Encode to magnitude & angle
    y_magn, y_angl = encoder(x)
    print(f'Magnitude shape: {y_magn.size()}')

    # Instantiate the spectrogram decoder
    decoder = SpecDecoder(window='hamming')

    # Decode back to waveform
    z = decoder(y_magn, y_angl, waveform_length=None)
    print(f'Decoded shape: {z.size()}')

    # Compare a short slice of input & output
    print(f'Original first batch sample (slice): {x[0, :8]}')
    print(f'Decoded first batch sample (slice):  {z[0, :8]}')