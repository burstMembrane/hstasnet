import torch  # For tensor operations
import torch.nn as nn  # Provides neural network layers like Linear, LSTM etc.
import torch.nn.functional as ff  # Functional interface for some operations (e.g. relu, pad)

EPS = 1e-8  # A small constant to avoid division by zero

class TimeEncoder(nn.Module):
    def __init__(self,
                 N: int = 1024,
                 O: int = 512,
                 M: int = 1500,
                 device=torch.device('cpu')):
        """
        A learnable time-domain encoder similar to the TasNet front-end (Section 3.2 in the paper).
        - N: frame size for segmenting the waveform into overlapping chunks
        - O: hop size (overlap) between chunks
        - M: the encoded feature dimension, i.e. how many learned basis signals
        """
        super().__init__()
        self.N = N  # Number of samples per frame
        self.O = O  # Number of samples to shift for the next frame
        self.M = M  # Encoded feature dimension

        # "conv" transforms each frame into M latent features (learned basis).
        self.conv = nn.Linear(in_features=N, out_features=M, device=device)

        # "gate" produces a gating signal for each of those M features (similar to GLU).
        self.gate = nn.Linear(in_features=N, out_features=M, device=device)

        # Non-linearities used after linear layers
        self.relu = ff.relu
        self.sigmoid = torch.sigmoid

    def forward(self, waveform):
        """
        Forward pass:
        waveform: [B, L], where B=batch, L=samples per batch entry.

        Returns:
          x     : [B, T, M] encoded features (T=number of frames, M=dimension)
          x_norm: [B, T, 1] L2 norm of each frame, for use in the decoder
        """
        # 1) Convert the 1D signals into overlapping frames of size N with hop O.
        x = waveform.unfold(dimension=1, size=self.N, step=self.O)  # shape: [B, T, N]

        # 2) Compute L2 norm of each frame for normalisation (avoid large amplitude).
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # shape: [B, T, 1]

        # 3) Normalise each frame by its L2 norm to have magnitude ~1 (plus EPS to avoid /0).
        x = x / (x_norm + EPS)  # shape remains [B, T, N]

        # 4) Apply learned basis (conv) with ReLU, plus gate with sigmoid, then combine (GLU-like).
        conv = self.relu(self.conv(x))       # [B, T, M]
        gate = self.sigmoid(self.gate(x))    # [B, T, M]
        x = conv * gate                      # Element-wise gating to produce final features

        return x, x_norm


class TimeDecoder(nn.Module):
    def __init__(self,
                 N: int = 1024,
                 O: int = 512,
                 M: int = 1500,
                 device=torch.device('cpu')):
        """
        A learnable time-domain decoder that inverts the TimeEncoder (Section 3.2 in the paper).
        - N: frame size
        - O: hop size
        - M: encoded feature dimension
        """
        super().__init__()
        self.N = N
        self.O = O
        self.M = M

        # Linear layer to map M-dim features back to N samples per frame
        self.linear = nn.Linear(in_features=M, out_features=N, device=device)

    def forward(self, waveform_encoding, waveform_norm, waveform_length=None):
        """
        Forward pass:
          waveform_encoding: [B, T, M] the time-domain features from TimeEncoder
          waveform_norm    : [B, T, 1] the L2 norms used to scale back the magnitude
          waveform_length  : optional integer specifying final signal length

        Returns:
          x: [B, L] time-domain signal reconstructed from frames
        """
        # 1) Map encoded features back to waveforms frames via learned linear transform.
        x = self.linear(waveform_encoding)  # [B, T, N]

        # 2) Reverse the L2 normalisation step using the stored norms.
        x = x * waveform_norm  # [B, T, N]

        # 3) Merge the overlapped frames back into a continuous signal.
        #    overlap_ratio = N // O, e.g. 1024 // 512 = 2
        x = overlap_add(x, self.N // self.O)  # [B, L]

        # 4) Optionally pad the output to a desired length.
        if waveform_length:
            L = x.size(-1)
            pad_amount = waveform_length - L
            x = ff.pad(x, (0, pad_amount), mode='constant')  # left=0, right=pad_amount

        return x


def overlap_add(frames, overlap_ratio=2):
    """
    Combine overlapping frames to reconstruct time-domain signals.

    frames: [B, T, N], T=number of frames, N=frame_size
    overlap_ratio: typically N//O (for T=overlap_ratio * (T-1) + 1 frames in total).
                   e.g. 1024//512 = 2

    Returns:
      signal: [B, L] where L is the recovered length after overlap-add
    """
    batch_size, num_frames, frame_size = frames.size()
    overlap_size = frame_size // overlap_ratio

    # Compute final signal length after stitching frames with overlap
    signal_size = (num_frames - 1) * overlap_size + frame_size

    # Prepare empty container for the reconstructed signal
    signal = torch.zeros(batch_size, signal_size, dtype=frames.dtype, device=frames.device)

    # Add each frame into the correct segment of the signal
    for i in range(num_frames):
        start_idx = i * overlap_size
        end_idx = start_idx + frame_size
        signal[:, start_idx:end_idx] += frames[:, i, :]

    # Multiply the extreme edges to adjust for partial overlaps
    signal[:, :overlap_size] *= overlap_ratio
    signal[:, -overlap_size:] *= overlap_ratio

    # Gradually taper the frames in the middle region for a smoother composite
    for i in range(1, overlap_ratio - 1):
        start_idx = i * overlap_size
        end_idx = start_idx + overlap_size
        scale = overlap_ratio / (i + 1)
        signal[:, start_idx:end_idx] *= scale
        signal[:, -end_idx:-start_idx] *= scale

    # Divide the entire signal by overlap_ratio at the end
    signal /= overlap_ratio

    return signal


if __name__ == '__main__':
    # Example usage
    B, C, S, L = 10, 2, 4, 500000  # B=batch, C=channels, S=sources (unused here), L=samples
    x = torch.randn(B, L)         # [10, 500000] random waveforms
    print(f'Input shape: {x.size()}')

    # Instantiate the TimeEncoder with typical settings
    N, O, M = 1024, 512, 1000
    encoder = TimeEncoder(N=N, O=O, M=M)

    # Encode
    y, y_norm = encoder(x)  # y: [B, T, M], y_norm: [B, T, 1]
    print(f'Encoded shape: {y.size()}')
    print(f'Frame norm shape: {y_norm.size()}')

    # Instantiate a corresponding TimeDecoder
    decoder = TimeDecoder(N=N, O=O, M=M)

    # Decode
    z = decoder(y, y_norm, waveform_length=None)  # z: [B, L']
    print(f'Decoded shape: {z.size()}')

    # Check overlap_add specifically
    x_ones = torch.ones(10, 16)
    frames = x_ones.unfold(1, size=4, step=2)  # shape: [B, T=7, N=4]
    recons = overlap_add(frames, 2)
    print(f'Original first row: {x_ones[0]}')
    print(f'Reconstructed first row: {recons[0]}')