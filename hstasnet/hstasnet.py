import torch  # Main PyTorch library for tensors
import torch.nn as nn  # Neural network layers (e.g. Linear, LSTM, etc.)
import torch.nn.functional as ff  # Functional interface for common operations

from spec_codec import SpecEncoder, SpecDecoder  # Modules for STFT-based encoding/decoding
from time_codec import TimeEncoder, TimeDecoder  # Modules for learned convolutional basis
from memory import Memory  # Custom LSTM-based "Memory" module

class HSTasNet(nn.Module):
    def __init__(self,
                 num_sources,
                 num_channels,
                 time_win_size: int = 1024,
                 time_hop_size: int = 512,
                 time_ftr_size: int = 1000,
                 spec_win_size: int = 1024,
                 spec_hop_size: int = 512,
                 spec_fft_size: int = 1024,
                 rnn_hidden_size: int = 1000,
                 rnn_num_layers: int = 1,
                 device=torch.device('cpu')):
        super().__init__()  # Init the superclass (nn.Module)

        self.num_sources = num_sources  # Number of target sources (vocals, drums, etc.)
        self.num_channels = num_channels  # Number of input channels (e.g. stereo=2)
        self.time_win_size = time_win_size  # Window size for time-domain encoder
        self.time_hop_size = time_hop_size  # Hop size for time-domain encoder
        self.time_ftr_size = time_ftr_size  # Dimensionality of time-domain features
        self.spec_win_size = spec_win_size  # Window size for frequency-domain STFT
        self.spec_hop_size = spec_hop_size  # Hop size for frequency-domain STFT
        self.spec_fft_size = spec_fft_size  # FFT size (e.g., 1024 points)
        self.rnn_hidden_size = rnn_hidden_size  # Hidden units for LSTMs
        self.rnn_num_layers = rnn_num_layers  # Number of LSTM layers in each RNN block

        # Calculate feature sizes for time and spec. This is the dimensionality after encoder.
        time_feature_size = time_ftr_size
        spec_feature_size = (spec_win_size // 2) + 1  # For real-valued STFT with n_fft=win_size
        self.time_feature_size = time_feature_size
        self.spec_feature_size = spec_feature_size

        # Assertions ensure time/spec windows/hops match for hybrid approach (Section 3.3).
        assert (time_win_size % time_hop_size) == 0, \
            f"time_win_size ({time_win_size}) must be a multiple of time_hop_size ({time_hop_size})"
        assert (spec_win_size % spec_hop_size) == 0, \
            f"spec_win_size ({spec_win_size}) must be a multiple of spec_hop_size ({spec_hop_size})"
        assert time_win_size == spec_win_size, \
            f"time_win_size ({time_win_size}) must equal spec_win_size ({spec_win_size})"
        assert time_hop_size == spec_hop_size, \
            f"time_hop_size ({time_hop_size}) must equal spec_hop_size ({spec_hop_size})"

        # Time-domain encoder & decoder (Section 3.2 for TasNet-like approach).
        self.time_encoder = TimeEncoder(N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device)
        self.time_decoder = TimeDecoder(N=time_win_size, O=time_hop_size, M=time_ftr_size, device=device)

        # LSTM block for time features input (Memory is a custom LSTM wrapper).
        self.time_rnn_in = Memory(
            input_size=num_channels * time_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=device,
        )
        # A linear skip-connection from input features to be added later.
        self.time_skip_fc = nn.Linear(
            in_features=num_channels * time_feature_size,
            out_features=rnn_hidden_size,
            device=device,
        )
        # Another LSTM block for post-hybrid time features.
        self.time_rnn_out = Memory(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            device=device,
        )
        # Final FC layer to produce time-domain masks (for S*C*M).
        self.time_mask_fc = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=num_channels * num_sources * time_feature_size,
            device=device,
        )

        # Frequency-domain encoder & decoder (Section 3.2 X-UMX-like approach).
        self.spec_encoder = SpecEncoder(
            n_win=spec_win_size,
            n_hop=spec_hop_size,
            n_fft=spec_fft_size,
            window='hamming',  # Window function for STFT
            device=device,
        )
        self.spec_decoder = SpecDecoder(
            n_win=spec_win_size,
            n_hop=spec_hop_size,
            n_fft=spec_fft_size,
            window='hamming',
            device=device,
        )

        # LSTM for frequency features input.
        self.spec_rnn_in = Memory(
            input_size=num_channels * spec_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            device=device,
        )
        # A linear skip-connection from the frequency input features.
        self.spec_skip_fc = nn.Linear(
            in_features=num_channels * spec_feature_size,
            out_features=rnn_hidden_size,
            device=device,
        )
        # Another LSTM block for post-hybrid frequency features.
        self.spec_rnn_out = Memory(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            device=device,
        )
        # Final FC layer to produce frequency-domain masks (S*C*F).
        self.spec_mask_fc = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=num_channels * num_sources * spec_feature_size,
            device=device,
        )

        # "Hybrid RNN" merges the time & frequency feature streams (Section 3.3).
        self.hybrid_rnn = Memory(
            input_size=2 * rnn_hidden_size,  # Concatenate time and freq hidden states
            hidden_size=2 * rnn_hidden_size,  # LSTMs with double dimension
            num_layers=rnn_num_layers,
            device=device,
        )

    def forward(self, waveform, length=None):
        """ Forward pass (Section 3.3: time path + freq path -> hybrid -> mask application -> decode).
            waveform: [B, C, L]
            length (optional): original length to pad after separation
        """
        B, C, L = waveform.size()  # Extract shapes
        x = waveform.view(B * C, L)  # Flatten channels into batch dimension

        # Encode in the time domain -> x_time has shape [(B*C), T, M]
        x_time, x_norm = self.time_encoder(x)

        # Reshape time features to feed RNN: [B, C, T, M] -> [B, T, C*M]
        BC, T, M = x_time.size()
        x_time = x_time.view(B, C, T, M)
        s_time = x_time.permute(0, 2, 1, 3).reshape(B, T, C * M)

        # Forward pass through the time-domain RNN block
        y_time = self.time_rnn_in(s_time)

        # Encode in the frequency domain -> x_spec has shape [(B*C), T, F]
        x_spec, x_angl = self.spec_encoder(x)

        # Reshape frequency features to feed RNN: [B, C, T, F] -> [B, T, C*F]
        BC, T, F = x_spec.size()
        x_spec = x_spec.view(B, C, T, F)
        s_spec = x_spec.permute(0, 2, 1, 3).reshape(B, T, C * F)

        # Forward pass through the frequency-domain RNN block
        y_spec = self.spec_rnn_in(s_spec)

        # Concatenate time & frequency features -> shape [B, T, 2*H]
        y = torch.cat((y_time, y_spec), dim=2)

        # The "hybrid RNN" merges the two feature streams (Section 3.3)
        y = self.hybrid_rnn(y)

        # Split back into time & frequency hidden states, each shape [B, T, H]
        H = self.rnn_hidden_size
        y_time, y_spec = torch.split(y, H, dim=2)

        # Additional RNN + skip-connection in time domain
        y_time = self.time_rnn_out(y_time)       # [B, T, H]
        s_time_fc = self.time_skip_fc(s_time)    # [B, T, H] skip
        y_time = y_time + s_time_fc             # Combine skip and RNN output

        # Additional RNN + skip-connection in frequency domain
        y_spec = self.spec_rnn_out(y_spec)       # [B, T, H]
        s_spec_fc = self.spec_skip_fc(s_spec)    # [B, T, H] skip
        y_spec = y_spec + s_spec_fc             # Combine skip and RNN output

        # Time-domain mask estimation -> shape [B, T, S*C*M], then rearrange
        m_time = self.time_mask_fc(y_time)
        S, C, M = self.num_sources, self.num_channels, self.time_feature_size
        m_time = m_time.view(B, T, S, C, M)         # [B, T, S, C, M]
        m_time = m_time.permute(0, 2, 3, 1, 4)      # [B, S, C, T, M]

        # Expand x_time for all sources, shape [B, S, C, T, M]
        x_time = x_time.view(B, 1, C, T, M).expand(B, S, C, T, M)
        # Apply time masks
        y_time = m_time * x_time

        # Frequency-domain mask estimation -> shape [B, T, S*C*F], then rearrange
        m_spec = self.spec_mask_fc(y_spec)
        S, C, F = self.num_sources, self.num_channels, self.spec_feature_size
        m_spec = m_spec.view(B, T, S, C, F)         # [B, T, S, C, F]
        m_spec = m_spec.permute(0, 2, 3, 1, 4)      # [B, S, C, T, F]

        # Expand x_spec for all sources, shape [B, S, C, T, F]
        x_spec = x_spec.view(B, 1, C, T, F).expand(B, S, C, T, F)
        # Apply frequency masks
        y_spec = m_spec * x_spec

        # Decode time-domain frames -> shape [(B*S*C), T, M] -> [B*S*C, L]
        y_time = y_time.reshape(B * S * C, T, M)
        x_norm = x_norm.view(B, 1, C, T, 1).expand(B, S, C, T, 1).reshape(B * S * C, T, 1)
        z_time = self.time_decoder(y_time, x_norm)  # [B*S*C, L]
        z_time = z_time.view(B, S, C, -1)           # [B, S, C, L]

        # Decode frequency-domain frames -> shape [(B*S*C), T, F] -> [B*S*C, L]
        y_spec = y_spec.reshape(B * S * C, T, F)
        x_angl = x_angl.view(B, 1, C, T, F).expand(B, S, C, T, F).reshape(B * S * C, T, F)
        z_spec = self.spec_decoder(y_spec, x_angl)  # [B*S*C, L]
        z_spec = z_spec.view(B, S, C, -1)           # [B, S, C, L]

        # Sum time + freq domain outputs (Section 3.3: see Fig. 1 in the paper)
        out = z_time + z_spec  # [B, S, C, L]

        # Optional zero-padding to match 'length' if specified
        if length:
            L_out = out.size(-1)
            out = ff.pad(out, (0, length - L_out), 'constant')

        return out.contiguous()  # Final shape [B, S, C, L]

    def _init_args_kwargs(self):
        # Return model constructor args/kwargs for easy serialization
        args = [self.num_sources, self.num_channels]
        kwargs = {
            'time_win_size': self.time_win_size,
            'time_hop_size': self.time_hop_size,
            'time_ftr_size': self.time_ftr_size,
            'spec_win_size': self.spec_win_size,
            'spec_hop_size': self.spec_hop_size,
            'spec_fft_size': self.spec_fft_size,
            'rnn_hidden_size': self.rnn_hidden_size,
        }
        return args, kwargs

    def serialize(self):
        """ Packages model class + arguments + weights. 
            Refer Section 5.2 in the paper (Implementation details). """
        klass = self.__class__
        args, kwargs = self._init_args_kwargs()
        state_dict = self.state_dict()
        package = {
            'klass': klass,
            'args': args,
            'kwargs': kwargs,
            'state_dict': state_dict,
        }
        return package

    def save_to_path(self, model_path):
        """ Saves the entire model to disk. 
            The paper references storing model weights for runtime usage. """
        model_package = self.serialize()
        torch.save(model_package, model_path)
        return model_path

if __name__ == '__main__':
    DEVICE = torch.device('cpu')  # Using CPU in this example
    B, C, L, S = 10, 2, 100000, 4  # Example: batch=10, 2-channels, length=100k, 4 sources
    x = torch.randn(B, C, L, device=DEVICE)  # Random input
    print(f'{x.size() = }')  # Debug print

    # Instantiate the model, as described in Section 3 for real-time separation
    model = HSTasNet(
        num_sources=S,
        num_channels=C,
        time_win_size=1024,
        time_hop_size=512,
        time_ftr_size=200,
        spec_win_size=1024,
        spec_hop_size=512,
        spec_fft_size=1024,
        rnn_hidden_size=500,
        rnn_num_layers=1,
        device=DEVICE,
    )

    y = model(x, length=L)  # Forward pass with optional length
    print(f'{y.size() = }')  # Outputs [B, S, C, L]