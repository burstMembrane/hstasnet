import torch
import torch.nn as nn
import torch.nn.functional as ff

class Memory(nn.Module):
    """
    A GRU-based memory module (Section 3.2 of the paper mentions LSTM/GRU RNNs).
    This implementation uses two GRU layers in sequence, with a skip-connection
    around the second GRU’s output (similar to an identity residual).
    """

    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers=1,
                 dropout=0.0,
                 device=torch.device('cpu')):
        """
        Args:
          input_size  : Dimensionality of the input features
          hidden_size : Dimensionality of each GRU's hidden state
          num_layers  : Number of stacked GRU layers for each GRU block
          dropout     : Dropout probability between GRU layers
          device      : Torch device (CPU, GPU)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # First GRU block
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # x shape: (B, T, H_in)
            dropout=dropout,
            device=device
        )

        # Second GRU block, receives hidden_size from rnn1
        self.rnn2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            device=device
        )

    def forward(self, x):
        """
        Forward pass:
          x: [B, T, H_in]  (batch, time, input_size)
        Returns:
          z: [B, T, H_out] same shape except H_in -> hidden_size
        """
        # 1) Pass input through the first GRU
        y, _ = self.rnn1(x)  # shape [B, T, hidden_size]

        # 2) Pass the output of the first GRU into the second GRU
        z, _ = self.rnn2(y)  # shape [B, T, hidden_size]

        # 3) Add skip connection around the second GRU
        z = z + y            # Residual connection -> same shape [B, T, hidden_size]

        return z

if __name__ == '__main__':
    # Example usage
    B = 80      # batch size
    T = 4       # time steps
    H_in = 1500 # input_size
    H_out = 1000 # hidden_size

    # Random input: shape [B, T, H_in]
    x = torch.randn(B, T, H_in)
    print(f'Input shape x: {x.size()}')

    # Instantiate the Memory module
    memory = Memory(input_size=H_in, hidden_size=H_out)

    # Forward pass
    y = memory(x)
    print(f'Output shape y: {y.size()}')