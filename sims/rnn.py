
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from torch import Tensor

class EmbeddingNet(torch.nn.Module):
    """ Recurrent neural network for SBI embedding """

    def __init__(
        self, in_channels: int, out_channels: int,
        hidden_features: int = 64, num_layers: int = 1
    ):
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels for the recurrent neural network
        out_channels: int
            Number of output channels for the recurrent neural network
        hidden_features: int
            Number of hidden features for the recurrent neural network
        num_layers: int
            Number of recurrent layers
        """
        super(EmbeddingNet, self).__init__()

        # Create recurrent layers
        self.rnn = torch.nn.ModuleList()
        for i in range(num_layers):
            n_in = in_channels if i==0 else hidden_features
            n_out = hidden_features
            self.rnn.append(
                torch.nn.GRU(n_in, n_out, batch_first=True))
        
        # Activation function
        self.activation = F.relu

    def forward(
            self, x: Tensor, h0: Optional[Tuple] = None,
            return_h: bool= False) -> Tuple[Tensor, Tuple]:
        """ Forward pass

        Parameters
        ----------
        x: Tensor
            Input tensor of shape (batch_size, seq_len, in_channels)
        h0: Optional[Tuple]
            Tuple of initial hidden states to pass into each recurrent layer
            If not specified, the hidden states are initialized to zero
            Has shape (num_layers, batch_size, hidden_features)
        return_h: bool
            Whether to return the hidden states
        """
        hout = []   # list of output hidden states

        # iterate over all recurrent layers
        for i in range(len(self.rnn)):
            x, h = self.rnn[i](x, h0[i] if h0 is not None else None)
            if i != len(self.rnn) - 1:
                x = self.activation(x)
            hout.append(h)

        # return only the last output of the sequence
        # return the hidden states if specified
        if return_h:
            return x[:, -1], hout
        else:
            return x[:, -1]

# # test the embedding net, comment in to run
# # define RNN parameters
# n_in = 2  # number of input channels
# n_out = 10  # number of output channels
# num_layers = 2  # number of recurrent layers
# hidden_features = 8  # number of hidden features

# # other parameters
# batch_size = 16  # number of samples in a batch
# seq_len = 100  # length of the sequence

# x = torch.randn(batch_size, seq_len, n_in)
# embedding_net = EmbeddingNet(n_in, n_out, hidden_features, num_layers)
# xout, hout = embedding_net(x, return_h=True)