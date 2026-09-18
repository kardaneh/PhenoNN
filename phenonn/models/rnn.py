# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
from typing import Tuple, Union


class BaseRNN(nn.Module):
    """
    Base class for bidirectional RNN modules (LSTM/GRU).

    This class provides a common interface for both LSTM and GRU models
    with bidirectional processing and a final 1D convolutional layer
    to map the hidden states to the desired output channels.

    Parameters
    ----------
    feature_channel : int
        Number of input features per time step.
    output_channel : int
        Number of output channels (target variables).
    hidden_size : int
        Number of hidden units in the RNN layers.
    num_layers : int
        Number of stacked RNN layers.
    rnn_type : str
        Type of RNN cell, either 'lstm' or 'gru'.

    Attributes
    ----------
    rnn : nn.LSTM or nn.GRU
        The bidirectional RNN layer.
    final : nn.Conv1d
        Final 1D convolution to project hidden states to output channels.
    hidden_size : int
        Number of hidden units.
    num_layers : int
        Number of stacked layers.
    output_channel : int
        Number of output channels.

    Examples
    --------
    >>> model = BaseRNN(
    ...     feature_channel=6,
    ...     output_channel=4,
    ...     hidden_size=64,
    ...     num_layers=2,
    ...     rnn_type='lstm'
    ... )
    >>> x = torch.randn(32, 6, 10)  # (batch, features, sequence)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 4, 10])
    """

    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: str,
    ) -> None:
        """
        Initialize the BaseRNN module.

        Parameters
        ----------
        feature_channel : int
            Number of input features.
        output_channel : int
            Number of output channels.
        hidden_size : int
            Size of hidden state.
        num_layers : int
            Number of RNN layers.
        rnn_type : str
            Type of RNN ('lstm' or 'gru').

        Raises
        ------
        ValueError
            If rnn_type is not 'lstm' or 'gru'.
        """
        super().__init__()

        # Validate input
        if rnn_type not in ["lstm", "gru"]:
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type}")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channel = output_channel
        self.rnn_type = rnn_type

        # Select RNN class
        rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU

        # Create bidirectional RNN
        self.rnn = rnn_class(
            input_size=feature_channel,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # BBBBBidirectional
        )

        # Final projection layer
        self.final = nn.Conv1d(
            in_channels=hidden_size,  # bidirectional doubles hidden size
            out_channels=output_channel,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Initialize the hidden state for the RNN.

        Parameters
        ----------
        batch_size : int
            Batch size for the input.
        device : torch.device
            Device to create the hidden state on.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            For GRU: returns hidden state tensor of shape
            (2 * num_layers, batch_size, hidden_size)
            For LSTM: returns tuple (hidden, cell) both of same shape.
        """
        hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device,
            requires_grad=False,
        )

        if self.rnn_type == "lstm":
            return (hidden, hidden.clone())
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bidirectional RNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, feature_channel, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_channel, seq_length).

        Notes
        -----
        The input is permuted to (batch_size, seq_length, feature_channel)
        for the RNN, then the output is permuted back for the convolution.
        """
        # Permute to (batch, seq, features) for RNN
        x = x.permute(0, 2, 1)

        # Initialize hidden state
        hidden = self.init_hidden(x.size(0), x.device)

        # Forward through RNN
        out, _ = self.rnn(x, hidden)

        # Permute back to (batch, features, seq) for convolution
        out = out.permute(0, 2, 1)

        # Final projection
        return self.final(out)


class RNN_LSTM(BaseRNN):
    """
    LSTM-based bidirectional RNN model.

    This class inherits from BaseRNN and configures it to use LSTM cells.

    Parameters
    ----------
    feature_channel : int
        Number of input features.
    output_channel : int
        Number of output channels.
    hidden_size : int
        Size of hidden state.
    num_layers : int
        Number of LSTM layers.

    Examples
    --------
    >>> model = RNN_LSTM(
    ...     feature_channel=6,
    ...     output_channel=4,
    ...     hidden_size=128,
    ...     num_layers=3
    ... )
    >>> x = torch.randn(16, 6, 10)
    >>> y = model(x)
    >>> print(y.shape)
    torch.Size([16, 4, 10])
    """

    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        """
        Initialize the LSTM model.

        Parameters
        ----------
        feature_channel : int
            Number of input features.
        output_channel : int
            Number of output channels.
        hidden_size : int
            Size of hidden state.
        num_layers : int
            Number of LSTM layers.
        """
        super().__init__(
            feature_channel, output_channel, hidden_size, num_layers, "lstm"
        )


class RNN_GRU(BaseRNN):
    """
    GRU-based bidirectional RNN model.

    This class inherits from BaseRNN and configures it to use GRU cells.

    Parameters
    ----------
    feature_channel : int
        Number of input features.
    output_channel : int
        Number of output channels.
    hidden_size : int
        Size of hidden state.
    num_layers : int
        Number of GRU layers.

    Examples
    --------
    >>> model = RNN_GRU(
    ...     feature_channel=6,
    ...     output_channel=4,
    ...     hidden_size=128,
    ...     num_layers=3
    ... )
    >>> x = torch.randn(16, 6, 10)
    >>> y = model(x)
    >>> print(y.shape)
    torch.Size([16, 4, 10])
    """

    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        """
        Initialize the GRU model.

        Parameters
        ----------
        feature_channel : int
            Number of input features.
        output_channel : int
            Number of output channels.
        hidden_size : int
            Size of hidden state.
        num_layers : int
            Number of GRU layers.
        """
        super().__init__(
            feature_channel, output_channel, hidden_size, num_layers, "gru"
        )
