# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn


class FCBlock(nn.Module):
    """
    A fully connected block with linear layer, batch normalization, and ReLU activation.

    This module applies a linear transformation, followed by batch normalization,
    and then a ReLU activation function.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.

    Attributes
    ----------
    linear : nn.Linear
        Linear transformation layer.
    bn : nn.BatchNorm1d
        Batch normalization layer.
    relu : nn.ReLU
        ReLU activation function.

    Examples
    --------
    >>> block = FCBlock(128, 64)
    >>> x = torch.randn(32, 128)
    >>> y = block(x)
    >>> y.shape
    torch.Size([32, 64])
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Initialize the FCBlock.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FCBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features).

        Notes
        -----
        The forward pass applies: ReLU(BatchNorm(Linear(x)))
        """
        return self.relu(self.bn(self.linear(x)))


class FCN(nn.Module):
    """
    Fully Connected Network with configurable depth and width.

    This model flattens the input sequence and processes it through a series
    of fully connected layers. It can optionally expand the sequence length
    using a linear transformation.

    Parameters
    ----------
    feature_channel : int
        Number of input features per time step.
    output_channel : int
        Number of output channels.
    num_layers : int
        Number of hidden layers.
    hidden_size : int
        Size of hidden layers.
    seq_length : int, optional
        Length of the input sequence. Default is 55.
    dim_expand : int, optional
        Number of time steps to expand the output sequence by.
        Default is 0 (no expansion).

    Attributes
    ----------
    feature_channel : int
        Number of input features.
    output_channel : int
        Number of output channels.
    seq_length : int
        Length of the input sequence.
    dim_expand : int
        Number of time steps to expand by.
    input_layer : FCBlock
        First fully connected layer.
    hidden_layers : nn.Sequential
        Stack of hidden layers.
    output_layer : nn.Linear
        Final output layer.
    dim_change : nn.Linear or None
        Optional layer for sequence length expansion.

    Examples
    --------
    >>> model = FCN(
    ...     feature_channel=6,
    ...     output_channel=4,
    ...     num_layers=3,
    ...     hidden_size=196,
    ...     seq_length=10
    ... )
    >>> x = torch.randn(32, 6, 10)
    >>> y = model(x)
    >>> y.shape
    torch.Size([32, 4, 10])
    """

    def __init__(
        self,
        feature_channel: int,
        output_channel: int,
        num_layers: int,
        hidden_size: int,
        seq_length: int = 55,
        dim_expand: int = 0,
    ) -> None:
        """
        Initialize the FCN model.

        Parameters
        ----------
        feature_channel : int
            Number of input features.
        output_channel : int
            Number of output channels.
        num_layers : int
            Number of hidden layers.
        hidden_size : int
            Size of hidden layers.
        seq_length : int, optional
            Length of the input sequence. Default is 55.
        dim_expand : int, optional
            Number of time steps to expand the output sequence by.
            Default is 0 (no expansion).

        Raises
        ------
        ValueError
            If num_layers is less than 1.
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.seq_length = seq_length
        self.dim_expand = dim_expand

        # Input layer: flattens feature_channel * seq_length to hidden_size
        self.input_layer = FCBlock(feature_channel * seq_length, hidden_size)

        # Hidden layers: stack of FCBlocks
        self.hidden_layers = nn.Sequential(
            *[FCBlock(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        # Output layer: projects hidden_size to output_channel * seq_length
        self.output_layer = nn.Linear(hidden_size, seq_length * output_channel)

        # Optional sequence length expansion
        self.dim_change = (
            nn.Linear(seq_length, seq_length + dim_expand) if dim_expand > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FCN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, feature_channel, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_channel, seq_length + dim_expand)
            if dim_expand > 0, otherwise (batch_size, output_channel, seq_length).

        Notes
        -----
        The forward pass:
        1. Flattens the input to (batch_size, feature_channel * seq_length)
        2. Passes through FCBlocks
        3. Projects to output dimensions
        4. Reshapes to (batch_size, output_channel, seq_length)
        5. Optionally expands sequence length
        """
        batch_size = x.size(0)

        # Flatten input: (batch, features, seq) -> (batch, features * seq)
        x = x.view(batch_size, -1)

        # Forward through FC layers
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        # Reshape to output format: (batch, output_channel, seq_length)
        x = x.view(batch_size, self.output_channel, -1)

        # Optional sequence length expansion
        if self.dim_change:
            x = self.dim_change(x.transpose(1, 2)).transpose(1, 2)

        return x
