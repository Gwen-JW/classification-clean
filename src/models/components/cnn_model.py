#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import torch.nn as nn

FILEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILEPATH))
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


class CnnModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        cell_array,
        kernel_size,
        dropout=0,
        act_fn_name="relu",
        **kwargs
    ):
        super().__init__()

        # self.length_sequence, self.n_features = input_size
        self.n_features, self.length_sequence= input_size
        self.output_size = output_size
        self.act_fn = act_fn_by_name[act_fn_name]

        self.filter_arrays = cell_array
        self.dropout = dropout
        self.kernel_size = kernel_size

        self._create_network()

    def _create_network(self):
        blocks = []
        kernel_size = self.kernel_size
        dropout = self.dropout
        for idx, filters in enumerate(self.filter_arrays):
            if idx == 0:
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=self.n_features,
                            out_channels=filters,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                        self.act_fn(),
                        nn.MaxPool1d(kernel_size=1),
                        nn.Dropout(p=dropout),
                    )
                )

            else:
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=self.filter_arrays[idx - 1],
                            out_channels=filters,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                        self.act_fn(),
                        nn.MaxPool1d(kernel_size=1),
                        nn.Dropout(p=dropout),
                    )
                )

        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool1d((1)),
            nn.Flatten(),
            nn.Linear(self.filter_arrays[-1], self.output_size),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.output_net(x)

        return x
