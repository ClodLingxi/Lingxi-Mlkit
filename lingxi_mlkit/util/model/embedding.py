from typing import Union

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

class AutoEmbedding(nn.Module):
    def __init__(self, discrete_flags: np.ndarray[int]):
        super().__init__()
        self.embedding_table = nn.ModuleList([])
        for feature_count in discrete_flags:
            if feature_count > 0:
                self.embedding_table.append(nn.Embedding(feature_count, self.get_embedding_dim(feature_count)))

        self.discrete_idx = np.argwhere(discrete_flags > 0).flatten()


    def forward(self, x: torch.Tensor, auto_split=False, auto_type=True, return_concat=True):
        if auto_split:
            x = x[:, self.discrete_idx]

        if auto_type:
            x = x.long()

        result = []
        for column_idx in range(x.shape[1]):
            result.append(
                self.embedding_table[column_idx](x[:, column_idx])
            )
        if return_concat:
            return torch.cat(result, dim=1)
        return result

    @staticmethod
    def get_embedding_dim(_feature_count):
        return 32






if __name__ == '__main__':
    pd_data = pd.DataFrame({
        "idx": [1, 2, 3, 4, 5],
        "feature_1": ['A', 'A', 'B', 'B', 'C'],
        "feature_2": ['Q', 'W', 'E', 'E', 'W'],
        "feature_3": [2.3, 4.2, 5.5, 2.2, 5.1],
        "label": [True, False, True, True, False],
    })

    print(AutoEmbedding.auto_handle_set(pd_data.to_numpy()))
