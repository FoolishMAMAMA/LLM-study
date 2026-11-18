import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()

        self.W = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )

        std = 1
        nn.init.trunc_normal_(
            self.W,
            mean=0,
            std=std,
            a=-3,
            b=3
        )

    # import torch

    # table = torch.tensor([
    #     [10.0, 11.0, 12.0],  # 第 0 行
    #     [20.0, 21.0, 22.0],  # 第 1 行
    #     [30.0, 31.0, 32.0],  # 第 2 行
    # ])
    ids = torch.tensor([2, 0])

    # print(table[ids])

    # ->tensor([
    #     [30., 31., 32.],  # 第 2 行
    #     [10., 11., 12.],  # 第 0 行
    # ])
    def forward(self,token_ids: torch.Tensor)->torch.Tensor:
       return self.W[token_ids]
