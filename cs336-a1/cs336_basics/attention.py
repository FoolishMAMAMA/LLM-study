import math

import torch

from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    dk = Q.shape[-1]
    K_T = K.transpose(-2,-1)

    QK = Q @ K_T
    value_matrix = QK/math.sqrt(dk)

    if mask is not None:
        condition = ~mask
        value_matrix = value_matrix.masked_fill(condition,float("-inf"))

    sm_value = softmax(value_matrix,-1)

    result = sm_value @ V
    return result
