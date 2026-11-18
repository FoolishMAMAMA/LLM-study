import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = torch.mean(x**2,dim=-1,keepdim=True)#最后一维没有消失，只是从长度 4 变成了长度 1。
        rms = torch.sqrt(mean_square+self.eps)

        result = (x/rms) * self.g
    # x 的形状：(2, 3, 4)
    # g 的形状：      (4,)
        return result.to(in_dtype)
