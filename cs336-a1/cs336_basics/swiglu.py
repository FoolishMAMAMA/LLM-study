import torch
from torch import nn
from cs336_basics.linear import Linear

def SiLU(x:torch.Tensor)->torch.Tensor:
        # SiLU 逐元素计算，不改变张量形状。
        return x*torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int|None = None, device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff:
            self.d_ff = d_ff
        else:
            self.d_ff = max(64,round(((8/3*d_model)/64))*64)

        # 按题目的列向量记法 W1 x：W1 权重应为 (d_ff, d_model)。
        # Linear 接收 (..., d_model)，返回 (..., d_ff)；转置由 Linear.forward 处理。
        self.W1 = Linear(self.d_model,self.d_ff,device,dtype)
        # W2 权重应为 (d_model, d_ff)：(..., d_ff) -> (..., d_model)。
        self.W2 = Linear(self.d_ff,self.d_model,device,dtype)
        # W3 与 W1 形状相同，但应当是独立的 Linear 对象。
        self.W3 = Linear(self.d_model,self.d_ff,device,dtype)




    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 用 B 表示批大小、T 表示序列长度，常见输入 x: (B, T, d_model)。
        # 数学中 W1 x 的权重 W1: (d_ff, d_model)；
        # 代码中的 Linear.forward 使用 x @ W1.T，得到 (B, T, d_ff)。
        # SiLU 逐元素作用，part1 的形状保持 (B, T, d_ff)。
        part1 = SiLU(self.W1(x))
        # W3 是另一套独立权重 (d_ff, d_model)；
        # x @ W3.T 的输出 part3 也是 (B, T, d_ff)。
        part3 = self.W3(x)
        # 两个同形状张量逐元素相乘，结果为 (B, T, d_ff)。
        # W2 权重为 (d_model, d_ff)，相乘结果 @ W2.T 后为 (B, T, d_model)。
        # 前面的 B、T 只是例子，实际可有任意数量的前置维度。
        result = self.W2(part1*part3)

        return result
