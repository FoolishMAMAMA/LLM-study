import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super().__init__()

        self.weight = nn.Parameter(#nn.Parameter(...)：把张量标记为可学习参数。
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )#empty 不代表填零，而是没有初始化数值。里面暂时可能是任意值，下一步必须给它设置初始值。
        #创建模块 → 分配权重空间 → 登记为可学习参数

        # 1. 为什么用 nn.Parameter，不直接用 Tensor？因为这个张量是模型需要学习的权重，我们希望 PyTorch 自动把它当作参数管理。
        # nn.Parameter 本身就是 Tensor 的一种特殊类型，因此仍然可以参与矩阵乘法等计算，但它多了“模型参数”的身份。

        std = (2/(in_features+out_features))**0.5
        #trunc_normal_：用截断正态分布的随机数填充张量。超出指定范围的样本会重新抽取，而不是直接压到边界值。末尾 _ 表示原地修改张量。
        nn.init.trunc_normal_(#trunc_normal_ 末尾的下划线表示原地修改。 它直接填充已有的 self.weight
            self.weight,
            mean=0,
            std=std,
            a=-3*std,
            b=3*std
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return x@ self.weight.T
