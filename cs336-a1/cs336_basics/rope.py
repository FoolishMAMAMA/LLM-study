import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """旋转 Q 或 K 向量，输出形状与输入相同；没有可学习参数。

    使用方式：rotated_x = rope(x, token_positions)。
    nn.Module 的调用机制会将这次调用交给下面的 forward 方法。

    x：浮点特征张量，形状 (..., T, d_k)，例如 (B, T, d_k)。
    token_positions：整数张量，给出本次输入的实际 token 序号。
        常见形状是 (T,)（各 batch 共用序号）或 (B, T)。
        序号应满足 0 <= 序号 < max_seq_len；查表结果应能与 x 的前置维度广播。
    模块与 x 应放在同一设备上。

    初始化：特征对编号 -> 分母的倒数 -> 完整角度 -> 保存 cos、sin 表。
    forward：按 token 序号查表 -> 旋转每对特征 -> 恢复原来的特征排列。
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        # super 是 Python 的内置机制；这里调用父类 nn.Module 的初始化方法，
        # 准备好参数、buffer 等登记信息，后面才能使用 register_buffer。
        super().__init__()

        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError(f"d_k must be a positive even integer for RoPE, got {d_k}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        # theta 对应教材大写 Θ（基数），不是实际旋转角度。
        # d_k 对应教材 d，即整个 Q/K 向量的特征数。
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # torch.arange 是 PyTorch 的函数；arange(n) 生成 0 到 n-1 的张量。
        # 每两个相邻特征组成一对，因此共有 d_k // 2 对。
        # j = [0, 1, ..., d_k // 2 - 1]，形状 (d_k // 2,)。
        # dtype 指定数值类型，device 指定在哪个设备上创建张量。
        pair_indices = torch.arange(
            d_k // 2,
            dtype=torch.float32,
            device=device,
        )

        # 教材：θ_{i,k} = i / Θ ** ((2k - 2) / d)，教材 k 从 1 开始。
        # 代码 j 从 0 开始，所以 j = k - 1，2k - 2 = 2j。
        # inv_freq = Θ ** (-2j / d)，是教材分母的倒数，还不是完整角度。
        # 例如 theta=10000、d_k=4：inv_freq = [1, 0.01]。
        # 形状：(d_k // 2,)。
        inv_freq = self.theta ** (-2 * pair_indices / self.d_k)

        # 提前准备全部可用的 token 序号 i：[0, ..., max_seq_len - 1]。
        # 这里的局部变量用于浮点计算，不是 forward 中用于查表的整数参数。
        # 形状：(max_seq_len,)。
        token_positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
            device=device,
        )

        # 完整角度 = i * inv_freq；这里是乘法，因为已经取过分母的倒数！
        # None 在指定位置增加一个长度为 1 的维度：
        # token_positions[:, None]：(max_seq_len, 1)。
        # inv_freq[None, :]：       (1, d_k // 2)。
        # 广播使每个 i 分别乘上每个特征对的 inv_freq。
        # angles 形状：(max_seq_len, d_k // 2)，单位为弧度。
        angles = token_positions[:, None] * inv_freq[None, :]

        # torch.cos / torch.sin 对 angles 的每个数计算三角函数，形状不变。
        # 一对特征需要两个系数：一个 cos 值、一个 sin 值，分别存入两张表。
        # 两张表均为 (max_seq_len, d_k // 2)：行 = token 序号，列 = 特征对编号。
        # self.cos_cache[3]：token 序号 3 的所有特征对所用的 cos 值。
        # self.cos_cache[3, 0]：token 序号 3 中，(x_0, x_1) 所用的一个 cos 值。
        # self.register_buffer 是从父类 nn.Module 继承的方法，返回 None：
        #   第一个参数：登记名，登记后可用 self.cos_cache / self.sin_cache 访问。
        #   第二个参数：要交给模块管理的 Tensor。
        #   persistent=False：不写进 state_dict()，但仍保存在模块中供反复使用。
        # buffer 会随模块 .to(device) 移动，不出现在模块 .parameters() 中。
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Tensor.dtype 是张量的类型属性；Tensor.to 是 PyTorch 张量的方法。
        # .to(torch.float32) 得到 float32 类型的张量，用于下面的计算。
        # 最后再把结果转回输入类型；这里没有改变设备。
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # token_positions 是调用方传入的整数序号，通常使用 torch.long。
        # 张量索引 [...] 根据这些序号选取缓存表的行：
        #   序号形状 (T,)    -> cos、sin 形状 (T, d_k // 2)。
        #   序号形状 (B, T) -> cos、sin 形状 (B, T, d_k // 2)。
        # 例如序号 [1, 3] 会选出第 1、3 行；不会重新假设序号从 0 开始。
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # Python 切片格式是 start:stop:step，省略 stop 表示一直取到末尾。
        # ... 是 Ellipsis，这里保留前面的维度，只对最后一维的特征进行切片。
        # [a, b, c, d] -> x_even=[a, c]，x_odd=[b, d]。
        # 二者形状均为 (..., T, d_k // 2)；不会跨 token 配对。
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # * 是逐元素乘法，可按广播规则使用查出的系数。
        # 每对 (a, b) 共用角度 α，使用 cos(α)、sin(α) 两个系数：
        #   a' = a*cos(α) - b*sin(α)
        #   b' = a*sin(α) + b*cos(α)
        # 得到 rotated_even=[a', c']、rotated_odd=[b', d']。
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # torch.stack 是 PyTorch 的函数，会增加一条维度。
        # dim=-1 表示把新维度放在最后，将每对结果放到一起：
        # [a', c'] 和 [b', d'] -> [[a', b'], [c', d']]。
        # 形状：(..., T, d_k // 2, 2)。
        paired = torch.stack((rotated_even, rotated_odd), dim=-1)

        # Tensor.flatten 是 PyTorch 张量的方法。
        # start_dim=-2 表示从倒数第二维起，合并到最后一维。
        # (..., T, d_k // 2, 2) -> (..., T, d_k)。
        # [[a', b'], [c', d']] -> [a', b', c', d']。
        # 数值及元素先后顺序不变，只合并维度；前面的 batch、token 维度保留。
        result = paired.flatten(start_dim=-2)

        return result.to(in_dtype)
