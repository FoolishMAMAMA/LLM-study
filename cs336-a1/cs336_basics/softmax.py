import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿 dim 对每组分数做 Softmax，返回同形状的归一化权重。

    输入：
        x：浮点张量，里面装着待转换的分数。
        dim：分组所沿的轴；下标从 0 开始，-1 表示最后一维。
            例如 x 的形状为 (2, 3)：
            dim=1 或 -1：每行的 3 个数作为一组。
            dim=0：每列的 2 个数作为一组。

    输出：
        形状和数据类型与输入相同的张量。
        每组输出为非负权重，沿 dim 相加约等于 1（浮点计算存在舍入误差）。

    计算路线：减去每组最大值 -> 求指数 -> 除以每组指数值的总和。
    整组减去同一个数不改变 Softmax 结果，同时避免指数过大导致溢出。
    这是普通函数，没有可学习参数，也不需要保存缓存表。
    """
    # x.dtype 是 PyTorch Tensor 的属性，记录输入的数据类型，供最后恢复。
    # intype 是这里自定义的变量名，存的是 dtype，并不是指整数类型。
    intype =  x.dtype

    # Tensor.to 是 PyTorch 张量的方法；这里转换数据类型，不改变设备。
    # 先统一用 float32 计算；若输入为 float64，这一步也会先降低精度。
    x = x.to(torch.float32)

    # torch.amax 是 PyTorch 函数，返回沿 dim 取最大值后的张量。
    # keepdim=True：保留被处理的维度，但把该维度的长度变成 1。
    # 例如 x 为 (2, 3)、dim=-1，max_value 就是 (2, 1)：每行一个最大值。
    max_value = torch.amax(x,dim=dim,keepdim=True)

    # 逐元素相减，利用广播让每组减去自己的最大值。
    # 例如 (2, 3) - (2, 1) -> (2, 3)，每行的三个数共用该行最大值。
    new_x = x-max_value

    # torch.exp 是 PyTorch 函数，对每个元素 z 计算 e**z，形状不变。
    # epx 是指数结果组成的张量，也是最终相除时的分子。
    epx = torch.exp(new_x)

    # torch.sum 是 PyTorch 函数，沿同一个 dim 对每组指数值求和。
    # 这里的 torch.exp(new_x) 与上面的 epx 相同，求和对象与分子一致。
    # keepdim=True 便于后续广播；例如 (2, 3) -> (2, 1)。
    denominator = torch.sum(epx,dim=dim,keepdim=True)

    # / 是逐元素除法：每组的指数值除以这一组自己的总和。
    # 例如 (2, 3) / (2, 1) -> (2, 3)，因此输出与输入形状相同。
    # 每组内的数值相加约等于 1。
    result = epx/denominator

    # 用 Tensor.to 将计算结果转回输入的数据类型，然后返回张量。
    return result.to(intype)
