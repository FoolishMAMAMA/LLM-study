# A1 §2.7 Tokenizer 实验报告

取样：各文件前10篇非空文档，非随机抽样；正文保留空白，不含分隔符。
TinyStories 样本来自开发集，OpenWebText 样本来自训练集；结论仅代表这些样本。


## (d) 全量编码与 uint16

输出为小端 uint16 原始 NumPy 数据（.bin），保留原换行及文档分隔 token。
每批在特殊 token 后切分并校验解码还原；有界任务队列按原文顺序写盘。
中断后重跑同一命令，从已持久化的字节偏移继续；不要同时启动两个写同一输出的任务。

uint16 范围为0～65535，覆盖本实验10K/32K词表的全部ID；每ID仅2字节，比int64节省75%的数组存储空间。转换前检查整个词表ID范围。

|数据集|状态|已保存token数|累计编码秒数|
|---|---|---:|---:|
|tiny_valid|完成|5465883|1.8|
|tiny_train|完成|541229347|37.6|
|owt_valid|完成|66401098|9.7|
|owt_train|完成|2727120452|339.2|

读取示例：`np.memmap(path, dtype='<u2', mode='r')`；训练时抽取batch再转成模型所需整数类型，不要把整个数组复制到显存。

## 复现命令（WSL 项目目录）

```bash
.venv/bin/python -u -m cs336_basics.tokenizer_pipeline --measure
.venv/bin/python -u -m cs336_basics.tokenizer_pipeline --encode all --workers 4
```

Windows PowerShell 中在相同命令前添加 `wsl --exec`。
全量任务输出于本报告同目录；JSON 保存输入、模型签名和断点。
累计编码耗时包含多进程启动、编码、解码校验、数据传输与写盘，与(c)的纯encode时间统计口径不同。
