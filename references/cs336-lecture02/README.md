# CS336 Lecture 2 本地学习材料

来源：[官方2026录课版网页](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording)、[官方源码仓库](https://github.com/stanford-cs336/lectures)。获取日期：2026-10-21。原课程内容及插图归各原作者；本目录是学习用转换，不是官方发布的 Notebook。

## 打开哪个文件

- **`lecture_02_recording.ipynb`：主课件，可执行。** 按原录课源码的章节顺序组织；包含imports、辅助函数、71个真正的Python代码单元格，说明和全部9张插图放在Markdown中。去除了执行记录步号和旧变量快照，运行后显示本机结果。默认CPU；GPU和大规模实验需要显式开启。
- `lecture_02_practice.ipynb`：助手编写的CPU小规模交互练习，25个单元格。可修改代码并用 Shift+Enter 运行。不是原课件的完整可执行移植，也不复现GPU性能数据。
- `lecture_02_recording_source.py`：从原始trace提取的对应源码。
- `lecture_02.py`：下载时main分支的源码，与录课版有差异，不作为本Notebook转换来源。
- `lecture_02_recording.json`：原始执行记录；`manifest.json`记录来源、校验值和图片下载情况。

## VS Code / 兼容编辑器中的用法

两份Notebook都可逐格执行，需要PyTorch、einops、ipykernel；内核已安装在现有WSL虚拟环境中，不改变作业依赖声明或Windows Python。

1. 在WSL终端运行 `code /mnt/e/LLMStudy/LLM-study`，打开WSL工作区。
2. 打开本目录的 **`lecture_02_recording.ipynb`**。先运行导入、配置和辅助函数，再按章节向下运行。
3. 右上角选择内核 → Python环境 → 选择 `/mnt/e/LLMStudy/LLM-study/cs336-a1/.venv/bin/python`。若没列出，手动输入解释器路径。需要编辑器的Python/Jupyter扩展。
4. 按顺序使用 Shift+Enter；可修改形状、数据类型和输入观察输出。不用F5直接运行.py，也不需要在本地部署edtrace网页。

当前C3目标优先看主课件的 tensors basics、tensors memory、einops、deep linear network、gradients basics。也可使用practice的1～7节巩固。梯度累积/checkpointing可稍后补。

## 转换限制

第一版错误地把执行轨迹当成课件转换，旧版保存在 `archive/lecture_02_recording_trace_v1.ipynb`，不再作为学习入口。现在主课件来自原始源码的结构转换。练习版12个代码单元此前已执行验证。

- 主课件支持Jupyter逐格运行，不复刻官方网页的单步调试UI、变量悬浮提示或调用栈交互。原 `@inspect` 改为实际打印当前值和张量形状。
- 外部参考链接仍需网络。原图文使用英文；中文说明和CPU练习属于新增辅助材料。
- 主课件的适配/勘误：rank-3示例使用三维张量；两周按14天计算；GPT-3大张量使用meta设备；预测按样本取均值；修正梯度和AdaGrad状态字节数公式；checkpoint使用标量loss和非重入模式，默认缩小维度。均在相应章节注明，原始源码文件保持不变。
- GPU示例默认关闭；CPU和未录入规格的GPU不报告虚构的MFU，CPU显存测量显示NaN。只有开启相关配置后才执行原规模GPU实验。

如需**原版网页交互**，可以按官方仓库的edtrace说明在本地搭建前端；该路线与Notebook是两种选择，此次没有部署原版前端。

Validation (2026-10-21, v2): all 71 code cells executed successfully in WSL with the default CPU configuration. Outputs saved; no error outputs. Optional GPU branches were not executed.
