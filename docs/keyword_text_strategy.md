# 关键词 + Encoder 策略

## 目标

这套策略保留当前项目的文本编码器和跨模态融合主干，只把原来的长篇专家文本替换成更短的结构化关键词提示。

当前流程：

`专家文本 -> tokenizer -> PubMedBERT + Mamba -> cross-attention -> decoder`

新流程：

`专家文本 -> 关键词提取器 -> 结构化关键词文本 -> tokenizer -> PubMedBERT + Mamba -> cross-attention -> decoder`

核心目的：

- 减少文本噪声
- 保留对医学分割有价值的条件信息
- 尽量不改动现有模型结构

## 为什么选择这个方向

本项目中的 TextBraTS 文本描述具有很强的模板化特征。大量样本都遵循相似的结构：

1. 病灶位置
2. 水肿描述
3. 坏死描述
4. 脑室受压描述

因为文本结构相对固定，所以规则法提取器是一个很合适的第一步：

- 比自由文本 NLP 更可控
- 更容易进行医学意义上的审核
- 更容易和原始文本做对照
- 可以直接兼容当前 text encoder

## 设计选择

第一阶段我们选择的是 **关键词 + encoder**，而不是 **关键词 + embedding**。

原因：

- 改动最小
- 保留当前 tokenizer 与文本编码器
- 可以和现有全文本基线做干净对比

这意味着当前实验只改变一个变量：

- 原始专家全文文本
- 结构化关键词文本

而其余模型结构保持不变。

## 提取设计思路

这版提取器不做开放式关键词生成，而是把原始文本转换成固定槽位的医学提示文本。

相比自由关键词挖掘，这种做法对医学分割任务更稳定。

### 当前升级后的字段

目前结构化 prompt 包含以下字段：

- `lesion_side`
- `lesion_lobes`
- `edema_status`
- `edema_extent`
- `edema_side`
- `edema_lobes`
- `necrosis_status`
- `necrosis_pattern`
- `necrosis_side`
- `necrosis_lobes`
- `ventricular_compression_status`
- `ventricular_compression_extent`

## 示例

原始文本：

```text
The lesion area is in the right frontal and parietal lobes with a mixed pattern of high and low signals with speckled high signal regions. Edema is mainly observed in the right parietal lobe, partially extending to the frontal lobe, presenting as high signal, indicating significant tissue swelling around the lesion. Necrosis is within the lesions of the right parietal and frontal lobes, appearing as mixed, with alternating high and low signal regions. Ventricular compression is seen in the lateral ventricles with significant compressive effects on the brain tissue and ventricles.
```

提取后的关键词 prompt：

```text
lesion_side: right; lesion_lobes: frontal,parietal; edema_status: present; edema_extent: significant; edema_side: right; edema_lobes: frontal,parietal; necrosis_status: present; necrosis_pattern: mixed; necrosis_side: right; necrosis_lobes: frontal,parietal; ventricular_compression_status: present; ventricular_compression_extent: significant
```

## 当前规则逻辑

实现位置：

[data/keyword_extractor.py](../data/keyword_extractor.py)

### 第一步：文本标准化

- 去掉多余空格
- 把换行替换为空格
- 在规则匹配前统一文本格式

### 第二步：按固定 marker 切分

提取器会寻找这些模板锚点：

- `The lesion area is in`
- `Edema is`
- `Necrosis is`
- `Ventricular compression is`

这样可以把整段报告切分成几个医学意义明确的片段。

### 第三步：提取病灶位置

对于病灶描述段，提取器会保留解剖位置部分，并在出现以下词时停止截取：

- `with`
- `showing`
- `exhibiting`

这样做的目的是：

- 保留位置
- 丢弃过多影像信号外观描述

### 第四步：提取侧别与脑叶标签

当前会标准化以下内容：

- 侧别：`left`、`right`、`bilateral`
- 脑叶：`frontal`、`parietal`、`temporal`、`occipital`、`insular`

### 第五步：提取状态

对于 edema、necrosis、ventricular compression，首先判断：

- `present`
- `absent`
- `unknown`

当前通过一些较保守的否定表达识别 `absent`，例如：

- `not observed`
- `not seen`
- `absent`
- `without`
- `no`

### 第六步：提取程度

提取器还会寻找以下程度词：

- `significant`
- `marked`
- `considerable`
- `extensive`
- `pronounced`
- `severe`
- `moderate`
- `notable`
- `mild`
- `slight`
- `partial`

如果没有检测到明确程度词，则返回 `unknown`。
如果同一段里出现多个程度词，则会按出现顺序保留多个标签，例如：

- `significant,extensive`
- `notable,significant`

### 第七步：针对坏死提取模式而不是硬提程度

对于坏死描述，当前版本不再强行提取 `necrosis_extent`，而是改为提取更符合原文风格的 `necrosis_pattern`。

当前会尝试识别：

- `clustered`
- `scattered`
- `patchy`
- `mixed`
- `central`
- `diffuse`
- `focal`

这样可以减少“原文写了坏死特征，但被提成 `unknown`”的情况。

### 第八步：当状态明确缺失时，用 `none` 替代附属字段

如果某个字段已经明确判断为 `absent`，那么相关附属字段不再返回 `unknown`，而是统一返回：

- `none`

例如：

- `necrosis_status: absent`
- `necrosis_pattern: none`
- `necrosis_side: none`
- `necrosis_lobes: none`

这样可以避免把“明确不存在”和“原文没说清楚”混在一起。

## 数据集接入方式

关键词模式通过数据集里的一个新开关控制，代码位置：

[data/brats_textbrats_dataset.py](../data/brats_textbrats_dataset.py)

并在配置文件中暴露：

[configs/textbrats.yaml](../configs/textbrats.yaml)

配置项如下：

```yaml
data:
  use_keyword_text: true
```

行为说明：

- `false`：使用原始专家全文文本
- `true`：在 tokenization 前先转换成结构化关键词文本

## 哪些部分保持不变

这套设计**有意不改变**以下部分：

- tokenizer
- PubMedBERT 文本编码器
- Mamba 文本适配层
- 图像 encoder-decoder
- 跨模态融合模块

唯一变化的是文本表示形式。

## 为什么这是一个有价值的实验

这套方案可以形成非常干净的对照：

1. `全文文本 + encoder`
2. `关键词文本 + encoder`

通过这个对照，可以回答几个重要问题：

- 全文叙述是否包含额外的有用信息
- 结构化 prompt 是否更容易被模型利用
- 更短的 prompt 是否能提高小样本条件下的稳定性

## 优势

- 架构改动最小
- 易于打开/关闭
- 医学可解释性更强
- 比自由文本条件输入更容易调试
- 是进一步研究 `关键词 + embedding` 的良好过渡版本

## 当前局限

- 规则法可能漏掉少数非常规表达
- 程度提取仍然比较浅
- 部分 case 的 `necrosis_pattern` 仍可能为 `unknown`
- 脑室相关信息还不够细粒度
- 类似 `partially extending to` 这类扩展关系还没有显式建模

## 下一步改进方向

后续可以考虑：

1. 在 dataset 中同时返回 `original_text` 与 `keyword_text`，方便调试
2. 增加更细的脑室字段，例如侧别或具体脑室类型
3. 增加病灶扩展与水肿扩展字段
4. 对解剖位置做更强的标准化
5. 将 `关键词 + encoder` 与 `关键词 + embedding` 做系统对比

## 实际使用方式

在配置文件中开启关键词模式：

[configs/textbrats.yaml](../configs/textbrats.yaml)

```yaml
data:
  use_keyword_text: true
```

然后像平时一样训练：

```bash
python train.py --config configs/textbrats.yaml --max-samples 10 --max-epochs 2
```

这样模型就会使用结构化关键词 prompt 进行训练，而不是使用原始专家全文文本。
