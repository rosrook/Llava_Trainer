# LLaVA v1.5 7B：本地 Hugging Face 目录的加载与推理（自洽说明）

本文**只依赖本页内容**即可理解流程；不要求再去打开仓库里其它 `.py` / `.md` 文件对照阅读。运行时代码来源见第二节：**可以**在完整 LLaVA 仓库里 `pip install -e .`，**也可以**只拷贝本仓库提供的可移植子包 `llava_v15_portable/`（不依赖「整棵 LLaVA 工程」安装）。

---

## 一、你手里的目录是什么

若你通过类似下面的流程，把 `liuhaotian/llava-v1.5-7b` **导出成单个本地文件夹**，则该文件夹通常包含：

- `config.json`、权重文件（如 `model-*.safetensors` 或 `pytorch_model.bin`）
- 分词器相关文件（`tokenizer.model`、`tokenizer_config.json` 等）
- 视觉侧用的图像处理器（与 CLIP ViT 配套，随模型一并保存）

含义可以概括为：**多模态 LLaVA 的完整 checkpoint**，格式与 Hugging Face 兼容，便于离线拷贝。导出时一般会调用「加载 Hub 上的完整模型 → 再 `save_pretrained` 到本地路径」这一类逻辑；你只需记住：**本地路径 = 之后推理时传入的 `model_path`**。

---

## 二、代码从哪里来（二选一）

LLaVA 的架构（例如 `LlavaLlamaForCausalLM`）**不在**纯 `transformers` 标准模型列表里默认注册；**仅拷贝权重目录不够**，必须额外有一份**模型实现 Python 代码**（与权重配套）。实现方式如下。

### 方式 A：在官方 LLaVA 仓库里安装包

在完整 LLaVA 仓库根目录执行 `pip install -e .`，则全局存在 `llava` 包，下文 `from llava.... import ...` 直接可用。

### 方式 B：可移植子包（适合「不能把整个 LLaVA 工程拷过去」的场景）

本仓库在 **`llava_v15_portable/`**（与 `Llava/LLaVA` 目录平级）下提供**推理所需子集**：内含精简后的 `llava/` 包、`example_infer.py`、`requirements.txt`、`README.txt`。

1. 将 **`llava_v15_portable` 整个文件夹**复制到你的目标工程任意位置。  
2. 让 Python 能 import 到其中的 `llava` 包：把 **`llava_v15_portable` 所在目录**（即包含 `llava` 子文件夹的那一层）加入 `sys.path`，或设置 `export PYTHONPATH="/path/to/llava_v15_portable:$PYTHONPATH"`。`example_infer.py` 开头已给出示例写法。  
3. `pip install -r requirements.txt`（见该目录内列表）。  
4. 运行：`python example_infer.py --model <本地HF模型目录> --image <图片路径>`。

方式 B 下**不需要**在目标工程里再 `pip install` 整份官方 LLaVA 仓库；也不要求拷贝 `Llava/LLaVA` 下除该子包以外的脚本。代码版权与上游 LLaVA 一致（Apache-2.0），请保留各文件头声明。

**推理常见依赖**（两种方式相同）：`torch`、`transformers`、`accelerate`、`sentencepiece`、`pillow` 等，以 `llava_v15_portable/requirements.txt` 为准。

---

## 三、最短示例：加载本地模型文件夹并调用（复制可运行）

下面是一段**尽量短**的完整脚本：把 `MODEL_DIR` 换成你导出的目录路径，把 `IMAGE_PATH` 换成图片路径即可运行。依赖第二节：**方式 A** 已安装 `llava` 包，或 **方式 B** 已把 `llava_v15_portable` 加入 `PYTHONPATH`（二者 import 路径相同）。若只想快速验证，可直接运行 **方式 B** 自带的 `llava_v15_portable/example_infer.py`。

逻辑顺序即：**加载 → 按 v1.5 拼 prompt → 图像张量 + `input_ids` → `generate` → 只解码新生成 token**。

```python
import torch
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

MODEL_DIR = "/path/to/your/exported_folder"  # 本地 HF 目录
IMAGE_PATH = "/path/to/image.jpg"
USER_QUESTION = "简要描述这张图片。"

disable_torch_init()
tokenizer, model, image_processor, _ = load_pretrained_model(
    MODEL_DIR, None, get_model_name_from_path(MODEL_DIR)
)

if model.config.mm_use_im_start_end:
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + USER_QUESTION
else:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + USER_QUESTION
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

image = Image.open(IMAGE_PATH).convert("RGB")
image_sizes = [image.size]
images_tensor = process_images([image], image_processor, model.config).to(
    model.device, dtype=torch.float16
)
input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .to(model.device)
)

with torch.inference_mode():
    out_ids = model.generate(
        input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=512,
        use_cache=True,
    )
text = tokenizer.decode(out_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
print(text)
```

若你的模型目录名不含 `v1`、且你改用别的加载方式导致对话模板选错，请确认上面 **`conv_templates["llava_v1"]`** 未被改成其它名字（见第五节）。

---

## 四、推理在做什么（四步，与是否本地目录无关）

对 **LLaVA v1.5**，一次「单图 + 一句问题」的推理可以拆成：

1. **加载**  
   用 LLaVA 提供的统一入口 `load_pretrained_model(model_path, ...)`，从**你的本地目录**加载：语言模型 + 视觉塔 + 分词器 + 图像处理器。  
   `model_path` 填导出得到的文件夹路径即可（与从 Hub 拉取相比，只是路径从 `liuhaotian/llava-v1.5-7b` 换成了磁盘路径）。

2. **拼「带图像占位」的文本**  
   模型在文本里用占位符表示图像位置。常见约定包括：
   - 占位字符串 **`<image>`**（与单张图对齐）
   - 若配置里开启 `mm_use_im_start_end`，还会在 `<image>` 外包一层 `<im_start>…<im_end>`（具体符号名由常量给出，见第六节）
   用户问题前要接上这些 token，再交给**对话模板**格式化成一整段 prompt 字符串。

3. **图像 → 张量**  
   用加载得到的 `image_processor` 把 PIL 图像变成模型需要的 `pixel_values`（形状、归一化与训练时一致）。  
   同时需要记录每张图的 `(宽, 高)`，生成时作为 `image_sizes` 传入——多尺度 / 宽高信息会参与后续计算。

4. **文本 → `input_ids`，并把 `<image>` 换成特殊下标**  
   分词器先把大部分文本编成 token，但 **`<image>` 位置不能当普通子词**：要换成一个专用的整数下标（LLaVA 里常用 **`-200`**，在代码里常命名为 `IMAGE_TOKEN_INDEX`）。  
   这样前向时模型才能把「这一位」和视觉特征对齐。然后调用 `model.generate(...)`，传入 `input_ids`、`images`（以及 `image_sizes` 等），得到输出 token，再**只解码新生成部分**为自然语言（避免把 prompt 又 decode 一遍）。

下面第七节、第八节给出与上述步骤对应的**命令行与更长 Python 示例**。

---

## 五、对话模板 `llava_v1`（v1.5 必用）

LLaVA 不同版本用的「系统提示 + 角色名 + 分隔符」不同，统称为 **conversation / conv_mode**。

- **LLaVA v1.5** 应使用 **`llava_v1`** 模板。  
- 若某命令行工具根据「模型目录名的最后一段」自动猜模板，而你的文件夹叫 `base_model` 这类**不含 `v1` 的名字**，可能误选旧模板；此时务必**手动指定** `llava_v1`，否则输出会不正常。

这一条与路径是否在 Hub 无关，**本地导出目录同样适用**。

---

## 六、常量约定（读懂代码片段用）

下列数值与字符串来自 LLaVA 惯例，**在本文中写清**，便于你单看文档就明白 import 进来的名字表示什么：

| 含义 | 典型值 / 说明 |
|------|----------------|
| 图像在词表中的占位下标 | `IMAGE_TOKEN_INDEX = -200`（与 `generate` 内部对齐视觉特征） |
| 单图占位字符串 | `DEFAULT_IMAGE_TOKEN = "<image>"` |
| 可选的图像起止包裹 | `DEFAULT_IM_START_TOKEN = "<im_start>"`，`DEFAULT_IM_END_TOKEN = "<im_end>"` |
| 用户文案里可写的占位 | `IMAGE_PLACEHOLDER = "<image-placeholder>"`（若出现则替换为上面某一种形式） |

配置项 `mm_use_im_start_end`：为真时，往往在 `<image>` 外再包 `<im_start>…<im_end>`；为假时可能只用 `<image>`。以你加载的 `model.config` 为准。第三节最短示例里已按该配置二选一拼接 `qs`。

---

## 七、方式 A：命令行（最少手写代码）

在已 `pip install -e .` 的环境中，可用模块方式调用官方入口（模块名以你安装的包为准，常见为 `llava.eval.run_llava`）。

```bash
cd /path/to/LLaVA   # 安装过 llava 包的环境即可，不必与权重同目录

python -m llava.eval.run_llava \
  --model-path "/path/to/your/exported_folder" \
  --image-file "/path/to/image.jpg" \
  --query "Describe this image in detail." \
  --temperature 0.2 \
  --max-new-tokens 512
```

若自动推断的对话模板不对，加上（**v1.5 推荐始终核对**）：

```bash
  --conv-mode llava_v1
```

**设备**：部分官方脚本默认把 `input_ids` 放到 CUDA；若无 GPU，需改用与 `model.device` 一致的设备，或参考方式 B 与第三节自行放置张量。

---

## 八、方式 B：Python 脚本（拆成函数 + 仅解码新生成）

下面示例**自洽**：展示从加载到 `generate` 的调用顺序；`from llava...` 仅依赖已安装的 `llava` 包，不假设读者打开仓库其它源文件。

说明两点实现细节：

- **`disable_torch_init()`**：在部分官方脚本里用于避免按大矩阵默认初始化以加快加载；若你的环境没有该工具函数，可删去这两行，一般不影响正确性，仅可能略慢。
- **`get_model_name_from_path`**：仅取路径最后一段名字，供少数分支判断用；若你固定使用 v1.5，可直接把对话模式写死为 `llava_v1`，不必依赖该函数推断。

```python
import re
from typing import Optional

import torch
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


def load_local_llava(model_path: str, model_base=None, device_map="auto"):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name,
        device_map=device_map,
    )
    return tokenizer, model, image_processor, context_len


def build_prompt_llava_v15(model, user_query: str, conv_mode: str = "llava_v1") -> str:
    """在问题前插入图像占位，再套对话模板，得到送入模型的完整字符串。"""
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in user_query:
        if model.config.mm_use_im_start_end:
            user_query = re.sub(IMAGE_PLACEHOLDER, image_token_se, user_query)
        else:
            user_query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, user_query)
    else:
        if model.config.mm_use_im_start_end:
            user_query = image_token_se + "\n" + user_query
        else:
            user_query = DEFAULT_IMAGE_TOKEN + "\n" + user_query

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], user_query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    num_beams: int = 1,
) -> str:
    image_sizes = [image.size]
    images_tensor = process_images(
        [image], image_processor, model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)
    input_ids = input_ids.to(model.device)

    output_ids = model.generate(
        input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    new_tokens = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    MODEL_DIR = "/path/to/your/exported_folder"
    IMAGE_PATH = "/path/to/image.jpg"

    tokenizer, model, image_processor, _ = load_local_llava(MODEL_DIR)
    conv_mode = "llava_v1"

    image = Image.open(IMAGE_PATH).convert("RGB")
    prompt = build_prompt_llava_v15(model, "What is happening in this image?", conv_mode)
    print(generate_answer(model, tokenizer, image_processor, image, prompt))
```

---

## 九、`load_pretrained_model` 常用可选参数（文内说明，不跳转源码）

该函数在 LLaVA 里封装了「分词器 + 多模态模型 + 视觉塔懒加载 + 图像处理器」；你可按需传入（名称以当前版本为准）：

- **`load_8bit` / `load_4bit`**：量化加载，减轻显存（需 `bitsandbytes` 等环境）。
- **`use_flash_attn`**：使用 Flash Attention 2（需硬件与库支持）。
- **`device_map`**：例如 `"auto"` 或多卡映射；非 CUDA 时需显式设为 CPU 等，并与张量 `to(device)` 一致。

本地目录与 Hub 模型在 API 上**仅 `model_path` 不同**，其余参数含义相同。

---

## 十、常见问题（仍不依赖其它文档）

1. **提示找不到 `LlavaLlamaForCausalLM` 或 `llava`**  
   若用方式 A：在对应环境执行 `pip install -e .`（官方 LLaVA 根目录）。若用方式 B：检查 `PYTHONPATH` / `sys.path` 是否包含 **`llava_v15_portable` 的父目录**（使 `import llava` 解析到拷贝过去的包），且目标环境已安装 `torch` 等依赖。

2. **输出乱、不像对话**  
   优先检查 **`conv_mode` 是否为 `llava_v1`**（v1.5），以及 `mm_use_im_start_end` 与占位符是否一致。

3. **显存不够**  
   打开 4bit/8bit 或减小 `max_new_tokens`。

4. **离线机器**  
   除权重目录外，仍需一份模型实现代码：要么完整 LLaVA 源码（方式 A），要么 **`llava_v15_portable` 整夹拷贝**（方式 B）。仅权重无法用纯 `transformers` 标准类名单独加载该自定义架构。

---

读完以上章节，无需再打开本仓库其它文件即可理解：**本地 HF 目录即 `model_path`**、**v1.5 用 `llava_v1`**、**`<image>` 要换成 `IMAGE_TOKEN_INDEX`**、**生成后只 decode 新增 token**。移植到其它工程时优先使用 **`llava_v15_portable/`** 即可避免安装整棵 LLaVA 仓库。若希望完全脱离任何 LLaVA 实现代码，只能自行实现多模态前向，已超出本文范围。
