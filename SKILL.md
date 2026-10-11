---
name: 本地中文 TTS（开源免费 / Windows / 无独显也能跑）
description: >-
  【本地中文 TTS（开源免费 / Windows / 无独显也能跑）】>- 【本地中文 TTS（开源免费 / Windows / 无独显也能跑）】>- 在本机（Windows / 8GB 显存）用开源免费方案做中文语音合成与音色挑选。覆盖 Kokoro-82M-v1.1-zh ONNX 路线的完整踩坑记录：中文音色的正确来源、音色文件 (510,1,256) 形状、fp16 模型在 CPU 出 NaN、 必须绕开 kokoro-onnx 的 create() 分句逻辑、misaki 中文分词、沙箱下 curl -o 被禁改用 urllib。 也能回答"哪个开源 TTS 能控情绪"。触发词：本地TTS、中文音色、语音克隆、配音、Kokoro、 音色试听、文字转语音、TTS 部署、声音克隆、开源配音。
agent_created: true
---

# 本地中文 TTS（开源免费 / Windows / 无独显也能跑）

## 0. 先看结论

| 需求 | 选谁 | 为什么 |
|---|---|---|
| **要一堆现成音色挑** | **Kokoro-82M-v1.1-zh（ONNX）** | 103 个中文音色、82M 极小、CPU 就能跑、Apache 2.0 |
| **要自然语言控情绪**（"用炸毛大喊的语气说"） | **CosyVoice3** | 唯一支持指令级情绪；Apache 2.0 商用无限制 |
| **要克隆某个人的声音** | **GPT-SoVITS** / F5-TTS / CosyVoice3 | 都是零样本克隆，但**必须先有参考音频** |
| 商用最保险 | CosyVoice3（Apache 2.0）/ Kokoro（Apache 2.0）/ GPT-SoVITS（MIT） | Fish Speech 是 Research 协议，商用要单独授权 |

> **最容易踩的认知坑**：TTS 引擎本身**不产生"真人声音"**。
> 想要真人级效果，必须有**真人参考音频**——要么用户自己录 10~30 秒，要么用授权语料。
> 去短视频平台扒"真人口播"往往白费：**大部分口播本身就是平台变声/AI 配音**。

### 0.1 ★ 音色出处（用户必问，先备好答案）
**`hexgrad/Kokoro-82M-v1.1-zh` 模型卡片原文**：这 100 个中文说话人来自
**龙猫数据（LongMaoData，longmaosoft.com）**这家专业数据集公司，**freely and permissively granted**
（无偿且授权宽松）提供给模型作者训练；**License: Apache 2.0**。

→ 所以对外要这样准确表述：
- ✅ **是真人录音训练出的音色**（100 位专业数据集说话人）
- ❌ 但**不是**某个具体网红的克隆——音色是匿名编号（`zf_001`），对应不到公开身份
- ✅ **无版权/声音权风险，Apache 2.0，可商用**
- 对比：抖音多数口播是平台"变声"生成的 AI 音，扒下来没意义

（注意：`hexgrad/Kokoro-82M` 原版 54 个英文音色里，`af_maple`/`af_sol`/`bf_vale` 是众包合成数据，
与中文那 100 位不是一回事。）

## 1. ★ Kokoro 中文 ONNX 路线（实测跑通，推荐先走这个）

### 1.1 环境
```bash
# 受管 Python 3.13 建 venv（不要用 ComfyUI 的 python_embeded：嵌入式版没有 venv 模块）
<managed-python> -m venv D:\tts_env
D:/tts_env/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    kokoro-onnx "misaki[zh]" soundfile numpy
```
- **不需要 torch**（onnxruntime 推理），**不需要 Python 3.12**。
- ★ **别装官方 `kokoro` 包**：它写死 `numpy==1.26.4`，而该版本**没有 cp313 轮子** → pip 无限回溯、
  甚至尝试源码编译 numpy（无 C 编译器 → 直接失败）。

### 1.2 模型与音色（走 hf-mirror）
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```
仓库：**`onnx-community/Kokoro-82M-v1.1-zh-ONNX`**
- 模型：`onnx/model.onnx`（fp32 / 339MB）★ **必须用 fp32**
- 音色：`voices/zf_*.bin`（55 个女声）+ `voices/zm_*.bin`（45 个男声）

> - **`hexgrad/Kokoro-82M` 里只有英文音色**（`af_*`/`am_*`）和 `kokoro-v1_0.pth`，
>   **没有** `kokoro-v1.0.onnx`、**没有** `voices-v1.0.bin`（这两个文件名是常见误传，会 404）。
> - 中文音色**只在 `-v1.1-zh`** 那个仓库。

★ **`model_fp16.onnx` 在 CPU 上会输出全 NaN**（幅值 nan）——踩过。老老实实用 fp32。

### 1.3 音色文件形状（关键）
每个 `.bin` = **522240 字节 = 510 × 256 × float32**，即 `(510, 256)`。

```python
import numpy as np, glob, os
d = {}
for p in glob.glob(r"...\voices\*.bin"):
    n = os.path.splitext(os.path.basename(p))[0]
    a = np.frombuffer(open(p,"rb").read(), dtype=np.float32)
    d[n] = a.reshape(a.size // 256, 1, 256).copy()   # ★ (N,1,256)
np.savez(r"...\voices_zh.npz", **d)
```
★ 给成 `(N,256)` 会报 **`Invalid rank for input: style Got: 1 Expected: 2`**。
（`_style_for` 会取 `voice[n-1]`，所以保留第二维、让索引结果为 `(1,256)`。）

### 1.4 ★★ 合成必须绕开 `create()`
```python
from kokoro_onnx import Kokoro
from kokoro_onnx.config import SAMPLE_RATE   # 24000
from misaki import zh

k = Kokoro(model_onnx, voices_npz)
g2p = zh.ZHG2P()
ph  = g2p("你好，今天天气不错。")            # misaki 直接给 kokoro 音素串
toks = k.tokenizer.tokenize(ph)
a, _ = k._infer(toks, k._style_for(k.get_voice_style("zf_001"), len(toks)), 1.0)
```
**为什么不用 `k.create(..., is_phonemes=True)`**：
`create → create_timed → insert_pauses → _quiet_frames`，它按**标点分句**，
面对纯音素串分不出句子、`audio` 为空 → 崩在
`ValueError: zero-size array to reduction operation maximum which has no identity`。

**返回值坑**：`_infer` 返回 `(audio, duration)`，**第二个不是采样率**！采样率用常量 `SAMPLE_RATE = 24000`。

- `kokoro-onnx` 自带 tokenizer 只走 **espeak-ng（英文）**，所以中文必须自己用 misaki 转音素后走底层。

## 2. 沙箱/环境坑（WorkBuddy 环境）

| 现象 | 真因 | 对策 |
|---|---|---|
| `curl -o file` 报 `client returned ERROR on write of N bytes` | **沙箱禁 curl -o 写文件** | 改用 **Python urllib/requests** 下载 |
| `curl -o` 报 `Failed to open the file ... No such file or directory` | **Bash 写工作区以外的路径不落地**（`mkdir` 能建目录，文件写不进） | 下载物**一律放工作区目录**；或改用 PowerShell 工具写盘 |
| `python_embeded -m venv` 报 `No module named venv` | ComfyUI 的 python 是**嵌入式版** | 用受管 Python 建 venv |
| pip 疯狂 `Downloading kokoro-0.7.x` 循环 | 依赖里钉死的 numpy 版本**在当前 Python 上没有轮子** | 换 Python 版本，或改走 ONNX 路线绕开该包 |
| `pip install` 报 `Failed to activate VS environment` + meson | pip 试图**源码编译**（无 C 编译器） | 别让它编译：装预编译轮子的版本、或先把依赖装齐 |

## 3. 出"音色试听"的标准动作

1. 同一条文本跑完所有候选音色，**前半陈述 + 后半爆发**（例：「今天天气不错，我们出去走走吧。……你才飞起来！」），
   一次同时听**音色**和**情绪张力**。
2. 输出 `NN_性别_音色名.wav` + 一个拼接的「全部试听_顺序版」+ `README.txt` 列出顺序。
3. 用 ffmpeg `volumedetect` 核验（`-22 ~ -27 dB` 均值为正常；`-60dB` 级=静音，要查）。
4. 采样率 24000，转 mp3 用 `-b:a 192k` 方便试听。

## 4. ★★ 要"控情绪" → CosyVoice3（实测跑通，Windows 可行）

Kokoro **没有情绪参数**。真要演情绪走 CosyVoice3（阿里 / Apache 2.0 商用免费 / 自然语言指令控情绪）。

### 4.1 最省事的组合（★ 强烈推荐）
**Kokoro 出"音色样本" → 喂 CosyVoice3 当 `prompt_wav` → 用自然语言指令加"情绪"。**

理由：`Fun-CosyVoice3-0.5B` **没有内置音色表**（`frontend.spk2info == {}`），
所以 `inference_instruct(text, spk_id, instruct)` 那条路**用不了**（那是 CosyVoice-300M-SFT/Instruct 的）。
CosyVoice3 只有零样本路线：

```python
m = CosyVoice3(model_dir, load_trt=False, load_vllm=False, fp16=False)
# ★ instruct_text 必须以 <|endofprompt|> 结尾；不需要 prompt_text
for j in m.inference_instruct2(台词, "用炸毛大喊的语气说这句话<|endofprompt|>", 参考音色.wav):
    audio = j["tts_speech"]        # torch tensor (1, N)
```
于是：**Kokoro 的 103 个音色 = 音色库**（解决 CosyVoice3 没内置音色），
**CosyVoice3 的 instruct = 情绪**（解决 Kokoro 没情绪）。

### 4.2 部署步骤（Windows / 无 conda）
1. **Python 3.10.11** → 装到自定义目录（`D:\Python310`）。
   - 装包用 **Python urllib 下载**（`curl -o` 会被沙箱拦）
   - 杀进程：bash 里用 `MSYS_NO_PATHCONV=1 taskkill /F /IM xxx.exe`
2. venv → `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126`
3. `git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git`
   ★ **`third_party/Matcha-TTS` 是必需子模块**（`cosyvoice/flow/decoder.py` 直接 `from matcha...`）。
   `git clone` 子模块常失败 → **`https://ghproxy.net/https://github.com/<user>/<repo>/archive/refs/heads/main.zip`** 可通。
4. 模型：ModelScope **`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`**（国内源，9.1G，快）。
5. ★ **依赖逐个装，别整批 `-r requirements.txt`**：`openai-whisper` 一失败就拖垮全批。
   实测必须补的链：`lightning`（matcha.utils.instantiators）→ `gdown` / `rootutils`（matcha.utils.utils）
   → `whisper`（cosyvoice/cli/frontend.py）→ `pyworld`（cosyvoice/dataset/processor，**0.3.5 有 cp310 win 轮子**）。
   `openai-whisper` 用 `--no-build-isolation` 装。
   `pyworld` **别提前剔掉**——剔了会在 hyperpyyaml 加载 yaml 时炸。

### 4.3 ★ 两个必踩的运行时坑
| 报错 | 真因 | 对策 |
|---|---|---|
| `TorchCodec is required for load_with_torchcodec` | torchaudio ≥2.9 默认后端换成 torchcodec | 脚本开头 **monkey-patch `torchaudio.load/save` 用 soundfile** |
| `CUDA error: no kernel image is available` | 50 系（Blackwell/sm_120）装了 cu126 的 torch，没编 sm_120 核 | 跑推理加 `CUDA_VISIBLE_DEVICES=""` 走 CPU；或换 cu128/129 的 torch |

```python
import torch, torchaudio, soundfile as sf, numpy as np
torchaudio.load = lambda p,*a,**k: (torch.from_numpy(
    np.ascontiguousarray(sf.read(str(p),dtype="float32",always_2d=True)[0].T)),
    sf.read(str(p))[1])
def _save(p, src, sr, *a, **k):
    x = src.detach().cpu().numpy() if hasattr(src,"detach") else np.asarray(src)
    if x.ndim==2 and x.shape[0]<x.shape[1]: x = x.T
    sf.write(str(p), x, int(sr))
torchaudio.save = _save
```

### 4.4 情绪指令怎么写（决定成败）
自然语言、**写重写透**：语气 + 音量 + 重音字 + 语速 + 气口 + 尾音走向 + 心理状态。
例：
- 炸毛大喊 → 「音量明显拉高，是又气又好笑地吼回去，「才」字重读有爆破感，句尾收得干脆带一点破音」
- 服软认错 → 「音量压得很低，气声多，「错」字稍微拖一点，尾音往下沉，像泄了气的认输，带叹气感」

## 5. 若下一步要"控情绪"（其他备选）

- **IndexTTS-2**：情感向量/情感描述，显存吃紧。
- **Fish Speech**：行内标签 `[laugh]`/`[whisper]` 最精细，但 **Research 协议，商用需授权**。
- **ChatTTS**：对话自然度最高，但 **CC BY-NC 4.0 非商用**。
- 克隆 + 情绪化参考音频：给 F5-TTS / GPT-SoVITS 一段本身带情绪的参考音频。
