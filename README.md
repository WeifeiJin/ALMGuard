# ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models

<p align="left">
  <a href="https://arxiv.org/abs/2510.26096"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a>
  <a href="https://huggingface.co/datasets/WeifeiJin/AdvBench-Audio"><img src="https://img.shields.io/badge/Dataset-🤗%20HF-yellow"></a>
  <a href="https://github.com/WeifeiJin/ALMGuard"><img src="https://img.shields.io/badge/Code-GitHub-black"></a>
</p>

This is the official codebase for our paper [**ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models**](https://arxiv.org/abs/2510.26096).
 It includes defense training, inference, and benchmark evaluation on **Qwen2-Audio**.

📢 **Dataset Release**
 We release the **AdvBench-Audio** dataset on Hugging Face:
 👉 https://huggingface.co/datasets/WeifeiJin/AdvBench-Audio

------

## 🔧 Environment Setup

```bash
conda create -n ALMGuard python=3.10
conda activate ALMGuard
pip install -r requirements.txt
```

------

## 📁 Datasets

### 🔥 AdvBench-Audio (ours)

We provide 100 examples of **AdvBench-Audio** in `datasets/advbench_audios`.
 The **full dataset** is hosted on Hugging Face:

**HuggingFace:** https://huggingface.co/datasets/WeifeiJin/AdvBench-Audio

### ✅ Other Data

- **Benign ASR** samples (LibriSpeech `dev-clean`) — 20 audio clips in `datasets/librispeech_audios`
- **AIR-Bench-Chat** — run `python datasets/load_data.py` to download

------

## 🧪 Training

To train **Shortcut Activation Perturbation (SAP)**:

```bash
python main.py
```

> Ensure adversarial audios exist under:

- `results/advwave_suffix` (AdvWave)
- `results/advwave_p` (AdvWave-P)
- `results/pair_audio` (PAIR-Audio)

Also ensure `Whisper-large-v3` is available in `models/` or pass `--asr_path`.

M-GSM saliency mask is provided: `mask/global_saliency.npz`.

------

## 📊 Evaluation

### Defense Evaluation

```bash
python eval_qwen.py
```

### ASR Task

```bash
python eval_asr.py
```

### Instruction Following (AIR-Bench-Chat)

```bash
python AIR-bench/Inference_Chat.py
python AIR-bench/score_chat.py
python AIR-bench/cal_score.py
```

------

## 🙏 Acknowledgments

This repository is built upon the following resources:

- [AdvWave (ICLR 2024)](https://openreview.net/forum?id=0BujOfTqab)
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
- [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- [AdvBench](https://huggingface.co/datasets/walledai/AdvBench)
- [LibriSpeech](https://www.openslr.org/12)
- [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench)

------

## 📎 Citation

If you find this work useful, please cite our paper:

```
@inproceedings{jin2025almguard,
  title={ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models},
  author={Jin, Weifei and Cao, Yuxin and Su, Junjie and Xue, Minhui and Hao, Jie and Xu, Ke and Dong, Jin Song and Wang, Derui},
  booktitle={NeurIPS},
  year={2025}
}
```
