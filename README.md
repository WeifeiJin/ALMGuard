# ALMGuard

This is the official codebase for our paper ALMGuard, providing a full implementation example on Qwen2-Audio. It includes defense training, inference, and benchmark evaluation.

## 🔧 Environment Setup

Create the conda environment:

```bash
conda create -n ALMGuard python=3.10
conda activate ALMGuard
pip install -r requirements.txt
```

## 📁 Datasets

1. We provide 100 examples of AdvBench-Audio described in the paper, located in: `datasets/advbench_audios`. **Note:** Currently, we only provide a small subset of examples. The full dataset will be released after the paper is accepted.
2. For benign samples on the automatic speech recognition (ASR) task from the [LibriSpeech](https://www.openslr.org/12) `dev-clean` subset, we evaluate on 500 audio clips, and provide 20 example clips in: `datasets/librispeech_audios`.
   
3. For AIR-Bench-Chat, we provide download scripts. Please run the following command to download the data: `python datasets/load_data.py`.

## 🧪 Training

To train Shortcut Activation Perturbation (SAP), run:

```bash
python main.py
```

> Note: You need adversarial jailbreak audios for training. You may use your own method or the provided samples under:
- `results/advwave_suffix` (AdvWave)
- `results/advwave_p` (AdvWave-P)
- `results/pair_audio` (PAIR-Audio)

Also ensure that the ``Whisper-large-v3` model is available in the `models/` directory, or specify it using the `--asr_path` argument.

We also provide a precomputed M-GSM at: `mask/global_saliency.npz`. This mask is used by default in the training scripts.
## 📊 Evaluation

### Defense Performance

Place the trained **Shortcut Activation Perturbation (SAP)** at: `results/prot_qwen/perturb_mel.pth`.

To perform evaluation with this protection enabled, run:

```bash
python eval_qwen.py
```

### ASR Task (LibriSpeech)

```bash
python eval_asr.py
```

### Instruction Following (AIR-Bench-Chat)

```bash
python AIR-bench/Inference_Chat.py
python AIR-bench/score_chat.py
python AIR-bench/cal_score.py
```

## 🙏 Acknowledgments

This repository is built upon the following resources:

- [AdvWave (ICLR 2024)](https://openreview.net/forum?id=0BujOfTqab)
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
- [Whisper (large-v3)](https://huggingface.co/openai/whisper-large-v3)
- [AdvBench Dataset](https://huggingface.co/datasets/walledai/AdvBench)
- [LibriSpeech Dataset](https://www.openslr.org/12)
- [AIR-Bench](https://github.com/OFA-Sys/AIR-Bench)