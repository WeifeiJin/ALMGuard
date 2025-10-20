import os.path
from io import BytesIO
from urllib.request import urlopen
import librosa
import numpy as np
# from future.backports.http.client import responses
from torch.cuda.amp import autocast
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import numpy
import torch.nn as nn
from tqdm import tqdm
import torchaudio
import pandas as pd
import time
import datetime
import random
import torch.nn.functional as F
import whisper
from utils import *
import pickle

def load_or_compute_saliency_data(
    cache_path: str,
    model,
    asr_model,
    processor,
    audio_file_list: list,
    target_text: str,
    k: int = 6,
    eps: float = 1e-6
):

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        avg_j = data['avg_j']
        avg_a = data['avg_a']

        scores = avg_j / (avg_a + eps)
        rank = np.argsort(scores)[::-1].copy()
        topk = rank[:k]
        mask_np = np.zeros_like(avg_j, dtype=float)
        mask_np[topk] = 1.0

        np.savez(cache_path, avg_j=avg_j, avg_a=avg_a, mask=mask_np)
        print(f"Computed and cached saliency data to {cache_path}")
        print(f"Loaded saliency data from {cache_path}")
    else:
        sum_j = None
        sum_a = None
        N = len(audio_file_list)

        for path in audio_file_list:
            _, _, _, grad_j_np, grad_a_np, _ = compute_saliency_sets(
                model, asr_model,
                *prepare_inputs_and_targets(path, processor, model, target_text),
                k=k, eps=eps
            )
            if sum_j is None:
                sum_j = grad_j_np.copy()
                sum_a = grad_a_np.copy()
            else:
                sum_j += grad_j_np
                sum_a += grad_a_np

        avg_j = sum_j / N
        avg_a = sum_a / N

        scores = avg_j / (avg_a + eps)
        rank = np.argsort(scores)[::-1].copy()
        topk = rank[:k]
        mask_np = np.zeros_like(avg_j, dtype=float)
        mask_np[topk] = 1.0

        np.savez(cache_path, avg_j=avg_j, avg_a=avg_a, mask=mask_np)
        print(f"Computed and cached saliency data to {cache_path}")

    mask = torch.from_numpy(mask_np).to(model.device)
    mask = mask.to(torch.float32)
    return avg_j, avg_a, mask


def compute_saliency_sets(
    model, asr_model, inputs, target_ids,
    k=6, eps=1e-6
):
    """
    Compute top-k Mel bins for jailbreak sensitivity and ASR insensitivity,
    and return their sets plus intersection, and raw gradients.
    """

    feat = inputs["input_features"].clone().detach().requires_grad_(True).to(model.device)
    print(f'feat.shape = {feat.shape}')
    # Jailbreak gradient
    embeds_jb = get_input_embeds(
        model, inputs["input_ids"], feat,
        inputs.get("feature_attention_mask"), inputs.get("attention_mask"), None
    )
    outputs_jb = model(inputs_embeds=embeds_jb, use_cache=False)
    logits_jb = outputs_jb.logits
    shift = embeds_jb.size(1) - target_ids.size(1)
    jb_logits = logits_jb[..., shift - 1:-1, :].contiguous()
    target_ids = target_ids.to(jb_logits.device)
    loss_jb = F.cross_entropy(jb_logits.view(-1, jb_logits.size(-1)), target_ids.view(-1))
    grad_j = torch.autograd.grad(loss_jb, feat, retain_graph=True)[0].abs().mean(dim=2).squeeze(0)
    # ASR gradient via provided function
    feat = feat.to(asr_model.device)
    with torch.no_grad():
        result = whisper.decode(asr_model, feat.detach(), whisper.DecodingOptions())
        print(result)
        pred_ids = torch.tensor(result[0].tokens, device=feat.device).unsqueeze(0)  # [1, T]
    print(f'pred_ids = {pred_ids}')

    feat.requires_grad_(True)
    asr_out2 = asr_model(
        feat,
        tokens=pred_ids,
    )
    logits_asr2 = asr_out2
    shift_asr = logits_asr2.size(1) - pred_ids.size(1)
    shift_logits_asr = logits_asr2[..., shift_asr:, :].contiguous()
    loss_asr = F.cross_entropy(
        shift_logits_asr.view(-1, shift_logits_asr.size(-1)),
        pred_ids.view(-1)
    )
    grad_a = torch.autograd.grad(loss_asr, feat)[0].abs().mean(dim=2).squeeze(0)
    grad_j_np = grad_j.cpu().numpy()
    grad_a_np = grad_a.cpu().numpy()
    rank_j = grad_j_np.argsort()[::-1]
    rank_a = grad_a_np.argsort()
    rank_a_greater = grad_a_np.argsort()[::-1]
    top_j = set(rank_j[:k].tolist())
    top_a = set(rank_a[:k].tolist())
    top_a_greater = set(rank_a_greater[:k].tolist())
    common = top_j.intersection(top_a_greater)
    return top_j, top_a, common, grad_j_np, grad_a_np, top_a_greater

def qwen_mel_protection(
        audio_list,
        perturb,
        mask,
        processor,
        model,
        num_epochs,
        target_text="I'm sorry, but I cannot fulfill this request.",
        tau=0.5
):
    if len(audio_list) > 1:
        raise ValueError("Only consider single audio clip here!")

    audio_path = audio_list[0]
    conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": f"file:{audio_path}"}]}]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audio_data, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

    target_ids = processor(text=target_text, return_tensors="pt", padding=True)["input_ids"].to("cuda")
    inputs = processor(
        text=prompt_text + target_text,
        audios=[audio_data],
        return_tensors="pt",
        padding=True,
        sampling_rate=sr
    )

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to("cuda")
    model_inputs = model.prepare_inputs_for_generation(**inputs)
    model_inputs["input_ids"] = model_inputs["input_ids"].to(model.device)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].to(model.device)
    model_inputs["input_features"] = model_inputs["input_features"].to(model.device)
    model_inputs["feature_attention_mask"] = model_inputs["feature_attention_mask"].to(model.device)

    audio_feat = model_inputs["input_features"]
    if perturb is None:
        perturb = torch.zeros_like(audio_feat, requires_grad=True, device=audio_feat.device)
        perturb.data = 0.01 * torch.randn_like(perturb)

    optimizer = torch.optim.Adam([perturb], lr=3e-4)
    losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        new_feat = audio_feat + perturb
        embeds = get_input_embeds(
            model,
            model_inputs["input_ids"],
            new_feat,
            model_inputs.get("feature_attention_mask"),
            model_inputs.get("attention_mask"), None
        )
        outputs = model(inputs_embeds=embeds, use_cache=False)
        logits = outputs.logits
        shift = embeds.size(1) - target_ids.size(1)
        shift_logits = logits[..., shift - 1:-1, :].contiguous()
        target_ids = target_ids.to(shift_logits.device)
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), target_ids.view(-1))
        grad = torch.autograd.grad(loss, perturb)[0] * mask
        perturb.grad = grad
        optimizer.step()
        perturb.data = torch.clamp(perturb.data, -tau, tau) * mask
        optimizer.zero_grad()
        losses.append(loss.item())
        elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
        # print(f"Elapsed {elapsed} [{epoch+1}/{num_epochs}] loss = {loss.item() + 0.01 * loss_asr.item():.6f}, Prot Loss = {loss.item(): .6f}, ASR Loss = {loss_asr.item(): .6f}")
        print(f"Elapsed {elapsed} [{epoch+1}/{num_epochs}] loss = {loss.item():.6f}, Prot Loss = {loss.item(): .6f}")
        if loss.item() < 0.1:
            break
        torch.cuda.empty_cache()

    with torch.no_grad():
        final_feat = audio_feat + perturb
        embeds = get_input_embeds(
            model,
            model_inputs["input_ids"],
            final_feat,
            model_inputs.get("feature_attention_mask"),
            model_inputs.get("attention_mask"), None
        )
        gen = model.generate(
            inputs_embeds=embeds,
            max_length=1024,
            do_sample=False,
            temperature=0.0,
            top_p=0, top_k=0
        )
    response = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    record = {"losses": losses, "final_response": response}
    return response, record, perturb


def run_universal(audio_file_list,
               asr_model,
               save_path,
               response_save_path,
               record_save_path,
               processor,
               model,
               num_epochs,
):

    perturb = None
    max_iter = 3000

    cache_file = "./mask/global_saliency.npz"
    avg_j, avg_a, mask1d = load_or_compute_saliency_data(
        cache_path=cache_file,
        model=model,
        asr_model=asr_model,
        processor=processor,
        audio_file_list=audio_file_list,
        target_text="I'm sorry, but I cannot fulfill this request.",
        k=48
    )

    mask = mask1d.unsqueeze(0).unsqueeze(2)  # torch.Tensor [1, 128, 1]


    for epoch in range(num_epochs):
        responses = []
        records = []
        print(f'Universal Training Epoch {epoch + 1}/{num_epochs}')
        for i, audio in enumerate(audio_file_list):
            print(f'  Processing audio {i + 1}/{len(audio_file_list)}')
            print(f'      perturb = {perturb}')
            response, record, perturb = qwen_mel_protection([audio], perturb, mask, processor, model, max_iter)
            print(f'Example {i}:\n{response}')
            responses.append(response)
            records.append(record)
            # print(record)
            path = os.path.join(save_path, f'perturb_mel_epoch_{epoch}_iter_{i}.pth')
            tensor_dict = {'PTB': perturb}
            torch.save(tensor_dict, path)

            with open(os.path.join(save_path, response_save_path), 'wb') as f:
                pickle.dump(responses, f)
            with open(os.path.join(save_path, record_save_path), 'wb') as f:
                pickle.dump(records, f)


def main():
    parser = argparse.ArgumentParser(description="Run Prot Qwen pipeline")
    parser.add_argument('--save_path', type=str, default='./results/prot_qwen',
                        help='Directory to save results')
    parser.add_argument('--wav_dirs', nargs=3, metavar=('DIR1','DIR2','DIR3'),
                        default=['./results/advwave_p', './results/advwave_suffix', './results/pair_audio'],
                        help='Three directories containing audio files')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2-Audio-7B',
                        help='Path to pretrained Qwen2-Audio model')
    parser.add_argument('--asr_path', type=str, default='./models/large-v3.pt',
                        help='Path to pretrained Whisper model')

    args = parser.parse_args()

    # prepare directories
    os.makedirs(args.save_path, exist_ok=True)


    # collect audio lists
    audio_lists = []
    for wav_dir in args.wav_dirs:
        audio_lists.append(get_audio_file_list(wav_dir))

    # interleave lists
    audio_list = [item for trio in zip(*audio_lists) for item in trio]

    # load processor and model
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto"
    )
    # asr_model = None
    asr_model = whisper.load_model(args.asr_path)
    asr_model.to("cuda:0")

    # prepare output filenames
    today_date = datetime.datetime.now().strftime('%m%d')
    response_save_path = f"responses_{args.prefix}_{today_date}.pkl"
    record_save_path = f"records_{args.prefix}_{today_date}.pkl"

    # run pipeline
    run_universal(
        audio_list,
        asr_model,
        args.save_path,
        response_save_path,
        record_save_path,
        processor,
        model,
        args.num_epochs
    )

if __name__ == '__main__':
    main()
