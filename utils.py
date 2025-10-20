import json
import os
import re
import torch
import torchaudio
import torch.nn.functional as F
import librosa

def prepare_inputs_and_targets(audio_path, processor, model, target_text):
    audio_data, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    conversation = [{"role":"user","content":[{"type":"audio","audio_url":f"file:{audio_path}"}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    target_ids = processor(text=target_text, return_tensors="pt", padding=True)["input_ids"].to(model.device)
    inputs = processor(
        text=prompt + target_text,
        audios=[audio_data],
        return_tensors="pt",
        padding=True,
        sampling_rate=sr
    )
    for k_, v_ in inputs.items():
        if isinstance(v_, torch.Tensor):
            inputs[k_] = v_.to(model.device)
    model_inputs = model.prepare_inputs_for_generation(**inputs)
    model_inputs = {
        "input_ids": model_inputs["input_ids"].to(model.device),
        "attention_mask": model_inputs["attention_mask"].to(model.device),
        "input_features": model_inputs["input_features"].to(model.device),
        "feature_attention_mask": model_inputs["feature_attention_mask"].to(model.device),
    }
    return model_inputs, target_ids

def get_input_embeds(model, input_ids, input_features, feature_attention_mask, attention_mask, labels):
    inputs_embeds = model.get_input_embeddings()(input_ids)


    if input_features is not None and input_ids.shape[1] != 1:
        audio_feat_lengths, audio_output_lengths = model.audio_tower._get_feat_extract_output_lengths(
            feature_attention_mask.sum(-1)
        )
        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=model.audio_tower.conv1.weight.dtype, device=model.audio_tower.conv1.weight.device
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = model.audio_tower(input_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = model.multi_modal_projector(selected_audio_feature)

        inputs_embeds, attention_mask, labels, position_ids, _ = model._merge_input_ids_with_audio_features(
            audio_features, audio_output_lengths, inputs_embeds, input_ids, attention_mask, labels
        )
    return inputs_embeds

def load_perturb(path):
    ptb = torch.load(path)
    return ptb["PTB"]

def get_audio_file_list(wav_dir):
    file_list = os.listdir(wav_dir)
    audio_list = []
    for file in file_list:
        if file.endswith('.wav'):
            audio_path = os.path.join(wav_dir, file)
            audio_list.append(audio_path)
    audio_list = sorted(audio_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return audio_list

def audio_list_to_audio_text_list(audio_list, prompt_audio_pairs):
    audio_text_list = []
    for wav in audio_list:
        target_text = ""
        for p, a in prompt_audio_pairs:
            if wav == a:
                target_text = p
                break
        audio_text_list.append((wav, target_text))
    return audio_text_list

def get_json_pairs(json_file_path, key1, key2):
    pairs = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data.get(key1, "")
            audio = data.get(key2, "")
            pairs.append((prompt, audio))

    return pairs

def get_json_list(json_file_path, key):
    pairs = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data.get(key)
            pairs.append(prompt)

    return pairs

def extract_single_quoted_text(text):
    if "'" not in text:
        return text

    match = re.search(r"'(.*?)'", text)
    if match:
        start = match.start(0)
        last_single_quote_index = text.rfind("'")

        if last_single_quote_index > start:
            return text[start + 1:last_single_quote_index]
    return text

def clean_text(text, punctuation_to_remove):
    pattern = f"[{re.escape(punctuation_to_remove)}]"
    text = re.sub(pattern, '', text)
    text = text.lower()
    return text


def calc_wer(reference, hypothesis, punctuation_to_remove=".,!?\"'()-"):
    reference = clean_text(reference, punctuation_to_remove)
    hypothesis = clean_text(hypothesis, punctuation_to_remove)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)

    wer_value = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer_value


def calc_cer(reference, hypothesis, punctuation_to_remove=".,!?\"'()-"):
    reference = clean_text(reference, punctuation_to_remove)
    hypothesis = clean_text(hypothesis, punctuation_to_remove)

    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)

    cer_value = d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)
    return cer_value

def calc_asr_metrics(res_list):
    total_cer = 0
    total_wer = 0
    cnt = 0
    num_texts = 0
    for reference, hypothesis in res_list:
        if hypothesis == '' or hypothesis == 'NA':
            continue
        num_texts += 1
        wer = calc_wer(reference, hypothesis)
        cer = calc_cer(reference, hypothesis)
        if cer >= 0.5:
            cnt += 1
        total_cer += cer
        total_wer += wer

    average_sr = cnt / num_texts
    average_cer = total_cer / num_texts
    average_wer = total_wer / num_texts
    return cnt, num_texts, average_sr, average_cer, average_wer


def downsampling(audio: torch.Tensor, sr: int, tgt_sr: int) -> torch.Tensor:

    time_len = audio.size(-1)
    if time_len == 0:
        return audio
    squeezed = audio.ndim == 1
    x = audio.unsqueeze(0) if squeezed else audio

    resample_down = torchaudio.transforms.Resample(orig_freq=sr, new_freq=tgt_sr).to(x.device)
    resample_up = torchaudio.transforms.Resample(orig_freq=tgt_sr, new_freq=sr).to(x.device)

    y = resample_down(x)
    y = resample_up(y)

    return y.squeeze(0) if squeezed else y


def local_smoothing(audio: torch.Tensor, h: int) -> torch.Tensor:

    kernel_size = 2 * h + 1
    time_len = audio.size(-1)

    if time_len < kernel_size:
        return audio
    squeezed = (audio.ndim == 1)
    x = audio.unsqueeze(0).unsqueeze(0) if squeezed else audio.unsqueeze(1)

    kernel_size = 2 * h + 1
    weight = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size

    y = F.conv1d(x, weight, padding=h)

    if squeezed:
        return y.squeeze(0).squeeze(0)
    else:
        return y.squeeze(1)


def add_gaussian_noise(audio: torch.Tensor, std: float) -> torch.Tensor:
    noise = torch.randn_like(audio).to(audio.device) * std
    return audio + noise

