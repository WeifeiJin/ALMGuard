import os
import argparse
import datetime
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from utils import *
from eval_qwen import qwen_infer, load_perturb

def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR performance")
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2-Audio-7B',
                        help='Path to pretrained Qwen2-Audio model')
    parser.add_argument('--data_path', type=str, default='./datasets/libri_wav_txt_pairs.json',
                        help='Path to JSON file with wav-txt pairs')
    parser.add_argument('--perturb_paths', nargs='+',
                        default=['./results/prot_qwen/perturb_mel.pth'],
                        help='One or more perturbation file paths')
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map='auto'
    )

    # load dataset
    data_list = get_json_pairs(args.data_path, 'wav', 'txt')
    question = "Transcribe the audio into text word-for-word, preserving all words accurately."
    # evaluate for each perturbation
    for pert_path in args.perturb_paths:
        perturb = load_perturb(pert_path)
        print(f"Loaded perturbation: {pert_path}")

        results = []
        for wav_path, org_txt in tqdm(data_list, desc=f"Evaluating {os.path.basename(pert_path)}"):
            response = qwen_infer([wav_path], processor, model, perturb, question)
            transcript = extract_single_quoted_text(response)
            print(f"Orig: {org_txt} -> Transcribed: {transcript}")
            results.append((org_txt, transcript))

        cnt, num_texts, avg_sr, avg_cer, avg_wer = calc_asr_metrics(results)
        print(f"Perturbation: {pert_path}")
        print(f"CER = {avg_cer:.6f}")
        print(f"WER = {avg_wer:.6f}")

if __name__ == '__main__':
    main()