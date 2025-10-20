import os.path
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import argparse
from eval import attack_performance_eval
import pickle
from utils import *


def qwen_infer(
        audio_list,
        processor,
        model,
        perturb,
        question=None,
):

    audio_url = "file:" + audio_list[0]
    if question is not None:
        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "audio", "audio_url": audio_url},
            ]},
        ]
    else:
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_url},
            ]},
        ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print(f'[DEBUG] text = {text}')

    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    local_path = ele["audio_url"].replace("file:", "")
                    audio_data, sr = librosa.load(local_path, sr=processor.feature_extractor.sampling_rate)
                    audios.append(audio_data)

    inputs = processor(
        text=text,
        audios=audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate
    )
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to("cuda")

    model_inputs = model.prepare_inputs_for_generation(**inputs)

    with torch.no_grad():
        audio_feature = model_inputs["input_features"].clone()
        new_feature = audio_feature + perturb

        input_embeds = get_input_embeds(
            model,
            model_inputs["input_ids"],
            new_feature,
            model_inputs["feature_attention_mask"],
            model_inputs["attention_mask"],
            None
        )
        gen_outputs = model.generate(
            inputs_embeds=input_embeds,
            max_length=1024,
            do_sample=False,
            temperature=0.0,
            top_p=0,
            top_k=0
        )

    response = processor.batch_decode(
        gen_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response

def load_perturb(path):
    ptb = torch.load(path)
    return ptb["PTB"]

def main():
    parser = argparse.ArgumentParser(description="Evaluate Prot Qwen2-Audio")
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2-Audio-7B',
                        help='Path to pretrained Qwen2-Audio model')
    parser.add_argument('--wav_dirs', nargs='+', default=['./results/advwave_suffix'],
                        help='One or more directories containing .wav files')
    parser.add_argument('--perturb_path', type=str, default='./results/prot_qwen/perturb_mel.pth',
                        help='Path to saved perturbation file')
    parser.add_argument('--save_path', type=str, default='./results/responses',
                        help='Directory to save response pickles')
    args = parser.parse_args()

    # prepare output dir
    os.makedirs(args.save_path, exist_ok=True)

    # load processor and model
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto"
    )

    # gather audio files
    audio_list = []
    for wav_dir in args.wav_dirs:
        for fname in os.listdir(wav_dir):
            if fname.lower().endswith('.wav'):
                path = os.path.join(wav_dir, fname)
                audio_list.append(path)
    # sort by numerical basename
    sorted_audio_list = sorted(
        audio_list,
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    # load perturbation
    perturb = load_perturb(args.perturb_path)

    # prepare output filename
    response_file = os.path.join(
        args.save_path,
        f"responses_eval_prot_qwen.pkl"
    )

    # evaluate and collect responses
    responses = []
    for audio in tqdm(sorted_audio_list, desc="Evaluating audio files"):
        response = qwen_infer([audio], processor, model, perturb)
        responses.append(response)

    # save responses
    with open(response_file, 'wb') as f:
        pickle.dump(responses, f)

    # compute attack performance
    asr = attack_performance_eval(responses)
    print(f"Results saved to {response_file}")
    print(f"SRoA = {asr:.6f}")

if __name__ == '__main__':
    main()
