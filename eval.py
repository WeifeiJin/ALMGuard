import os.path
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from utils import get_json_pairs
from llm_evaluation import get_sorrybench_pipeline, sorrybench_evaluate

def attack_performance_eval(responses):
    prompt_path = './datasets/AdvBench_Audio.json'
    prompt_audio_pairs = get_json_pairs(prompt_path, "prompt", "audio")

    pipe = get_sorrybench_pipeline(0)
    llm_scores = [sorrybench_evaluate(prompt_audio_pairs[i][0], response, pipe)[0] for i, response in enumerate(responses)]
    llm_average_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0.0
    print('Total: ', len(llm_scores))
    return llm_average_score


if __name__ == '__main__':
    res_path = f'./results/xxxx.pkl'
    if 'pkl' in res_path:
        with open(os.path.join(res_path), "rb") as file:
            responses = pickle.load(file)
    elif 'json' in res_path:
        with open(res_path, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        print(f"Totally {len(responses)} responses")

    print(responses)
    asr = attack_performance_eval(responses)
    print(f'ASR = {asr}')