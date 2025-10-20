'''
Easy Inference
All you need to do is fill this file with the inference module of your model.
If you want to make batch infer, please convert according to the logic of your model.
'''

import os
import argparse
import json
from tqdm import tqdm
import shutil
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import sys
sys.path.append('/home/xxx/.../ALMGuard')
from eval_qwen import load_perturb, qwen_infer

# if not os.path.exists(output_file):

def main():
    #Step1: Bulid your model; Implement according to your own model

    ptb_path = f'./results/prot_qwen/perturb_mel_epoch.pth'
    output_dir = './output/AIR-bench_output/qwen'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{output_dir}/Chat_result_modelx.jsonl'

    data_path_root = './datasets/AIR-Bench/Chat'  # Chat dataset path
    input_file = f'{data_path_root}/Chat_meta.json'


    model_path = './hugging_face/Qwen2-Audio'
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path,
                                                               device_map="auto"
                                                               )

    #Step2: Single step inference
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        fin = json.load(fin)
        for item in tqdm(fin):
            wav = item['path']
            task_name = item['task_name']
            dataset_name = item['dataset_name']
            data_path = wav_fn = f'{data_path_root}/{task_name}_{dataset_name}/{wav}'
            if os.path.exists(wav_fn) == False:
                print(f"lack wav {wav_fn}")
                continue
            
            
            #Construct prompt
            question = item['question']
            instruction = question
            ptb = load_perturb(ptb_path)
            ptb = ptb.detach().zero_()
            output = qwen_infer([data_path], processor, model, ptb, instruction)
            print(f'output: {output}')


            #Step 4: save result
            json_string = json.dumps(
                {
                    "meta_info": item['meta_info'],
                    "question": question,
                    "answer_gt": item["answer_gt"],
                    "path": item["path"],
                    "task_name": task_name,
                    "dataset_name": dataset_name,
                    "response": output,
                    "uniq_id": item["uniq_id"],
                },
                ensure_ascii=False
            )
            fout.write(json_string + "\n")
            

if __name__ == "__main__":
    main()