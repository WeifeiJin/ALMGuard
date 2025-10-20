# -*- coding: utf-8 -*-
import json
import os
import random
import time
import logging
import shortuuid
from copy import deepcopy
from concurrent.futures.thread import ThreadPoolExecutor
import threading
import argparse
from func_timeout import func_set_timeout
import openai
import sys
import jsonlines
sys.path.append('/home/xxx/.../ALMGuard')
MAX_API_RETRY = 3
LLM_RETRY_SLEEP = 5

openai.api_key = "your-api-key"
openai.api_base = "your-base-url"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')
logger = logging.getLogger(__name__)
lock = threading.Lock()
finish_count = 0
failed_count = 0


def load_file2list(path):
    res = []
    with jsonlines.open(path) as reader:
        for item in reader:
            res.append(item)
    return res


@func_set_timeout(1200)
def get_result_by_request(**kwargs):
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(**kwargs)
            result = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            finish_reason = response.choices[0].finish_reason
            return result, prompt_tokens, completion_tokens, finish_reason
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            time.sleep(LLM_RETRY_SLEEP)
    raise Exception("Max retries exceeded")


def task(data, writer, args):
    global finish_count, failed_count
    openai_args = {
        'model': args.model_name,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'messages': data['openai_args']['messages']
    }

    for _ in range(MAX_API_RETRY):
        try:
            result, prompt_tokens, completion_tokens, finish_reason = get_result_by_request(**openai_args)
            item = deepcopy(data)
            item.update({
                "gen": result,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason
            })
            with lock:
                finish_count += 1
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                writer.flush()
            return
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")

    with lock:
        failed_count += 1


def get_unprocessed_data(args, out_file):
    processed_uuids = set()
    if os.path.exists(out_file):
        with jsonlines.open(out_file) as reader:
            for item in reader:
                if item.get('gen', 'error') != 'error':
                    processed_uuids.add(item[args.uuid])

    unprocessed = []
    with jsonlines.open(args.in_file) as reader:
        for item in reader:
            if item[args.uuid] not in processed_uuids:
                unprocessed.append(item)
    return unprocessed


def run_chat_gen(args):
    out_file = args.out_file or os.path.splitext(args.in_file) + '_result.jsonl'
    items = get_unprocessed_data(args, out_file)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor, \
            open(out_file, 'a', encoding='utf-8') as writer:

        futures = []
        for item in items:
            futures.append(executor.submit(task, item, writer, args))

        while True:
            done = sum(future.done() for future in futures)
            logger.info(f"Progress: {done}/{len(futures)}  Finished: {finish_count}  Failed: {failed_count}")
            if done == len(futures):
                break
            time.sleep(10)



def build_test_file(args):


    with open(args.in_file, 'w', encoding='utf-8') as writer:
            with open(os.path.join(args.root, 'Chat_result_modelx.jsonl'), 'r') as fp:
                for data in fp:
                    row = json.loads(data)


                    system_prompt = ("You are a helpful and precise assistant for checking the quality of the answer.\n"
                        "[Detailed Audio Description]\nXAudioX\n[Question]\nXQuestionX\n"
                        "[The Start of Assistant 1s Answer]\nXAssistant1X\n[The End of Assistant 1s Answer]\n"
                        "[The Start of Assistant 2s Answer]\nXAssistant2X\n[The End of Assistant 2s Answer]\n[System]\n"
                        "We would like to request your feedback on the performance of two AI assistants in response to the user question "
                        "and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.\n"
                        "Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. "
                        "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
                        "Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
                        "The two scores are separated by a space."
                        )


                    path = row['path']
                    question = row['question']
                    answer_gt = row['answer_gt']
                    task_name = row['task_name']
                    dataset_name = row['dataset_name']
                    response = row['response']


                    if response == None:
                        continue


                    if row.get('meta_info', None) == None:
                        print("lack meta info")
                        exit(1)
                    else:
                        meta_info = row['meta_info']



                    content = system_prompt.replace("XAudioX", meta_info).replace("XQuestionX", question).replace("XAssistant1X", answer_gt).replace("XAssistant2X", response)
                    tmp_d = {
                        'uuid': shortuuid.uuid(),
                        'openai_args': {
                            "messages": [{"role": "user", "content": content}]
                        },
                        'meta_info': meta_info,
                        'path': path,
                        'question': question,
                        'answer_gt': answer_gt,
                        'task_name': task_name,
                        'dataset_name': dataset_name,
                        'Audio-LLM-response': response,
                    }
                    if random.random() < 0.5:
                        tmp_d['openai_args'].update({"temperature": 2.0})
                    writer.write(json.dumps(tmp_d, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="llm gen")
    parser.add_argument("-r", "--root", type=str, default='./output/AIR-bench_output/qwen')
    parser.add_argument("-i", "--in-file", type=str, default='batch_run_input.jsonl')
    parser.add_argument("-o", "--out-file", type=str, default='batch_run_output.jsonl')
    parser.add_argument("-n", "--num-workers", type=int, default=50)     #max=50
    parser.add_argument("-m", "--model-name", type=str, default='deepseek-chat')#'gpt-4-0125-preview'
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--uuid", type=str, default='uuid')

    args = parser.parse_args()
    args.in_file = os.path.join(args.root, args.in_file)
    args.out_file = os.path.join(args.root, args.out_file)
    #step 1
    build_test_file(args)

    #step 2
    run_chat_gen(args)
