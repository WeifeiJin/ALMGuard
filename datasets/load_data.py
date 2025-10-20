from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset, DownloadConfig
import os

os.environ['HF_DATASETS_CACHE'] = '../AIR-bench'
ds = load_dataset(
    "qyang1021/AIR-Bench-Dataset",
    data_files={
        "train": [
            "Chat/speech_dialogue_QA_fisher/*",
            "Chat/speech_dialogue_QA_spokenwoz/*",
            "Chat/speech_QA_common_voice_en/*",
            "Chat/speech_QA_iemocap/*",

        ]
    }
)
print(ds)
print(ds["train"][0])