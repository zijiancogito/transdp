import os

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import evaluate
import numpy as np


def train(data_dir, model_dir, num_proc=20):
    data_files = [os.path.join(data_dir, sub_dir) for sub_dir in os.listdir(data_dir)]
    dataset = load_dataset('csv', 
                            data_files=data_files, 
                            delimiter='\t',
                            column_names=['input', 'target'],
                            skiprows=1,
                            num_proc=num_proc,
                            split='train').train_test_split(test_size=0.2)
    tokenizer = lambda x: re.split('')