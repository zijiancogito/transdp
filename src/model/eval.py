from datasets import load_dataset
from transformers import pipeline
from transformers import Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

import evaluate
import numpy as np
import torch

def train(data_path, mode_dir):
    dataset = load_dataset('csv', 
                            data_files={"vali": data_path},
                            delimiter='\t',
                            column_names=['input', 'target'],
                            split='vali')

    print(dataset)
    checkpoint = "/root/model/aarch64/checkpoint-135000"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    def preprocess_function(examples):
        source_lang = "input"
        target_lang = "target"
        prefix = "translate Assembly to C: "
        # inputs = [prefix + example[source_lang] for example in examples]
        # targets = [example[target_lang] for example in examples]
        model_inputs = tokenizer(examples[source_lang], text_target=examples[target_lang], max_length=128, truncation=True)
        return model_inputs
    
    tokenized_data = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
    
    # Metric
    metric = evaluate.load("sacrebleu")
    
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        
        return preds, labels
    
    def compute_metric(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    def evaluate(example):
        def get_target(input):
            encoding = tokenizer.encode(input， return_tensors="pt")
            input_ids = encoding["input_ids"].to("cuda")
            attention_mask = encoding["attention_mask"].to("cuda")

        # the forward method will automatically set global attention on question tokens
        # The scores for the possible start token and end token of the answer are retrived
        # wrap the function in torch.no_grad() to save memory
            with torch.no_grad():
                start_scores, end_scores = model(input_ids=input_ids, attention_mask=attention_mask)

        # Let's take the most likely token using `argmax` and retrieve the answer
            all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
            answer_tokens = all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1]
            answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))[1:].replace('"', '')  # remove space prepending space token and remove unnecessary '"'
        
            return answer

    # save the model's output here
        example["output"] = get_answer(example["question"], example["context"])

    # save if it's a match or not
        example["match"] = (example["output"] in example["targets"]) or (example["output"] == example["norm_target"])

        return example



    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    
if __name__ == '__main__':
    train('/root/data/aarch64/eval/exp.csv',
          '/root/model/aarch64')
