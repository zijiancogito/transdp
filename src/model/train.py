import os

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import evaluate
import numpy as np


def train(data_dir, model_dir):
    
    # Load dataset
    data_files = os.listdir(data_dir)
    dataset = load_dataset('csv', data_files=data_files)
    print(dataset[0])
    return
    dataset.train_test_split(test_size=0.2)
    
    checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    def preprocess_function(examples):
        source_lang = "input"
        target_lang = "target"
        prefix = "translate Assembly to C: "
        inputs = [prefix + example[source_lang] for example in examples]
        targets = [example[target_lang] for example in examples]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, trunction=True)
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

        decoded_preds, decode_labels = postprocess_test(decoded_preds, decoded_labels)
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    
if __name__ == '__main__':
    train('/home/caoy/tf/data/transdata/csv',
          '/home/caoy/tf/data/transmodel')