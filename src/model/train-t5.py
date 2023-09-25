import os

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import evaluate
import numpy as np


def train(data_dir, model_dir, num_proc=20):
    
    # Load dataset
    data_files = [os.path.join(data_dir, sub_dir) for sub_dir in os.listdir(data_dir)]
    data_files = data_files[0:100000]
    dataset = load_dataset('csv', 
                            data_files=data_files, 
                            delimiter='\t',
                            column_names=['input', 'target'],
                            skiprows=1,
                            num_proc=num_proc,
                            split='train').train_test_split(test_size=0.2)

    # print(dataset['train'][0])
    # return
    # data = dataset.train_test_split(test_size=0.2)
    print(dataset['train'][0])
    print(dataset['test'][0])
    # return
    
    checkpoint = "t5-small"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    import pdb
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


    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=512,
        per_device_eval_batch_size=512,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metric,
    )
    
    trainer.train()

    
if __name__ == '__main__':
    train('/root/data/mips/csv-src',
          '/root/model/mips-norm-src')

# {'eval_loss': 0.19186386466026306, 'eval_bleu': 58.7168, 'eval_gen_len': 13.5775, 'eval_runtime': 9640.1935, 'eval_samples_per_second': 906.116, 'eval_steps_per_second': 1.77, 'epoch': 2.0} aarch64

# {'eval_loss': 0.2583463191986084, 'eval_bleu': 59.9352, 'eval_gen_len': 12.1518, 'eval_runtime': 8876.0422, 'eval_samples_per_second': 984.125, 'eval_steps_per_second': 1.922, 'epoch': 2.0} 

# aarch64 norm all with out inst emb
#{'eval_loss': 0.056397534906864166, 'eval_bleu': 86.8677, 'eval_gen_len': 11.0137, 'eval_runtime': 8736.5132, 'eval_samples_per_second': 999.842, 'eval_steps_per_second': 1.953, 'epoch': 2.0}                                                                                                            
# {'train_runtime': 69938.7887, 'train_samples_per_second': 999.174, 'train_steps_per_second': 1.952, 'train_loss': 0.06497675813129698, 'epoch': 2.0} 

# x64 norm all with out inst emb 
# {'eval_loss': 0.05804423615336418, 'eval_bleu': 90.1805, 'eval_gen_len': 11.6534, 'eval_runtime': 8734.6852, 'eval_samples_per_second': 1000.147, 'eval_steps_per_second': 1.953, 'epoch': 1.0}     

# x64 norm asm 
# {'eval_loss': 0.3170093894004822, 'eval_bleu': 56.0844, 'eval_gen_len': 11.9852, 'eval_runtime': 3534.8417, 'eval_samples_per_second': 988.461, 'eval_steps_per_second': 1.931, 'epoch': 2.0}  

# aarch64 norm asm
# {'eval_loss': 0.3480462431907654, 'eval_bleu': 49.494, 'eval_gen_len': 11.9248, 'eval_runtime': 3518.0866, 'eval_samples_per_second': 993.073, 'eval_steps_per_second': 1.94, 'epoch': 1.0}


# x64 norm src
# {'eval_loss': 0.05873624607920647, 'eval_bleu': 91.6471, 'eval_gen_len': 11.3392, 'eval_runtime': 3614.4158, 'eval_samples_per_second': 966.699, 'eval_steps_per_second': 1.888, 'epoch': 1.0} 

# aarch64 norm src
# 'eval_loss': 0.05211639031767845, 'eval_bleu': 89.3345, 'eval_gen_len': 11.2891, 'eval_runtime': 3479.4367, 'eval_samples_per_second': 1004.104, 'eval_steps_per_second': 1.961, 'epoch': 2.0}                                                       
# {'train_runtime': 28074.4181, 'train_samples_per_second': 995.559, 'train_steps_per_second': 1.944, 'train_loss': 0.07406730816944722, 'epoch': 2.0} 

# mips norm src
