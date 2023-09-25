import os

from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import re
from transformers import BertConfig, DebertaConfig
from transformers import EncoderDecoderConfig, EncoderDecoderModel

def train(data_dir, model_dir, num_proc=20):
    
    # Load dataset
    data_files = [os.path.join(data_dir, sub_dir) for sub_dir in os.listdir(data_dir)]
    data_files = data_files[0:200000]
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

    # src_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer = lambda x, y: (x.split(' ; '), y.split(' '))
    import pdb
    def preprocess_function(examples):
        source_lang = "input"
        target_lang = "target"
        prefix = "translate Assembly to C: "
        # inputs = [prefix + example[source_lang] for example in examples]
        # targets = [example[target_lang] for example in examples]
        # inputs = [asm_tokenizer(i) for i in examples[source_lang]]
        # targets = [src_tokenizer(i) for i in examples[target_lang]]
        model_inputs = [tokenizer(x, y) for x, y in zip(examples[source_lang], examples[target_lang])]

        # model_inputs = tokenizer(examples[source_lang], text_target=examples[target_lang], max_length=128, truncation=True)
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

    enc_config = DebertaConfig(
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        posision_embedding_type='relative_key_query',
        relative_attention=True,
        max_relative_positions=5
    )
    dec_config = DebertaConfig(
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        relative_attention=True,
        max_relative_positions=5
    )
    config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    
    model = EncoderDecoderModel(config=config, )
    
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
    train('/root/data/aarch64/csv',
          '/root/model/aarch64-norm-all-emb')

