# Run: accelerate launch --num_processes 4 --gpu_ids '0,1,2,3' --main_process_port 29600 exercise.py \

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import evaluate
import transformers
transformers.set_seed(42)
from transformers import (AutoFeatureExtractor, 
                          AutoModelForAudioClassification, 
                          TrainingArguments, 
                          Trainer)
from datasets import load_dataset, Audio

def main():

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
            # return_attention_mask=True,
        )
        return inputs

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    gtzan = load_dataset("marsyas/gtzan", "all")
    gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
    id2label_fn = gtzan["train"].features["genre"].int2str

    model_id = "openai/whisper-large-v3"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, do_normalize=True, # return_attention_mask=True
    )
    # Resampling
    sampling_rate = feature_extractor.sampling_rate
    gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))

    max_duration = 30.0

    gtzan_encoded = gtzan.map(
        preprocess_function,
        remove_columns=["audio", "file"],
        batched=True,
        batch_size=100,
        num_proc=1,
    )
    gtzan_encoded = gtzan_encoded.rename_column("genre", "label")

    id2label = {
        str(i): id2label_fn(i)
        for i in range(len(gtzan_encoded["train"].features["label"].names))
    }
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)

    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    model_name = model_id.split("/")[-1]
    batch_size = 1
    gradient_accumulation_steps = 4
    num_train_epochs = 10
    eval_steps = (1/2) * (1/num_train_epochs)

    training_args = TrainingArguments(
        f"{model_name}-finetuned-gtzan",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        learning_rate=4e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        fp16=True,
        # optim="adafactor",
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    metric = evaluate.load("accuracy")

    trainer = Trainer(
        model,
        training_args,
        train_dataset=gtzan_encoded["train"],
        eval_dataset=gtzan_encoded["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    kwargs = {
        "dataset_tags": "marsyas/gtzan",
        "dataset": "GTZAN",
        "model_name": f"{model_name}-finetuned-gtzan",
        "finetuned_from": model_id,
        "tasks": "audio-classification",
    }

    trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    main()