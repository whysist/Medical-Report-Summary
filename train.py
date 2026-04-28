from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

MODEL_NAME = "t5-small"

def train_model():
    dataset = load_dataset("ccdv/pubmed-summarization")
    train_data = dataset["train"].select(range(100))  # small subset for CPU

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    def preprocess(example):
        inputs = ["summarize: " + text for text in example["article"]]

        model_inputs = tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        labels = tokenizer(
            example["abstract"],
            truncation=True,
            padding="max_length",
            max_length=150
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized = train_data.map(preprocess, batched=True)

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=50
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized
    )

    trainer.train()
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")

