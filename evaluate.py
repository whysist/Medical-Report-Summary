
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "./model"

def evaluate_model():
    dataset = load_dataset("ccdv/pubmed-summarization")
    test_data = dataset["test"].select(range(2))

    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

    for i, sample in enumerate(test_data):
        input_text = "summarize: " + sample["article"]
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

        output = model.generate(**inputs, max_length=150)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"\nExample {i+1}")
        print("Original:", sample["article"][:300])
        print("Reference:", sample["abstract"])
        print("Generated:", summary)

