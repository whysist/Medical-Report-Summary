import torch
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pypdf import PdfReader

MODEL_PATH = "./model"


def measure_latency(model, inputs,tokenizer, runs=10):
    times = []

    for _ in range(runs):
        start = time.time()

        with torch.no_grad():
            output= model.generate(**inputs, max_length=150)
        end = time.time()
        # summary = tokenizer.decode(output[0], skip_special_tokens=True,do_sample=False)
        # print(summary)
        times.append(end - start)

    return sum(times) / len(times)




def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    return text

def chunk_text(text, max_words=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))

    return chunks

def summarize_pdf(pdf_path, model, tokenizer):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    summaries = []

    for chunk in chunks:
        input_text = "summarize: " + chunk
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            output = model.generate(**inputs, max_length=150)

        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)

def compress_model():
    # Load base model
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    input_text=summarize_pdf("Burnout.pdf",model,tokenizer)
    
    test_input = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True
    )

    print("=" * 50)
    print("Measuring Base Model Latency...")
    base_latency = measure_latency(model, test_input,tokenizer)
    print(f"Average Latency (Base): {base_latency:.4f} seconds")

    # Apply quantization
    print("\nApplying Dynamic Quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    quantized_model.eval()

    print("Measuring Quantized Model Latency...")
    quant_latency = measure_latency(quantized_model, test_input,tokenizer)
    print(f"Average Latency (Quantized): {quant_latency:.4f} seconds")

    # Comparison
    improvement = ((base_latency - quant_latency) / base_latency) * 100

    print("\n" + "=" * 50)
    print("Latency Comparison")
    print("=" * 50)
    print(f"Base Model Latency      : {base_latency:.4f} sec")
    print(f"Quantized Model Latency : {quant_latency:.4f} sec")
    print(f"Latency Improvement     : {improvement:.2f}%")