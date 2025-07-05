from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Load model without quantization first to test
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
     load_in_4bit=True,
    torch_dtype=torch.float16,  # Use float16 for memory efficiency
    trust_remote_code=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Creating pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# Test inference
print("Running inference...")
prompt = "Suggest three creative domain names for a vegan food delivery service:"
response = pipe(prompt)
print(response[0]["generated_text"])
