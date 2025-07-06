from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Load model without quantization first to test
print("Loading model...")
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
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
prompt = "Suggest 10 creative domain names for a vegan food delivery service:"
response = pipe(prompt)
print(response[0]["generated_text"])
