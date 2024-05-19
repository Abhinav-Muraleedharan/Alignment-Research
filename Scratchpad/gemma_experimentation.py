from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


access_token = "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"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16,  device_map="auto", token = access_token
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
