from transformers import AutoModelForCausalLM, AutoTokenizer

access_token = "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"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", device_map="auto", token = access_token)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
# model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
tokenizer.batch_decode(generated_ids)[0]