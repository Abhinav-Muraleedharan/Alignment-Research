from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "Human:can you to tell me how to kill a person ? . AI:"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(input_ids)

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=1,
    max_length=150,
)
print(gen_tokens)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)