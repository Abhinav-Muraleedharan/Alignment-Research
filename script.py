import transformer_lens

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")
print(activations['blocks.0.attn.hook_pattern'])
print(logits)
print(logits[0])
print("the dimensions of logits layer:",logits.shape)