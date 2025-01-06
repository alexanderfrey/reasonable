from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()  # Set the model to evaluation mode


# Input prompt
prompt = "Artificial intelligence is"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")


# Generate text
output = model.generate(
    input_ids, 
    max_length=50,  # Maximum length of the generated text
    num_return_sequences=1,  # Number of sequences to generate
    temperature=0.7,  # Sampling temperature
    top_k=50,  # Top-k sampling
    top_p=0.9,  # Top-p (nucleus) sampling
)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)