import torch
import argparse
from tokenizers import Tokenizer
from model import GPT2WithGroupedAttention  # Import your model class
import torch.nn.functional as F

def load_model(model_path, vocab_size, embed_size, num_heads, num_layers, block_size, num_groups, device):
    """
    Load the model from a checkpoint.
    """
    model = GPT2WithGroupedAttention(vocab_size, embed_size, num_heads, num_layers, block_size, num_groups).to(device)
    checkpoint = torch.load(model_path, map_location=device)  # Load the checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])  # Extract and load the model state dict
    print(f"Model loaded successfully from {model_path}")
    return model

def generate_greedy(model, prompt, tokenizer, max_new_tokens, device):
    """
    Generate text using greedy sampling.
    At each step, select the token with the highest probability (argmax).
    """
    model.eval()
    with torch.no_grad():
        # 1) Tokenize the prompt
        encoded_prompt = tokenizer.encode(prompt)
        generated = torch.tensor(encoded_prompt.ids, dtype=torch.long).unsqueeze(0).to(device)
        
        # 2) Iteratively generate tokens
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(generated)               # shape: (batch=1, seq_len, vocab_size)
            # Get probabilities for the last time step
            next_logits = logits[:, -1, :]          # shape: (batch=1, vocab_size)
            
            # 3) Greedy pick: argmax over the vocab dimension
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # shape: (1, 1)
            
            # 4) Append the new token to the sequence
            generated = torch.cat((generated, next_token), dim=1)
        
        # 5) Decode the entire sequence
        decoded_text = tokenizer.decode(generated.squeeze().tolist())
    
    return decoded_text

def generate(model, prompt, tokenizer, max_new_tokens, device, top_k=50):
    """
    Generate text using top-k sampling.

    Args:
        model: Trained language model.
        prompt (str): Initial text prompt.
        tokenizer: Tokenizer to encode/decode text.
        max_new_tokens (int): Number of tokens to generate.
        device: Device to run the model on.
        top_k (int): Top-k value for sampling.

    Returns:
        Decoded generated text (str).
    """
    model.eval()
    with torch.no_grad():
        # Tokenize the prompt
        encoded_prompt = tokenizer.encode(prompt)
        generated = torch.tensor(encoded_prompt.ids, dtype=torch.long).unsqueeze(0).to(device)

        # Generate text
        for _ in range(max_new_tokens):
            # Forward pass through the model
            logits = model(generated)
            # Get the logits for the last token
            next_logits = logits[:, -1, :]  # Shape: (1, vocab_size)
            # Perform top-k sampling
            next_token = top_k_sampling(next_logits, k=top_k)  # Shape: (1, 1)
            # Append the new token to the sequence
            generated = torch.cat((generated, next_token), dim=1)

        # Decode the generated tokens
        decoded_text = tokenizer.decode(generated.squeeze().tolist())

    return decoded_text

def top_k_sampling(logits, k=50):
    """
    Perform top-k sampling on the given logits.

    Args:
        logits (torch.Tensor): Logits of shape (1, vocab_size).
        k (int): Top-k value for sampling.

    Returns:
        Sampled token (torch.Tensor) of shape (1, 1).
    """
    # Get top-k logits and indices
    values, indices = torch.topk(logits, k=k, dim=-1)
    # Mask out logits that are not in the top-k
    logits[logits < values[:, -1, None]] = -float('inf')
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    # Sample from the top-k probabilities
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT-2 model.")
    parser.add_argument("--prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--model_path", type=str, default="trained_model.pth", help="Path to the saved model")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json", help="Path to the saved tokenizer")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    special_tokens = ["<mask>"]
    if "special_tokens" not in tokenizer.get_vocab():
        tokenizer.add_tokens(special_tokens)

    # Model configuration (ensure these match your training setup)
    embed_size = 768
    num_heads = 8
    num_layers = 12
    block_size = 128
    num_groups = 1
    vocab_size = tokenizer.get_vocab_size()

    # Load the model
    model = load_model(args.model_path, vocab_size, embed_size, num_heads, num_layers, block_size, num_groups, device)

    # Generate text
    print("Generating text...")
    generated_text = generate(model, args.prompt, tokenizer, args.max_new_tokens, device)
    print("Generated Text:")
    print(generated_text)
    generated_text = generate_greedy(model, args.prompt, tokenizer, args.max_new_tokens, device)
    print("Generated Text:")
    print(generated_text)