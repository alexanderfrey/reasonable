import torch
import argparse
from tokenizers import Tokenizer
import torch.nn.functional as F
from model import GPT2WithGroupedAttention  # Import your model class
from typing import List

def load_model(model_path, vocab_size, embed_size, num_heads, num_layers, block_size, num_groups, device):
    """
    Load the model from a checkpoint.
    """
    model = GPT2WithGroupedAttention(vocab_size, embed_size, num_heads, num_layers, block_size, num_groups).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Important for inference
    print(f"Model loaded successfully from {model_path}")
    return model

def fill_masks(model, masked_prompt: str, tokenizer, device, top_k=5):
    """
    Given a prompt with <mask> tokens, predict the top-k candidates for each <mask>.
    
    Args:
        model: Trained model that supports masked tokens.
        masked_prompt (str): A string containing one or more <mask> tokens.
        tokenizer: Tokenizer that recognizes <mask>.
        device: PyTorch device (cpu, cuda, mps, etc.)
        top_k (int): Number of top predictions to display for each <mask>.

    Returns:
        mask_predictions (List[List[str]]): For each <mask> in the prompt, a list of top-k predicted tokens.
    """
    # 1. Encode the prompt (including <mask> tokens)
    encoded = tokenizer.encode(masked_prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)  # shape: (1, seq_len)

    # 2. Forward pass through the model
    with torch.no_grad():
        logits = model(input_ids)  # shape: (batch_size=1, seq_len, vocab_size)

    # 3. Identify positions of <mask> in the input
    mask_id = tokenizer.token_to_id("<mask>")
    # mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True).tolist()  # list of indices
    mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=False)
    mask_positions = mask_positions.squeeze(1).tolist()

    mask_predictions = []

    for pos in mask_positions:
        # 4. Extract the logits for this <mask> position
        #    shape: (vocab_size,)
        token_logits = logits[0, pos, :]  

        # 5. Compute probabilities and get top-k
        probs = F.softmax(token_logits, dim=-1)        # (vocab_size,)
        top_k_ids = torch.topk(probs, k=top_k).indices  # (top_k,)

        # 6. Decode each top-k token ID
        predicted_tokens = [tokenizer.decode([tk_id.item()]) for tk_id in top_k_ids]
        mask_predictions.append(predicted_tokens)

    return mask_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a Masked Token Strategy with a GPT-like model.")
    parser.add_argument("prompt", type=str, help="Input prompt containing <mask> tokens.")
    parser.add_argument("--model_path", type=str, default="trained_model.pth", help="Path to the saved model")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json", help="Path to the saved tokenizer")
    parser.add_argument("--embed_size", type=int, default=256, help="Model embedding size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--block_size", type=int, default=64, help="Block size (max sequence length)")
    parser.add_argument("--num_groups", type=int, default=1, help="Number of grouped attentions")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions for each <mask>")
    args = parser.parse_args()

    # 1. Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # 3. Load model
    model = load_model(
        model_path=args.model_path,
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        block_size=args.block_size,
        num_groups=args.num_groups,
        device=device
    )

    # 4. Fill the <mask> tokens
    masked_prompt = args.prompt
    print(f"\nMasked Prompt: {masked_prompt}")

    predictions = fill_masks(model, masked_prompt, tokenizer, device, top_k=args.top_k)

    # 5. Display results
    mask_indices = 1
    for top_words in predictions:
        print(f"\nFor <mask> #{mask_indices}, top {args.top_k} predictions:")
        for rank, token in enumerate(top_words, start=1):
            print(f"{rank}: {token}")
        mask_indices += 1