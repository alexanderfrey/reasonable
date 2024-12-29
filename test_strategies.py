import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing
from sklearn.model_selection import train_test_split
from train import (create_dataloader, read_text_from_directory)
from data_preparation import prepare_data

def custom_decode_target(tokenizer, target_tokens):
    """
    Decode target tokens, showing <mask> whenever a token is -100.
    Otherwise, decode the token ID to its string representation.
    """
    decoded_output = []
    for tid in target_tokens:
        if tid == -100:
            decoded_output.append("<mask>")
        else:
            # Decode this single token ID
            token_str = tokenizer.decode_batch([[tid]])[0]
            decoded_output.append(token_str)

    # Join tokens with a space for readability
    return " ".join(decoded_output)


# Load existing tokenizer
tokenizer = Tokenizer.from_file("tokenizer_with_special_tokens.json")



# Verify `[MASK]` is now in the vocabulary
mask_id = tokenizer.token_to_id("<mask>")
print(f"[MASK] token ID: {mask_id}")
strategy_names = ["whole_word_masking"]
combined_X, combined_Y = [], []


text = read_text_from_directory('./text_files')[:100000]
block_size = 64
mask_probability = 0.15
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for strategy_name in strategy_names:
    X, Y = prepare_data(strategy_name, text, block_size, tokenizer, mask_probability=mask_probability)
    combined_X.append(X)
    combined_Y.append(Y)

X = torch.cat(combined_X, dim=0)
Y = torch.cat(combined_Y, dim=0)

X, Y = X.to(device).to(dtype=torch.int32), Y.to(device).to(dtype=torch.int32)

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=42)

# Example usage
example_idx = 0
input_tokens = X_train[example_idx].tolist()
target_tokens = Y_train[example_idx].tolist()

# Decode the input as usual (possibly showing special tokens)
input_seq = tokenizer.decode_batch([input_tokens], skip_special_tokens=False)[0]

# Use our custom decode function for the target
target_seq = custom_decode_target(tokenizer, target_tokens)

print("=== Visualization of Whole Word Masking ===")
print(f"Input Sequence (Masked):\n{input_seq}\n")
print(f"Target Sequence (Original with <mask> placeholders):\n{target_seq}\n")
