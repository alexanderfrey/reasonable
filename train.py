import os, random, math 
import glob
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2Tokenizer
from model import GPT2WithGroupedAttentionRotary  # Import the model from model.py
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_preparation import prepare_data
from torch.cuda.amp import autocast, GradScaler
import numpy as np




def decode_example(X, Y, tokenizer, index=9):
    """
    Decodes a single example from the prepared data.

    Args:
    - X (torch.Tensor): Input sequences.
    - Y (torch.Tensor): Target sequences.
    - tokenizer: Tokenizer object for decoding.
    - index (int): The index of the example to decode.

    Returns:
    - input_sequence (str): Decoded input sequence.
    - target_sequence (str): Decoded target sequence.
    """
    # Decode input sequence
    input_sequence = tokenizer.decode(X[index].tolist())

    # Filter out invalid token IDs (-100) from the target sequence
    valid_target_ids = [token_id for token_id in Y[index].tolist() if token_id >= 0]
    target_sequence = tokenizer.decode(valid_target_ids)

    return input_sequence, target_sequence

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Split data into training and validation sets
def split_data(X, Y, test_size=0.2):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, X_val, Y_train, Y_val

def create_dataloader(X, Y, batch_size):
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Read text from all `.txt` files in the specified directory
def read_text_from_directory(directory):
    files = glob.glob(os.path.join(directory, "*.txt"))
    text = ""
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            text += f.read() + "\n"
    return text

# Text generation function
def generate(model, prompt, max_new_tokens, tokenizer, device):
    model.eval()
    # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt)
    generated = torch.tensor(encoded_prompt.ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        logits = model(generated)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    # Decode the generated sequence
    decoded = tokenizer.decode(generated.squeeze().tolist())
    return decoded

def compute_accuracy(logits, targets):
    """
    Computes the accuracy for masked token prediction, ignoring positions where targets == -100.
    """
    mask = targets != -100
    preds = torch.argmax(logits, dim=-1)  # Get the index of the max logit
    correct = (preds[mask] == targets[mask]).sum().item()  # Compare only masked positions
    total = mask.sum().item()  # Total number of valid positions
    return correct, total

def compute_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param.abs() < 1e-6).sum().item()
    return zero_params / total_params


def generate_text(model, tokenizer, device, prompt="The quick brown fox", max_new_tokens=30):
    """
    A simple text generation function that appends tokens to the prompt
    one at a time, sampling from the model's output distribution.
    
    Args:
        model: Your GPT-like model
        tokenizer: Tokenizer used to encode/decode text
        device: Torch device
        prompt (str): Initial text prompt
        max_new_tokens (int): Number of tokens to generate

    Returns:
        A string of the generated text
    """
    model.eval()
    # Encode the prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)  # shape: (batch_size=1, seq_len, vocab_size)
            # Take the last token’s logits and make a distribution
            probs = F.softmax(logits[:, -1, :], dim=-1)
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # Append the sampled token to the sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)

    # Decode the entire sequence
    generated_text = tokenizer.decode(input_ids.squeeze().tolist())
    return generated_text

def train_with_batches(
    model,
    train_loader,
    val_loader,
    vocab_size,
    device,
    tokenizer,
    optimizer,
    scheduler,
    *,
    epochs=1,
    max_steps=None,
    generate_every=100,
    generate_prompt="The <mask> brown <mask> jumped over the <mask> dog.",
    max_gen_tokens=20,
    sparsity_alpha=1e-5,
    start_epoch=0,
    save_path="best_model.pth"
):
    """
    Train the model using batches with masked token prediction strategy.
    Save only the best model based on validation loss.
    Includes AMP for CUDA, while keeping compatibility with MPS.

    Args:
        (same as before)
        save_path (str): File path for saving the best model based on validation loss.
    """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    global_step = 0  # Track total training steps across epochs
    best_val_loss = float("inf")  # Initialize the best validation loss as infinity

    # Use AMP only if CUDA is available
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, start_epoch + epochs):
        # ---------------------
        #      Training
        # ---------------------
        model.train()
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

        for batch_X, batch_Y in train_progress_bar:
            # 1) Check if we've exceeded max_steps
            if max_steps is not None and global_step >= max_steps:
                print(f"Reached {max_steps} steps, stopping training.")
                break

            global_step += 1

            batch_X, batch_Y = batch_X.to(device, non_blocking=True), batch_Y.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Enable mixed precision if using CUDA
            with autocast(enabled=use_amp):
                logits = model(batch_X)
                loss = criterion(logits.view(-1, vocab_size), batch_Y.view(-1))

                # Add L1 sparsity regularization
                l1_regularization = sparsity_alpha * sum(
                    torch.abs(param).sum()
                    for name, param in model.named_parameters()
                    if "weight" in name
                )
                total_loss = loss + l1_regularization

            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()

            train_progress_bar.set_postfix(train_loss=total_loss.item())

            # Mid-training generation
            if generate_every is not None and generate_every > 0 and global_step % generate_every == 0:
                print(f"\n[Step {global_step}] Generating text for prompt:\n{generate_prompt}")
                sample_output = generate_text(
                    model,
                    tokenizer,
                    device,
                    prompt=generate_prompt,
                    max_new_tokens=max_gen_tokens
                )
                print(f"[Step {global_step}] Generated text:\n{sample_output}\n")
                model.train()  # switch back to train mode

        if max_steps is not None and global_step >= max_steps:
            print(f"Completed training after reaching {global_step} steps.")
            break

        # End of epoch: compute average train loss
        train_loss /= len(train_loader)
        train_ppl = math.exp(train_loss) if train_loss < 20 else float('inf')

        scheduler.step()

        # ---------------------
        #    Validation
        # ---------------------
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", leave=False)

        with torch.no_grad():
            for batch_X, batch_Y in val_progress_bar:
                batch_X, batch_Y = batch_X.to(device, non_blocking=True), batch_Y.to(device, non_blocking=True)

                logits = model(batch_X)
                loss = criterion(logits.view(-1, vocab_size), batch_Y.view(-1))
                val_loss += loss.item()

                # Compute accuracy for masked tokens only
                mask = (batch_Y != -100)
                masked_preds = torch.argmax(logits, dim=-1)[mask]
                masked_labels = batch_Y[mask]
                correct += (masked_preds == masked_labels).sum().item()
                total += mask.sum().item()

                val_progress_bar.set_postfix(val_loss=loss.item())

        val_loss /= len(val_loader)
        val_ppl = math.exp(val_loss) if val_loss < 20 else float('inf')
        val_accuracy = correct / total if total > 0 else 0.0

        try:
            sparsity = compute_sparsity(model)
        except NameError:
            sparsity = 0.0

        # Print epoch results
        print(
            f"Epoch {epoch + 1} (Step {global_step}) | "
            f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}), "
            f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}), "
            f"Val Accuracy: {val_accuracy:.4f}, "
            f"Sparsity: {sparsity:.4f}"
        )

        # Save the model only if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss improved to {val_loss:.4f}. Saving the best model...")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, save_path)

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }, path)
    print(f"Checkpoint saved to {path} with epoch {epoch}")

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Load model, optimizer, and scheduler states from a checkpoint.
    """
    checkpoint = torch.load(path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optionally load optimizer and scheduler states
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}, resuming at epoch {epoch}")
    return epoch  # Return the exact epoch saved

def get_batch(split, device, data_dir, block_size, batch_size):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train GPT-3 model with CLI parameters.")
    parser.add_argument("--text_files_directory", type=str, default="./text_files", help="Path to the text files directory")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for training sequences")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training")
    parser.add_argument("--model_save_path", type=str, default="trained_model.pth", help="Path to save the trained model")
    parser.add_argument("--model_load_path", type=str, default=None, help="Path to a pre-trained model file to continue training")
    parser.add_argument("--prompt", type=str, default="My name is", help="Prompt for text generation")
    args = parser.parse_args()

    from tokenizers import ByteLevelBPETokenizer

    # Load the trained ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer(
        "./byte_level_bpe/vocab.json",
        "./byte_level_bpe/merges.txt"
    )

    # Special tokens to be verified
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]

    # Verify that special tokens are correctly added
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            print(f"Error: Token {token} was not added to the vocabulary.")
        else:
            print(f"Token {token} has ID: {token_id}")

    # Save the updated tokenizer (if required)
    # tokenizer.save_model("./updated_byte_level_bpe")

    print("Tokenizer loaded and verified.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)


    # Load text data
    text = read_text_from_directory(args.text_files_directory)#[:100000]
    block_size = args.block_size

    strategy_names = ["next_token"]#, , "span_masking", "masked_token","whole_word_masking"
    combined_X, combined_Y = [], []

    for strategy_name in strategy_names:
        if strategy_name == "permuted_language":
            X, Y, _ = prepare_data(strategy_name, text, block_size, tokenizer, mask_probability=0.15)
        else:
            X, Y = prepare_data(strategy_name, text, block_size, tokenizer, mask_probability=0.15)

        combined_X.append(X)
        combined_Y.append(Y)

    X = torch.cat(combined_X, dim=0)
    Y = torch.cat(combined_Y, dim=0)

    X, Y = X.to(device).to(dtype=torch.long), Y.to(device).to(dtype=torch.long)

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=42)

    print(f"Number of training examples: {len(X_train)}")

    # Decode the 10th example for inspection
    # for i in range(15):
    #     input_seq, target_seq = decode_example(X, Y, tokenizer, index=1000+i)
    #     print(f"Input Sequence (X[{i}]): {input_seq}\n")
    #     print(f"Target Sequence (Y[{i}]): {target_seq}")
    #     print("\n\n\n")


    # Create DataLoaders for training and validation
    train_loader = create_dataloader(X_train, Y_train, args.batch_size)
    val_loader = create_dataloader(X_val, Y_val, args.batch_size)

    # Model configuration
    embed_size = 768
    num_heads = 12
    num_layers = 12
    num_groups = 1

    vocab_size = tokenizer.get_vocab_size()
    model = GPT2WithGroupedAttentionRotary(vocab_size, embed_size, num_heads, num_layers, block_size, num_groups, dropout=0.1, causal=True)


    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs)

    # Load model from file if specified
    start_epoch = 0
    if args.model_load_path:
        try:
            start_epoch = load_checkpoint(args.model_load_path, model, optimizer, scheduler)
            
        except Exception as e:
            print(f"Failed to load checkpoint from {args.model_load_path}: {e}")
            exit(1)
   
    if not args.model_load_path:
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    print("Model created.")

    print("Generated text:", generate_text(model, tokenizer, device, args.prompt, max_new_tokens=120))


    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    # Train with batches
    train_with_batches(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        device=device,
        epochs=args.epochs,
        max_steps=args.steps,
        optimizer=optimizer,
        scheduler=scheduler,
        sparsity_alpha=1e-5,
        start_epoch=start_epoch,
        save_path=args.model_save_path,
        generate_every=100,  # Generate text every 100 training steps
        generate_prompt="The house was build on",
        max_gen_tokens=20
    )

                
    # Generate text
    print("Generated text:", generate_text(model, tokenizer, device, args.prompt, max_new_tokens=120))