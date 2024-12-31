from tokenizers import ByteLevelBPETokenizer
import glob, os

# Paths to text files
text_files_directory = "./text_files/"
files = glob.glob(f"{text_files_directory}/*.txt")

# Read all text files and create an iterator for training
def file_iterator(file_paths):
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            yield f.read()

# Initialize the ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer directly from the file iterator
tokenizer.train_from_iterator(file_iterator(files), vocab_size=52000, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"])

# Save the tokenizer
tokenizer_path = os.path.join(os.path.dirname(__file__), 'byte_level_bpe')
os.makedirs(tokenizer_path, exist_ok=True)
tokenizer.save_model(tokenizer_path)

print(f"Tokenizer trained and saved to {tokenizer_path}")