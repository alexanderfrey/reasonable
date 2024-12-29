from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import glob

# Paths to text files
text_files_directory = "./text_files/"
files = glob.glob(f"{text_files_directory}/*.txt")

# Initialize a tokenizer with a BPE model
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# Use ByteLevel pre-tokenizer
tokenizer.pre_tokenizer = ByteLevel()

# Define a trainer
trainer = BpeTrainer(
    vocab_size=50257,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
        "<|endoftext|>"
    ]
)

# Train the tokenizer
tokenizer.train(files, trainer)

# Post-processing to handle ByteLevel
# This ensures trailing spaces are replaced with 'Ġ' etc., matching GPT-2 style
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor

tokenizer.decoder = ByteLevelDecoder()
tokenizer.post_processor = ByteLevelProcessor()

# Save the tokenizer
tokenizer.save("byte_level_bpe.json")