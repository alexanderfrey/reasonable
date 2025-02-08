import re
import os
from typing import List, Tuple
import argparse

def detect_header_footer(lines: List[str]) -> Tuple[int, int]:
    """Detect Gutenberg header and footer boundaries."""
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT"
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg's"
    ]
    
    header_end = 0
    footer_start = len(lines)
    
    for i, line in enumerate(lines):
        if any(marker in line for marker in start_markers):
            header_end = i + 1
            break
            
    for i in range(len(lines) - 1, -1, -1):
        if any(marker in lines[i] for marker in end_markers):
            footer_start = i
            break
            
    return header_end, footer_start

def clean_text(text: str) -> str:
    """Clean and normalize text content for LLM training."""
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove emails and URLs
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove non-printing characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Replace Unicode spaces/separators with standard space
    text = re.sub(r'[\u2000-\u200F\u2028-\u202F\u205F-\u206F]', ' ', text)
    # Replace various types of dashes and hyphens with standard dash
    text = re.sub(r'[‐‑‒–—―]', '-', text)
    
    # Normalize quotes
    text = re.sub(r'[""‟]', '"', text)
    text = re.sub(r'[''‛]', "'", text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove multiple newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove footnotes and references [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove ASCII art and decorative lines
    text = re.sub(r'[=\-*_]{3,}', '', text)
    
    # Fix common OCR errors
    text = re.sub(r'l\b', 'I', text)  # Lone 'l' at word boundary
    text = re.sub(r'\bI([^a-zA-Z])', 'l\\1', text)  # 'I' at start when not a word
    
    # Remove repeated punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Standardize ellipsis
    text = re.sub(r'\.{3,}', '...', text)
    
    # Remove text in parentheses (often annotations)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # Remove non-English characters while preserving common accented characters
    text = re.sub(r'[^\x00-\x7F\u00C0-\u00FF\u2018\u2019\u201C\u201D]', '', text)
    
    # Normalize whitespace around quotes
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)
    
    # Remove sections that are likely tables or structured data
    text = re.sub(r'(?:\s*\|[^|]*){3,}', '', text)
    
    return text.strip()

def process_file(input_path: str, output_path: str, min_length: int = 100) -> None:
    """Process a single file."""
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Remove header and footer
    header_end, footer_start = detect_header_footer(lines)
    content = ''.join(lines[header_end:footer_start])
    
    # Clean the text
    cleaned_text = clean_text(content)
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    
    # Filter out short paragraphs and those that look like chapter headers
    valid_paragraphs = [p for p in paragraphs 
                       if len(p) >= min_length and 
                       not re.match(r'^(chapter|book|volume)\s+[IVXLCDM\d]+', p.lower())]
    
    # Write cleaned text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(valid_paragraphs))

def main():
    parser = argparse.ArgumentParser(description='Clean Gutenberg text files for LLM training')
    parser.add_argument('input_dir', help='Input directory containing raw text files')
    parser.add_argument('output_dir', help='Output directory for cleaned files')
    parser.add_argument('--min-length', type=int, default=100,
                       help='Minimum paragraph length to keep')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, f'clean_{filename}')
            print(f'Processing {filename}...')
            process_file(input_path, output_path, args.min_length)

if __name__ == '__main__':
    main()