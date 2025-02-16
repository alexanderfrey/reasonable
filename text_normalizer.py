import re
from typing import List, Dict, Optional
from functools import partial
import unicodedata

class TextNormalizer:
    """Text normalization pipeline for cleaning training data."""
    
    def __init__(self, 
                 remove_gutenberg_boilerplate: bool = True,
                 normalize_unicode: bool = True,
                 normalize_whitespace: bool = True,
                 normalize_quotes: bool = True,
                 normalize_dashes: bool = True,
                 remove_headers_footers: bool = True):
        """
        Initialize the text normalizer with configurable options.
        
        Args:
            remove_gutenberg_boilerplate: Remove Project Gutenberg license and header text
            normalize_unicode: Convert unicode to ASCII where possible
            normalize_whitespace: Standardize whitespace and newlines
            normalize_quotes: Standardize different types of quotes
            normalize_dashes: Standardize different types of dashes/hyphens
            remove_headers_footers: Remove common headers/footers like page numbers
        """
        self.config = {
            'remove_gutenberg_boilerplate': remove_gutenberg_boilerplate,
            'normalize_unicode': normalize_unicode,
            'normalize_whitespace': normalize_whitespace,
            'normalize_quotes': normalize_quotes,
            'normalize_dashes': normalize_dashes,
            'remove_headers_footers': remove_headers_footers
        }
        
        # Common Project Gutenberg patterns
        self.gutenberg_header_pattern = re.compile(
            r"(\*\*\* START OF (THIS|THE) PROJECT GUTENBERG.*?\*\*\*)",
            re.DOTALL | re.IGNORECASE
        )
        self.gutenberg_footer_pattern = re.compile(
            r"(\*\*\* END OF (THIS|THE) PROJECT GUTENBERG.*?)\Z",
            re.DOTALL | re.IGNORECASE
        )
        
        # Common header/footer patterns
        self.header_footer_patterns = [
            r"^\s*Page \d+\s*$",  # Page numbers
            r"^\s*Chapter \d+\s*$",  # Chapter headers
            r"^\s*\[\d+\]\s*$",  # Reference numbers
            r"^\s*\[Illustration:.*?\]",  # Illustration captions
        ]
        
    def remove_gutenberg_boilerplate(self, text: str) -> str:
        """Remove Project Gutenberg header and footer text."""
        # Remove header
        text = self.gutenberg_header_pattern.sub("", text)
        # Remove footer
        text = self.gutenberg_footer_pattern.sub("", text)
        return text.strip()
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to their closest ASCII representation."""
        # Normalize to NFKD form and remove non-ASCII
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Standardize whitespace and newlines."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Remove spaces at start/end of lines
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n +', '\n', text)
        return text.strip()
    
    def normalize_quotes(self, text: str) -> str:
        """Standardize different types of quotes."""
        # Convert fancy quotes to standard quotes
        quote_pairs = {
            '\u201c': '"',  # left double quotation mark
            '\u201d': '"',  # right double quotation mark
            '\u2018': "'",  # left single quotation mark
            '\u2019': "'",  # right single quotation mark
            '`': "'",      # backtick
            '\xb4': "'",   # acute accent
            '\u2032': "'", # prime
            '\u201e': '"', # double low-9 quotation mark
            '\u201f': '"', # double high-reversed-9 quotation mark
        }
        for fancy, plain in quote_pairs.items():
            text = text.replace(fancy, plain)
        return text
    
    def normalize_dashes(self, text: str) -> str:
        """Standardize different types of dashes and hyphens."""
        # Convert various dashes to standard hyphen or em-dash
        text = re.sub(r'‒|–|—|―', '-', text)  # Convert to hyphen
        text = re.sub(r'-{2,}', '—', text)    # Convert double hyphen to em-dash
        return text
    
    def remove_headers_footers(self, text: str) -> str:
        """Remove common headers and footers."""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if not any(re.match(pattern, line) for pattern in self.header_footer_patterns):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def clean_text(self, text: str) -> str:
        """Apply all enabled cleaning steps to the text."""
        if self.config['remove_gutenberg_boilerplate']:
            text = self.remove_gutenberg_boilerplate(text)
            
        if self.config['normalize_unicode']:
            text = self.normalize_unicode(text)
            
        if self.config['normalize_whitespace']:
            text = self.normalize_whitespace(text)
            
        if self.config['normalize_quotes']:
            text = self.normalize_quotes(text)
            
        if self.config['normalize_dashes']:
            text = self.normalize_dashes(text)
            
        if self.config['remove_headers_footers']:
            text = self.remove_headers_footers(text)
            
        return text
