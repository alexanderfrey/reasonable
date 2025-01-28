import os
from pathlib import Path
import json
import random
from typing import List, Dict, Set
import argparse
from text_processor import TextProcessor
from tqdm import tqdm
import openai
from time import sleep
from itertools import islice
from utils import vllm_request
import asyncio
from pathlib import Path
from typing import List, Dict
from tqdm.asyncio import tqdm as async_tqdm
import aiohttp
from time import sleep
from asyncio import Semaphore


def save_examples(
    examples: List[Dict], output_file: str, format: str = "jsonl"
) -> None:
    """
    Save generated examples to a file in either JSON or JSONL format.

    Args:
        examples: List of dictionaries containing the training examples
        output_file: Path to the output file
        format: Output format ('json' or 'jsonl')
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)

        elif format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"Successfully saved {len(examples)} examples to {output_file}")

    except Exception as e:
        print(f"Error saving examples to {output_file}: {str(e)}")
        raise


def make_associations_helper(text: str) -> str:
    """
    Generates a prompt to identify and analyze meaningful associations within the given text.
    Designed to reveal patterns, connections, and resonances while maintaining focus on textual evidence.
    """
    helper = f"""Identify and analyze meaningful associations within this text. Consider the following types of connections:

Types of Associations to Consider:
- Recurring motifs or symbols that echo throughout the text
- Thematic patterns that link different passages
- Sensory or emotional resonances between elements
- Conceptual bridges between ideas or scenes
- Networks of related imagery or metaphors
- Parallel structures or situations
- Contrasting elements that illuminate each other

TEXT:
{text}

Requirements for Your Analysis:
1. Identify 2-3 significant associations from the text
2. For each association:
   - Clearly state the connected elements
   - Explain how they relate or create meaning together
   - Quote or reference specific passages as evidence
   - Describe the significance of this connection
3. Focus exclusively on connections within this text
4. Avoid external references or background knowledge
5. Express each association in 2-3 clear, well-structured sentences

Write in a precise, analytical style that:
- Draws explicit connections between textual elements
- Uses specific evidence to support each association
- Explains the significance of each connection
- Maintains focus on internal textual relationships"""

    return helper


def make_thinking_helper(text: str) -> str:
    """
    Generates a prompt to produce a single analytical thought about the given text.
    Designed to elicit specific, text-supported insights while avoiding background knowledge.
    """
    helper = f"""Analyze this text and share ONE insightful thought about it. Consider aspects like:
- Patterns in language or structure
- Relationships between ideas
- Implicit assumptions or implications
- Notable contrasts or parallels
- Literary devices and their effects

TEXT:
{text}

Requirements for your thought:
1. Keep it focused and specific - no broad generalizations
2. Point to specific evidence from the text
3. Go beyond surface-level summary to deeper meaning
4. Express it in 2-3 clear, well-structured sentences
5. Base your analysis solely on this text, without referencing external context or background knowledge

Write in a precise, analytical style that directly connects your insight to textual evidence."""

    return helper


def generate_reasoning_prompt(text: str) -> str:
    """
    Generates a prompt that instructs an LLM to reason about a given text snippet.

    Args:
        text (str): The text snippet to analyze

    Returns:
        str: A formatted prompt for the LLM
    """
    prompt = f"""Please analyze the following text carefully and provide your reasoning:

{text}

Follow these steps in your analysis:

1. First, identify the key concepts and main ideas presented in the text.
   - What are the central themes?
   - What are the core arguments or claims being made?

2. Examine the logical structure:
   - How are ideas connected?
   - What evidence or support is provided?
   - Are there any assumptions or implicit premises?

3. Consider the context:
   - What background knowledge is relevant?
   - Are there any important cultural, historical, or domain-specific references?

4. Evaluate the reasoning:
   - Is the logic sound and valid?
   - Are there any potential fallacies or gaps in reasoning?
   - How strong is the supporting evidence?

5. Draw conclusions:
   - What are the main implications?
   - What follow-up questions arise?
   - What additional context would be helpful?

Please structure your response as a clear, step-by-step analysis. For each point, explain your reasoning and provide specific examples from the text to support your observations.

Remember to:
- Think carefully about each step before moving to the next
- Be explicit about your reasoning process
- Consider alternative interpretations
- Acknowledge any uncertainties or limitations in your analysis
- Connect your observations to form a coherent overall assessment

Begin your analysis:"""

    return prompt


def chunk_sentences(sentences: List[str], n: int) -> List[List[str]]:
    """
    Split sentences into chunks of size n.

    Args:
        sentences: List of sentences
        n: Number of sentences per chunk
    Returns:
        List of sentence chunks
    """
    chunks = []
    iterator = iter(sentences)
    while chunk := list(islice(iterator, n)):
        if len(chunk) == n:  # Only keep full chunks
            chunks.append(chunk)
    return chunks


async def append_example(example: Dict, output_file: str) -> None:
    """
    Append a single example to the output file in JSONL format.

    Args:
        example: Dictionary containing the training example
        output_file: Path to the output file
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Append the example to the file
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Error appending example to {output_file}: {str(e)}")
        raise


def load_processed_files(output_file: str) -> Set[str]:
    """
    Load the set of files that have already been processed from existing training examples.

    Args:
        output_file: Path to the existing training examples file

    Returns:
        Set of filenames that have already been processed
    """
    processed_files = set()

    if not Path(output_file).exists():
        return processed_files

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    example = json.loads(line)
                    if "source_file" in example:
                        processed_files.add(example["source_file"])
                except json.JSONDecodeError:
                    continue

        print(f"Found {len(processed_files)} already processed files")
        return processed_files

    except Exception as e:
        print(f"Error loading processed files from {output_file}: {str(e)}")
        return set()


def get_unprocessed_files(directory_path: str, processed_files: Set[str]) -> List[Path]:
    """
    Get list of files that haven't been processed yet.

    Args:
        directory_path: Path to directory containing text files
        processed_files: Set of already processed filenames

    Returns:
        List of Path objects for unprocessed files
    """
    all_files = list(Path(directory_path).glob("*.txt"))
    print(all_files)
    unprocessed_files = [f for f in all_files if f.name not in processed_files]

    print(
        f"Found {len(unprocessed_files)} unprocessed files out of {len(all_files)} total files"
    )
    return unprocessed_files


async def generate_training_examples(
    directory_path: str,
    text_processor: TextProcessor,
    output_file: str,
    sentences_per_example: int = 5,
    max_seq_length: int = 512,
    model: str = "microsoft/phi-4",
    max_retries: int = 3,
    delay_between_calls: float = 1.0,
    max_concurrent_requests: int = 5,
) -> None:
    # Load already processed files
    processed_files = load_processed_files(output_file)
    print(processed_files)
    # Get list of unprocessed files
    unprocessed_files = get_unprocessed_files(directory_path, processed_files)

    if not unprocessed_files:
        print("No new files to process")
        return

    examples_count = 0

    # Semaphore to control concurrent API requests
    sem = Semaphore(max_concurrent_requests)

    async def process_chunk(chunk: List[str], txt_file: Path, pbar) -> Dict:
        """Process a single chunk of sentences"""
        input_text = " ".join(chunk)

        async with sem:
            for attempt in range(max_retries):
                try:
                    prompt = make_thinking_helper(input_text)
                    thought = await vllm_request(
                        prompt=prompt,
                        context=None,
                        openai_api_base="http://100.68.45.88:8080/v1",
                        model_id=model,
                    )

                    if thought:
                        example = {
                            "instruction": "Produce analytical thought",
                            "input": input_text,
                            "thought": thought,
                            "source_file": str(txt_file.name),
                            "type": "summarization",
                            "metadata": {
                                "num_sentences": len(chunk),
                                "model_used": model,
                                "prompt": prompt,
                            },
                        }

                        # Immediately save the example
                        await append_example(example, output_file)
                        pbar.update(1)
                        return example

                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay_between_calls)

            pbar.update(1)
            return None

    total_files = len(unprocessed_files)
    print(f"Processing {total_files} new text files")

    # Use regular tqdm for file progress
    pbar_files = tqdm(total=total_files, desc="Processing files", position=0)

    for txt_file in unprocessed_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as file:
                text = file.read()  # [:5000]
                print(f"\nProcessing file: {txt_file.name}")

                sentences = text_processor.process_text_parallel(
                    text, show_progress=True
                )

                sentence_chunks = chunk_sentences(sentences, sentences_per_example)

                # Use regular tqdm for chunks
                pbar_chunks = tqdm(
                    total=len(sentence_chunks),
                    desc="Processing chunks",
                    position=1,
                    leave=False,
                )

                chunk_tasks = [
                    process_chunk(chunk, txt_file, pbar_chunks)
                    for chunk in sentence_chunks
                ]

                results = await asyncio.gather(*chunk_tasks)
                examples_count += sum(1 for r in results if r is not None)
                await asyncio.sleep(delay_between_calls / max_concurrent_requests)

                pbar_chunks.close()
                pbar_files.update(1)

        except Exception as e:
            print(f"Error processing {txt_file}: {str(e)}")
            pbar_files.update(1)

    pbar_files.close()
    print(
        f"\nGenerated and saved {examples_count} new training examples to {output_file}"
    )


# Update the main function to handle async execution
async def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM training examples with summaries"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing text files",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output file path"
    )
    parser.add_argument(
        "--sentences_per_example",
        type=int,
        default=10,
        help="Number of sentences to group together",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--model", type=str, default="microsoft/phi-4", help="LLM model to use"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size for text processor"
    )
    parser.add_argument(
        "--n_process",
        type=int,
        default=None,
        help="Number of processes for parallel processing",
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )

    args = parser.parse_args()

    # Initialize text processor
    text_processor = TextProcessor(batch_size=args.batch_size, n_process=args.n_process)

    # Generate examples
    print(f"Generating training examples from {args.input_dir}...")
    await generate_training_examples(
        directory_path=args.input_dir,
        text_processor=text_processor,
        output_file=args.output_file,
        sentences_per_example=args.sentences_per_example,
        max_seq_length=args.max_seq_length,
        model=args.model,
        max_concurrent_requests=args.max_concurrent_requests,
    )


if __name__ == "__main__":
    asyncio.run(main())
