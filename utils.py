import pandas as pd
from pandas import DataFrame
import json, re, os, uuid, datetime
import requests
from typing import List, Dict, Any
from dataclasses import dataclass
from typing import Optional
import psycopg2
import aiohttp
import openai
from openai import OpenAI
from psycopg2.pool import ThreadedConnectionPool
from torch.utils.data import DataLoader, TensorDataset
import torch
import logging
from psycopg2.extras import DictCursor
import tiktoken
from contextlib import contextmanager


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def format_instruction_alpaca(sample):
    if sample.get("input"):
        return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}
"""
    else:
        return f"""### Instruction:
{sample['instruction']}
"""


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    if not string:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class DatabaseConfig:
    def __init__(
        self, host: str, database: str, user: str, password: str, port: int = 5432
    ):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        # You can implement environment variable loading here
        return cls(
            host="",
            database="",
            user="",
            password="",
            port=5432,
        )


@contextmanager
def get_db_connection(config: DatabaseConfig):
    conn = None
    try:
        conn = psycopg2.connect(
            host=config.host,
            database=config.database,
            user=config.user,
            password=config.password,
            port=config.port,
        )
        yield conn
    finally:
        if conn is not None:
            conn.close()


async def vllm_request(
    prompt: str,
    context: str,
    model_id: str,
    openai_api_base: str,
    OPENROUTER_FALLBACK: bool = True,
    max_tokens: int = 4096,
    temperature: float = 0.5,
    show_logs: bool = True,
    save_as_training_data: bool = False,
    messages: Optional[List[Dict[str, str]]] = None,
    task_description: str = "unspecified",
    meta_data: Optional[Dict] = None,
) -> Optional[str]:

    prompt_template = format_instruction_alpaca(
        {"instruction": prompt, "input": context}
    )

    if not messages:
        messages = [
            {"role": "user", "content": prompt_template},
        ]

    num_tokens = num_tokens_from_string(prompt_template, "cl100k_base")
    if show_logs:
        logging.info(
            f"vllm: Processing prompt with {num_tokens} tokens for task: {task_description}"
        )

    try:
        # Use aiohttp for async HTTP requests
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{openai_api_base}/chat/completions",
                json={
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers={"Authorization": "Bearer EMPTY"},
            ) as response:
                if response.status == 200:
                    completion = await response.json()
                    result = completion["choices"][0]["message"]["content"]

                    if save_as_training_data:
                        # Consider making this async too if needed
                        export_training_data_ps(
                            [
                                {
                                    "input": json.dumps(messages),
                                    "output": result,
                                    "created_at": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                    "prompt": prompt,
                                    "uuid": str(uuid.uuid4()),
                                    "system_prompt": None,
                                    "model_version": model_id,
                                    "task_description": task_description,
                                    "data": (
                                        json.dumps(meta_data) if meta_data else None
                                    ),
                                }
                            ]
                        )

                    if show_logs:
                        num_tokens = num_tokens_from_string(result, "cl100k_base")
                        logging.info(
                            f"vllm: Answered with {num_tokens} tokens for task: {task_description}"
                        )

                    return result
                else:
                    logging.error(f"vllm API error: {await response.text()}")

    except Exception as e:
        logging.error(f"vllm API error: {str(e)}", exc_info=True)

        if OPENROUTER_FALLBACK:
            try:
                # Make sure call_xai_api is also async
                result = await call_xai_api(
                    api_key="xai-YiFHGz5vD5phZNIBRKsbvpsSNPM4GxF4LEli1jUe7fqq1MQg7jQSSpQp9CchNxY6Z4ttD3arQ291uCNx",
                    prompt=prompt,
                    context=context,
                    temperature=0.5,
                    description="text_passage_thought",
                    save_as_training_data=True,
                )
                return result

            except Exception as fallback_e:
                logging.error(
                    f"Error during fallback call_xai_api: {str(fallback_e)}",
                    exc_info=True,
                )

    return None


def export_training_data_ps(data_items: List[Dict[str, Any]]) -> None:
    db_config = DatabaseConfig.from_env()

    with get_db_connection(db_config) as conn:
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                all_keys = sorted(set().union(*(item.keys() for item in data_items)))
                columns = ",".join(all_keys)

                args = ",".join(
                    cur.mogrify(
                        f"({','.join(['%s'] * len(all_keys))})",
                        tuple(item.get(key) for key in all_keys),
                    ).decode("utf-8")
                    for item in data_items
                )

                cur.execute(
                    f"INSERT INTO vector.training_data ({columns}) VALUES {args} ON CONFLICT DO NOTHING"
                )
                conn.commit()

                logging.info(
                    f"{len(data_items)} training_data written to postgres. "
                    f"Task: {data_items[0].get('task_description', 'N/A')} "
                    f"Model: {data_items[0].get('model_version', 'N/A')}"
                )
        except Exception as error:
            conn.rollback()
            logging.error("DatabaseError", exc_info=True)
            raise


def call_xai_api(
    api_key: str,
    prompt: str,
    context: Optional[str] = None,
    temperature: float = 0.0,
    stream: bool = False,
    save_as_training_data: bool = False,
    description: str = "general",
    model_version: str = "grok-2-latest",
    data: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Call xAI API and optionally save training data."""
    if not api_key:
        raise ValueError("API key is required")
    if not 0 <= temperature <= 1:
        raise ValueError("Temperature must be between 0 and 1")

    system_prompt = (
        system_prompt
        or f"You are a helpful assistant. Here is relevant data: {context or ''}"
    )

    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "model": model_version,
                "stream": stream,
                "temperature": temperature,
            },
            timeout=30,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        if save_as_training_data:
            export_training_data_ps(
                [
                    {
                        "input": context,
                        "output": content,
                        "created_at": datetime.now().isoformat(),
                        "prompt": prompt,
                        "uuid": str(uuid.uuid4()),
                        "system_prompt": system_prompt,
                        "model_version": model_version,
                        "task_description": description,
                        "data": json.dumps(data) if data else None,
                    }
                ]
            )

        return content

    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Invalid API response format: {str(e)}")


def create_data_loader(data, batch_size, tokenizer, shuffle=True):
    """
    Enhanced data loader creation with thought handling and tensor validation
    """
    processed_data = []

    # First pass: validate and process each item
    for idx, item in enumerate(data):
        processed_item = {}
        for key, value in item.items():
            # Handle raw text data
            if key == "raw_intermediate_answers":
                processed_item[key] = value
                continue

            # Convert to tensor if needed
            if not isinstance(value, torch.Tensor):
                processed_item[key] = torch.tensor(value, dtype=torch.long)
            elif value.dtype != torch.long:
                processed_item[key] = value.to(torch.long)
            else:
                processed_item[key] = value

            # Validate tensor values
            tensor = processed_item[key]
            if torch.any(tensor < 0):
                raise ValueError(f"Negative values found in {key} at index {idx}")
            if torch.any(tensor >= tokenizer.vocab_size):
                raise ValueError(f"Token ID >= vocab_size in {key} at index {idx}")

        processed_data.append(processed_item)

    # Create tensors for DataLoader
    try:
        questions = torch.stack([item["question"] for item in processed_data])
        contexts = torch.stack([item["context"] for item in processed_data])
        answers = torch.stack([item["answer"] for item in processed_data])

        # Handle intermediate answers (thoughts)
        if "intermediate_answers" in processed_data[0]:
            thoughts = torch.stack(
                [item["intermediate_answers"] for item in processed_data]
            )
            dataset = TensorDataset(questions, contexts, answers, thoughts)
        else:
            dataset = TensorDataset(questions, contexts, answers)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
        )

    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        print(
            f"Shapes - questions: {questions.shape}, contexts: {contexts.shape}, "
            f"answers: {answers.shape}"
        )
        if "thoughts" in locals():
            print(f"thoughts: {thoughts.shape}")
        raise


@dataclass
class ThoughtStep:
    number: int
    title: str
    content: str
    sub_points: List[Dict[str, Any]]
    correction: Optional[str] = None


class ThoughtProcessor:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thought_start_token = "<thought>"
        self.thought_end_token = "</thought>"
        self.title_start_token = "<title>"
        self.title_end_token = "</title>"
        self.value_start_token = "<value>"
        self.value_end_token = "</value>"

    def format_thought_step(self, step: ThoughtStep) -> str:
        """
        Formats a thought step with explicit token handling and better structure
        """
        # Start with thought token
        output_parts = [self.thought_start_token]

        # Add title with tokens
        output_parts.append(
            f"{self.title_start_token}{step.title}{self.title_end_token}"
        )

        if step.sub_points:
            output_parts.append(self.value_start_token)

        # Process sub-points with better formatting
        for point in step.sub_points:
            if "title" in point:
                # Main points with bullet
                output_parts.append(f"\n• {point['title']}")
                if "content" in point:
                    # Add content with proper indentation
                    output_parts.append(f"  {point['content']}")
                # Process sub-points with proper indentation
                for sub_point in point.get("sub_points", []):
                    output_parts.append(f"    - {sub_point}")
            elif "content" in point:
                output_parts.append(point["content"])

        # Add correction if present
        if step.correction:
            output_parts.append(f"\nSelf-Correction:\n{step.correction}")

        if step.sub_points:
            output_parts.append(self.value_end_token)

        # Close thought tag
        output_parts.append(self.thought_end_token)

        return "\n".join(output_parts)

    def tokenize_thought(self, thought_text: str) -> List[int]:
        """
        Tokenize a thought with proper handling of special tokens
        """
        # Ensure thought has proper XML structure
        if not thought_text.startswith(self.thought_start_token):
            thought_text = f"{self.thought_start_token}{thought_text}"
        if not thought_text.endswith(self.thought_end_token):
            thought_text = f"{thought_text}{self.thought_end_token}"

        # Tokenize with special token handling
        tokens = self.tokenizer.encode(
            thought_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        return tokens

    def batch_process_thoughts(
        self, thoughts: List[ThoughtStep], max_thoughts: int = 20
    ) -> torch.Tensor:
        """
        Process multiple thoughts into a batched tensor
        """
        formatted_thoughts = []

        for thought in thoughts[:max_thoughts]:
            # Format the thought
            formatted_text = self.format_thought_step(thought)
            # Tokenize with proper structure
            tokens = self.tokenize_thought(formatted_text)
            # Convert to tensor
            thought_tensor = torch.tensor(tokens, dtype=torch.long)
            formatted_thoughts.append(thought_tensor)

        # Pad sequence to same length
        padded_thoughts = pad_sequence(
            formatted_thoughts,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        return padded_thoughts


def parse_thinking_process(text: str) -> List[ThoughtStep]:
    """
    Enhanced parser with better regex patterns and error handling
    """
    try:
        # Split into main numbered sections with improved regex
        sections = re.split(r"(?=\d+\.\s+\*\*(?![^*]*\*\*))", text)
        sections = [s for s in sections if s.strip()]

        thought_steps = []

        for section in sections:
            # Extract step number with more robust pattern
            number_match = re.match(r"(\d+)\.\s+\*\*(?![^*]*\*\*)", section)
            if not number_match:
                continue

            step_number = int(number_match.group(1))

            # Extract title with improved pattern
            title_match = re.match(r"\d+\.\s+\*\*([^:*]+?)(?:\s*:)?\*\*\s*", section)
            if not title_match:
                continue

            title = title_match.group(1).strip()
            content = section[title_match.end() :].strip()

            # Process sub-points with better structure handling
            sub_points = process_sub_points(content)

            # Extract correction with improved pattern
            correction = extract_correction(section)

            thought_step = ThoughtStep(
                number=step_number,
                title=title,
                content=content,
                sub_points=sub_points,
                correction=correction,
            )
            thought_steps.append(thought_step)

        return sorted(thought_steps, key=lambda x: x.number)

    except Exception as e:
        print(f"Error parsing thinking process: {e}")
        return []


def process_sub_points(content: str) -> List[Dict[str, Any]]:
    """
    Process sub-points with improved structure handling
    """
    sub_points = []
    current_main_point = None

    lines = content.split("\n")
    for line in lines:
        line = line.strip()

        if line.startswith("* **"):
            # Handle main bullet points
            point_title = re.search(r"\* \*\*([^*]+)\*\*", line)
            if point_title:
                current_main_point = {
                    "title": point_title.group(1).strip(":"),
                    "sub_points": [],
                }
                sub_points.append(current_main_point)
        elif line.startswith("*") and current_main_point is not None:
            # Handle sub-bullet points
            sub_point = line.strip("* ").strip()
            if sub_point:
                current_main_point["sub_points"].append(sub_point)
        elif line:
            # Handle regular content
            if current_main_point is None:
                sub_points.append({"content": line})
            else:
                current_main_point.setdefault("content", "")
                current_main_point["content"] += f" {line}".strip()

    return sub_points


def extract_correction(section: str) -> Optional[str]:
    """
    Extract self-correction with improved pattern matching
    """
    if "Self-Correction" in section:
        correction_match = re.search(
            r"\*\*\(Self-Correction[^)]*\):\*\*\s*(.*?)(?=\n\n|\Z)", section, re.DOTALL
        )
        if correction_match:
            return correction_match.group(1).strip()
    return None


def prepare_training_data_including_thoughts(
    df: pd.DataFrame,
    block_size: int,
    tokenizer,
) -> List[Dict]:
    data_items = []
    max_thoughts = 5

    for idx, row in df.iterrows():
        try:
            context = row["input"]
            question = row["prompt"].split("**Title:**")[0].strip()

            # Get thoughts and format them
            formatted_thoughts = []
            thought = [t for t in json.loads(row["output"]) if t.get("thought", False)]

            if thought:
                thought_steps = parse_thinking_process(thought[0].get("text"))
                for step in thought_steps:
                    formatted_step = format_thought_step(step)
                    formatted_thoughts.append(formatted_step)

            # Get final answer
            final_answer = [
                t for t in json.loads(row["output"]) if not t.get("thought")
            ][0].get("text")

            # Print lengths before tokenization
            # print(f"Row {idx}:")
            # print(f"Question length: {len(question)}")
            # print(f"Context length: {len(context)}")
            # print(f"Answer length: {len(final_answer)}")
            # print(f"Number of thought steps: {len(formatted_thoughts)}")

            # Tokenize with explicit max lengths
            max_question_length = block_size // 16
            max_context_length = block_size // 2
            max_answer_length = block_size // 2
            max_thought_length = block_size // 16

            def validate_and_clip_tokens(tokens, max_val):
                return [min(t, max_val) for t in tokens]

            max_token_id = tokenizer.vocab_size - 1

            tokenized_question = validate_and_clip_tokens(
                tokenizer.encode(
                    question,
                    truncation=True,
                    padding="max_length",
                    max_length=max_question_length,
                ),
                max_token_id,
            )

            tokenized_context = validate_and_clip_tokens(
                tokenizer.encode(
                    context,
                    truncation=True,
                    padding="max_length",
                    max_length=max_context_length,
                ),
                max_token_id,
            )

            tokenized_complete_answer = validate_and_clip_tokens(
                tokenizer.encode(
                    final_answer,
                    truncation=True,
                    padding="max_length",
                    max_length=max_answer_length,
                ),
                max_token_id,
            )

            # Handle thought steps with explicit validation
            tokenized_thoughts = []
            for thought in formatted_thoughts[:max_thoughts]:
                thought_tokens = tokenizer.encode(
                    thought,
                    truncation=True,
                    padding="max_length",
                    max_length=max_thought_length,
                )
                tokenized_thoughts.append(
                    torch.tensor(thought_tokens, dtype=torch.long)
                )

            # Pad thoughts list with explicit zero tensors
            while len(tokenized_thoughts) < max_thoughts:
                pad_tensor = torch.zeros(max_thought_length, dtype=torch.long)
                pad_tensor.fill_(
                    tokenizer.pad_token_id
                )  # Use pad token ID instead of zeros
                tokenized_thoughts.append(pad_tensor)

            # Convert to tensors with size verification
            question_tensor = torch.tensor(tokenized_question, dtype=torch.long)
            context_tensor = torch.tensor(tokenized_context, dtype=torch.long)
            answer_tensor = torch.tensor(tokenized_complete_answer, dtype=torch.long)
            thoughts_tensor = torch.stack(tokenized_thoughts)

            # Print final tensor shapes
            # print(f"Final tensor shapes:")
            # print(f"Question: {question_tensor.shape}")
            # print(f"Context: {context_tensor.shape}")
            # print(f"Answer: {answer_tensor.shape}")
            # print(f"Thoughts: {thoughts_tensor.shape}")
            # print("-------------------")

            data_item = {
                "question": question_tensor,
                "context": context_tensor,
                "answer": answer_tensor,
                "intermediate_answers": thoughts_tensor,
                "raw_intermediate_answers": formatted_thoughts[:max_thoughts],
            }

            data_items.append(data_item)

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            raise  # Re-raise the exception to see the full stack trace
            continue

    return data_items


def read_text_from_directory(directory):
    files = glob(os.path.join(directory, "*.txt"))
    text = ""
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text += f.read() + "\n"
    return text


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


def generate_text(model, tokenizer, device, prompt, max_new_tokens=20):
    """Generate text using the model."""
    try:
        # Encode the prompt
        encoded = tokenizer.encode(prompt)  # This returns a list of ids
        input_ids = torch.tensor([encoded], dtype=torch.long).to(device)

        # Set model to eval mode
        model.eval()

        with torch.no_grad():
            # Generate
            output = model(input_ids)
            logits = output["logits"]

            # Sample from logits
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Decode
            generated = list(input_ids[0].cpu().numpy())
            generated.extend(next_token.cpu().numpy())

            return tokenizer.decode(generated)

    except Exception as e:
        print(f"Error in text generation: {str(e)}")
        return f"Error generating text: {str(e)}"
    finally:
        model.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        },
        path,
    )
    print(f"Checkpoint saved to {path} with epoch {epoch}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Load model, optimizer, and scheduler states from a checkpoint.
    """
    checkpoint = torch.load(path)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Optionally load optimizer and scheduler states
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from {path}, resuming at epoch {epoch}")
    return epoch  # Return the exact epoch saved


def fetch_training_data(task_descriptions, limit=50000, condition=None):
    conn = timescale_conn_pool.getconn()

    formatted_tasks = ", ".join("'" + str(task) + "'" for task in task_descriptions)

    query = f"""set statement_timeout = 300000;
            SELECT input, prompt, output, created_at, model_version, data
            FROM (
                SELECT DISTINCT input, prompt, output, created_at, model_version, data
                FROM vector.training_data
                WHERE task_description in ({formatted_tasks}) """

    if condition:
        query += f" AND {condition}"

    query += f"""
                ) sub
                ORDER BY
                    CASE
                        WHEN model_version ILIKE '%gpt-4%' THEN 1
                        WHEN model_version ILIKE '%sonnet%' THEN 2
                        WHEN model_version ILIKE '%gemini-flash-1.5%' THEN 3
                        WHEN model_version ILIKE '%qwen-2-72b-instruct%' THEN 4
                        WHEN model_version ILIKE '%llama-3-70b%' THEN 5
                        WHEN model_version ILIKE '%gpt-3.5-turbo-0125%' THEN 6
                        ELSE 7
                    END,
                    created_at DESC
                LIMIT {limit}"""

    # print(query)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    cur.close()

    news_df = pd.DataFrame(
        rows,
        columns=["input", "prompt", "output", "created_at", "model_version", "data"],
    )

    timescale_conn_pool.putconn(conn)

    return news_df


def load_news_from_ps(
    company_names=[],
    necessary_entities=[],
    msh_id=None,
    news_uuid=None,
    cut_off_date=None,
    limit=500,
    lookback_hours=None,
):
    conn = None
    try:
        conn = timescale_conn_pool.getconn()
        cur = conn.cursor()

        logging.info(
            f"Loading news with keywords {company_names} and limit {limit} and lookback_hours {lookback_hours}"
        )

        # Base SQL query
        sql_query = """
            SET statement_timeout = 2000000;
            SELECT raw.uuid AS news_uuid,
                   raw.title,
                   raw.article,
                   raw.link,
                   embed.summary_json,
                   embed.summary_model,
                   raw.published,
                   company_related.msh_id,
                   company_related.impact,
                   raw.publisher_name
            FROM public.msh_raw_news AS raw
            LEFT JOIN public.msh_news_embeddings AS embed ON raw.uuid = embed.news_uuid
            LEFT JOIN public.msh_company_related_news AS company_related ON raw.uuid = company_related.news_item_id
        """

        # Initialize WHERE conditions list
        where_conditions = []

        # Check for summary_json only if news_uuid is not provided
        if not news_uuid:
            where_conditions.append("embed.summary_json IS NOT NULL")

        # Apply cutoff date if specified
        if cut_off_date:
            where_conditions.append("raw.published >= %s")

        if lookback_hours:
            where_conditions.append("raw.published >= NOW() - INTERVAL '%s hours'")

        # Filter by specific news_uuid if provided
        if news_uuid:
            where_conditions.append("raw.uuid = %s")
        else:
            # Modifying the condition to accommodate multiple company names
            if necessary_entities:
                necessary_name_conditions = " AND ".join(
                    ["embed.summary_json ILIKE %s" for _ in necessary_entities]
                )
                where_conditions.append(f"({necessary_name_conditions})")
            if company_names:
                name_conditions = " OR ".join(
                    [
                        "embed.summary_json ILIKE %s OR raw.publisher_name ILIKE %s"
                        for _ in company_names
                    ]
                )
                where_conditions.append(f"({name_conditions})")

            # Include news associated with the provided msh_id
            if msh_id:
                where_conditions.append("company_related.msh_id = %s")

        # Join WHERE conditions with AND
        if where_conditions:
            sql_query += " WHERE " + " AND ".join(where_conditions)

        if not news_uuid:
            sql_query += " AND published is not null AND not published > now()"

        sql_query += " ORDER BY raw.published DESC LIMIT %s;"

        # Parameters preparation
        parameters = []
        if cut_off_date:
            parameters.append(cut_off_date)
        if lookback_hours:
            parameters.append(lookback_hours)
        if news_uuid:
            parameters.append(news_uuid)
        else:
            if necessary_entities:
                for name in necessary_entities:
                    parameters.extend([f"%{name}%"])
            if company_names:
                for name in company_names:
                    parameters.extend([f"%{name}%", f"%{name}%"])

            if msh_id:
                parameters.append(msh_id)
        parameters.append(limit)
        # Execute SQL query

        formatted_query = sql_query % tuple(map(psycopg2.extensions.adapt, parameters))
        # print("Executing SQL query:", formatted_query)

        cur.execute(sql_query, parameters)

        # Fetch all the rows
        rows = cur.fetchall()

        # Close the cursor and connection
        cur.close()

        # Create a DataFrame from the fetched results
        news_df = pd.DataFrame(
            rows,
            columns=[
                "news_uuid",
                "title",
                "article",
                "link",
                "summary_json",
                "summary_model",
                "published",
                "msh_id",
                "impact",
                "publisher",
            ],
        )

        news_df.drop_duplicates(subset=["news_uuid"], inplace=True)

        return news_df

    except (Exception, psycopg2.DatabaseError) as error:
        logging.error("DatabaseError", exc_info=True)
    finally:
        if conn is not None:
            timescale_conn_pool.putconn(conn)


def extract_facts_from_summary(summary_json):
    """
    Extracts facts from the summary_json field.
    Args:
        summary_json (str): JSON string containing facts.

    Returns:
        list of str: Extracted facts.
    """
    import json

    try:
        summary = json.loads(summary_json)
        return summary.get(
            "facts", []
        )  # Assuming 'facts' key contains the list of facts
    except (json.JSONDecodeError, TypeError):
        return []


# Preprocess the DataFrame
def prepare_question_context_answer_data(df, block_size, tokenizer):
    """
    Prepare data for training by extracting question, context, answer, and intermediate answers.

    Args:
        df (pd.DataFrame): DataFrame containing news data.
        block_size (int): Maximum sequence length.
        tokenizer: Tokenizer instance.

    Returns:
        list of dict: List of processed data items.
    """
    data = []
    for _, row in df.iterrows():
        # Extract fields
        question = row["title"]
        context = row["article"]
        answer = row["summary_model"]
        intermediate_answers = extract_facts_from_summary(row["summary_json"])

        # Tokenize question, context, and answer
        tokenized_question = tokenizer.encode(
            question, truncation=True, max_length=block_size // 2
        )
        tokenized_context = tokenizer.encode(
            context, truncation=True, max_length=block_size
        )
        tokenized_answer = tokenizer.encode(
            answer, truncation=True, max_length=block_size // 2
        )

        # Tokenize intermediate answers
        tokenized_intermediate_answers = [
            tokenizer.encode(fact, truncation=True, max_length=block_size // 4)
            for fact in intermediate_answers
        ]

        # Append processed item
        data.append(
            {
                "question": tokenized_question,
                "context": tokenized_context,
                "answer": tokenized_answer,
                "intermediate_answers": tokenized_intermediate_answers,
            }
        )

    return data


def test_gpu_operations():
    print("\nRunning GPU diagnostics...")

    try:
        # Test 1: Small tensor transfer
        print("\nTest 1: Basic tensor transfer")
        cpu_tensor = torch.arange(10, dtype=torch.long)
        gpu_tensor = cpu_tensor.to("cuda")
        print("Basic tensor transfer successful")

        # Test 2: Tensor with same shape as our data
        print("\nTest 2: Testing with data-like shapes")
        sample_question = torch.zeros((1, 1024), dtype=torch.long)
        sample_context = torch.zeros((1, 2048), dtype=torch.long)
        sample_answer = torch.zeros((1, 2048), dtype=torch.long)
        sample_intermediate = torch.zeros((1, 5, 512), dtype=torch.long)

        # Transfer each
        gpu_question = sample_question.to("cuda")
        print("Question transfer OK")
        gpu_context = sample_context.to("cuda")
        print("Context transfer OK")
        gpu_answer = sample_answer.to("cuda")
        print("Answer transfer OK")
        gpu_intermediate = sample_intermediate.to("cuda")
        print("Intermediate transfer OK")

        # Test 3: Concatenation
        print("\nTest 3: Testing concatenation")
        cat_result = torch.cat([gpu_question, gpu_context], dim=1)
        print(f"Concatenation successful, shape: {cat_result.shape}")

        # Test 4: Memory cleanup
        print("\nTest 4: Testing memory cleanup")
        del gpu_question, gpu_context, gpu_answer, gpu_intermediate, cat_result
        torch.cuda.empty_cache()
        print("Memory cleanup successful")

        print("\nAll GPU diagnostic tests passed!")
        return True

    except Exception as e:
        print(f"\nGPU diagnostic failed: {str(e)}")
        return False


def validate_dataset(data_items, tokenizer):
    shapes = []
    raw_text_lengths = []
    is_dataset_valid = True

    print("\nValidating dataset...")

    # First pass: Collect all shapes and check for basic tensor validity
    for idx, item in enumerate(data_items):
        try:
            # Basic tensor validation
            required_keys = ["question", "context", "answer", "intermediate_answers"]
            for key in required_keys:
                if key not in item:
                    print(f"Missing required key '{key}' at index {idx}")
                    is_dataset_valid = False
                    continue

                if not isinstance(item[key], torch.Tensor):
                    print(f"Item '{key}' at index {idx} is not a tensor")
                    is_dataset_valid = False
                    continue

            # Collect shapes
            shape_dict = {key: item[key].shape for key in required_keys}
            shapes.append(shape_dict)

            # Check for token indices within vocabulary bounds
            for key, tensor in item.items():
                if isinstance(tensor, torch.Tensor):
                    if torch.any(tensor < 0):
                        print(f"Found negative token indices in {key} at index {idx}")
                        is_dataset_valid = False
                    if torch.any(tensor >= tokenizer.vocab_size):
                        max_token = torch.max(tensor).item()
                        print(
                            f"Found out of bounds token in {key} at index {idx}: {max_token} >= {tokenizer.vocab_size}"
                        )
                        is_dataset_valid = False

        except Exception as e:
            print(f"Error validating item at index {idx}: {str(e)}")
            is_dataset_valid = False
            continue

    # Second pass: Check shape consistency
    if shapes:
        first_shape = shapes[0]
        print("\nExpected shapes:", first_shape)

        for idx, shape in enumerate(shapes):
            if shape != first_shape:
                print(f"\nInconsistent shapes at index {idx}:")
                for key in first_shape:
                    if shape[key] != first_shape[key]:
                        print(f"{key}: expected {first_shape[key]}, got {shape[key]}")
                is_dataset_valid = False

    # Third pass: Validate intermediate answers structure
    for idx, item in enumerate(data_items):
        try:
            # Check intermediate answers structure
            if "intermediate_answers" in item:
                int_answers = item["intermediate_answers"]

                # Print shape information for debugging
                print(f"\nIndex {idx} intermediate answers shape: {int_answers.shape}")

                # Validate basic shape requirements
                if len(int_answers.shape) != 2:
                    print(
                        f"Invalid intermediate_answers shape at index {idx}: expected 2D tensor, got {len(int_answers.shape)}D"
                    )
                    is_dataset_valid = False

                # Check for empty or invalid sequences
                if torch.all(int_answers == tokenizer.pad_token_id):
                    print(
                        f"Warning: All padding tokens in intermediate_answers at index {idx}"
                    )

                # Check for proper token structure
                thought_starts = (
                    (int_answers == tokenizer.convert_tokens_to_ids("<thought>"))
                    .sum()
                    .item()
                )
                thought_ends = (
                    (int_answers == tokenizer.convert_tokens_to_ids("</thought>"))
                    .sum()
                    .item()
                )

                if thought_starts != thought_ends:
                    print(
                        f"Mismatched thought tags at index {idx}: {thought_starts} starts, {thought_ends} ends"
                    )
                    is_dataset_valid = False

        except Exception as e:
            print(f"Error validating intermediate answers at index {idx}: {str(e)}")
            is_dataset_valid = False

    if is_dataset_valid:
        print("\nDataset validation successful!")
    else:
        print("\nDataset validation failed!")

    return is_dataset_valid


def print_epoch_summary(
    epoch, global_step, train_loss, val_loss, train_metrics, val_metrics
):
    print(f"\nEpoch {epoch + 1} (Step {global_step}) Summary:")
    print(f"Training - Total Loss: {train_loss:.4f}")

    print("Training Metrics:")
    for key, value in train_metrics.items():
        if key == "per_loop_latent_losses":
            print(f"  {key}:")
            # Check if value is a dict or list and handle accordingly
            if isinstance(value, dict):
                for loop_idx, loop_loss in sorted(value.items()):
                    print(f"    Loop {loop_idx}: {loop_loss:.4f}")
            elif isinstance(value, list):
                for i, loop_loss in enumerate(value):
                    print(f"    Loop {i}: {loop_loss:.4f}")
        elif isinstance(value, list):
            print(f"  {key}:")
            for i, v in enumerate(value):
                print(f"    Loop {i}: {v:.4f}")
        else:
            # Handle scalar metrics
            try:
                print(f"  {key}: {value:.4f}")
            except (ValueError, TypeError):
                print(f"  {key}: {value}")

    if val_loss is not None and val_metrics is not None:
        print(f"Validation - Total Loss: {val_loss:.4f}")
        print("Validation Metrics:")
        for key, value in val_metrics.items():
            if key == "per_loop_latent_losses":
                print(f"  {key}:")
                # Check if value is a dict or list and handle accordingly
                if isinstance(value, dict):
                    for loop_idx, loop_loss in sorted(value.items()):
                        print(f"    Loop {loop_idx}: {loop_loss:.4f}")
                elif isinstance(value, list):
                    for i, loop_loss in enumerate(value):
                        print(f"    Loop {i}: {loop_loss:.4f}")
            elif isinstance(value, list):
                print(f"  {key}:")
                for i, v in enumerate(value):
                    print(f"    Loop {i}: {v:.4f}")
            else:
                # Handle scalar metrics
                try:
                    print(f"  {key}: {value:.4f}")
                except (ValueError, TypeError):
                    print(f"  {key}: {value}")


def generate_sample_text(model, tokenizer, prompt, max_tokens, device, step):
    """Generate and print sample text from the model"""
    print(f"\n[Step {step}] Generating text for prompt: {prompt}")
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Assuming model has a generate method
        try:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                num_beams=4,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.9,
            )
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"[Step {step}] Generated text:\n{generated_text}\n")
        except Exception as e:
            print(f"Generation failed: {str(e)}")
    model.train()


def extract_json(input_string, return_single=True):
    """
    Extracts JSON strings from a string that contains JSON with additional commentary.

    Args:
    input_string (str): A string that contains JSON and additional commentary.
    return_single (bool): If True, returns a single JSON string when only one is found.
                         If False, always returns a list. Default is False.

    Returns:
    Union[str, list, str]: If return_single is True and exactly one JSON is found, returns that JSON string.
                          Otherwise returns a list of JSON strings. Returns an error message if no valid JSON is found.
    """

    def find_matching_bracket(s, start):
        stack = []
        brackets = {"{": "}", "[": "]"}
        for i, char in enumerate(s[start:], start):
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack:
                    return -1
                if char != brackets[stack.pop()]:
                    return -1
                if not stack:
                    return i
        return -1

    valid_jsons = []
    i = 0
    while i < len(input_string):
        # Find the start of a potential JSON structure
        while i < len(input_string) and input_string[i] not in "{[":
            i += 1
        if i >= len(input_string):
            break

        # Find the matching closing bracket/brace
        end = find_matching_bracket(input_string, i)
        if end == -1:
            i += 1
            continue

        # Extract the potential JSON string
        json_str = input_string[i : end + 1]
        try:
            # Validate the JSON by parsing it, but store the original string
            json.loads(json_str)  # Just for validation
            valid_jsons.append(json_str)
        except json.JSONDecodeError:
            pass
        i = end + 1

    if not valid_jsons:
        return "No valid JSON found in the input string."

    # Return single JSON string if requested and only one JSON was found
    if return_single and len(valid_jsons) == 1:
        return valid_jsons[0]

    return valid_jsons


def clean_model_answer_from_leading_json_str(s):
    if not s:
        return None
    # Replace the starting and ending patterns
    s = s.replace("JSON", "", 1)
    s = s.replace(
        "```json", "", 1
    )  # Remove starting ```json, only the first occurrence
    s = s.replace(
        "```", "", 1
    )  # Remove starting ``` if not json, only the first occurrence
    s = s[::-1].replace("```"[::-1], "", 1)[
        ::-1
    ]  # Remove the ending ```, only the last occurrence

    # Strip any leftover whitespace or newline characters
    return s.strip()
