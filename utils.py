import pandas as pd
from pandas import DataFrame
import json, re, os
from typing import List, Dict, Any
from dataclasses import dataclass
from typing import Optional
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from torch.utils.data import DataLoader, TensorDataset
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
timescale_conn_pool = ThreadedConnectionPool(
    1,  # minconn
    40,  # maxconn
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    port=int(os.getenv("DB_PORT", 5432)),
    password=os.getenv("DB_PASSWORD"),
)


def create_dataloader(X, Y, batch_size, num_workers=4, shuffle=True):
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    return dataloader


@dataclass
class ThoughtStep:
    number: int
    title: str
    content: str
    sub_points: List[Dict[str, Any]]
    correction: Optional[str] = None


def parse_thinking_process(text: str) -> List[ThoughtStep]:
    """
    Parses a structured thinking process text into component thoughts and sub-points.

    Args:
        text (str): The thinking process text to parse

    Returns:
        List[ThoughtStep]: List of parsed thought steps with their content and sub-points
    """
    # Split into main numbered sections, keeping the number
    sections = re.split(r"(?=\d+\.\s+\*\*)", text)

    # Remove any leading empty sections
    sections = [s for s in sections if s.strip()]

    thought_steps = []

    for section in sections:
        # Extract step number
        number_match = re.match(r"(\d+)\.\s+\*\*", section)
        if not number_match:
            continue

        step_number = int(number_match.group(1))

        # Extract title
        title_match = re.match(r"\d+\.\s+\*\*([^:*]+):\*\*\s*", section)
        if not title_match:
            continue

        title = title_match.group(1).strip()
        content = section[title_match.end() :].strip()

        # Extract sub-points
        sub_points = []
        current_main_point = None

        # Split content into lines, preserving empty lines
        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Handle main bullet points starting with *
            if line.startswith("* **"):
                point_title = re.search(r"\* \*\*([^*]+)\*\*", line)
                if point_title:
                    current_main_point = {
                        "title": point_title.group(1).strip(":"),
                        "sub_points": [],
                    }
                    sub_points.append(current_main_point)

            # Handle sub-bullet points (indented with spaces + *)
            elif line.strip().startswith("*") and current_main_point is not None:
                sub_point = line.strip("* ").strip()
                if sub_point:
                    current_main_point["sub_points"].append(sub_point)

            # Handle regular content
            elif line and not line.startswith("*"):
                if current_main_point is None:
                    sub_points.append({"content": line})
                else:
                    if "content" not in current_main_point:
                        current_main_point["content"] = line
                    else:
                        current_main_point["content"] += " " + line

            i += 1

        # Check for self-correction section
        correction = None
        if "Self-Correction" in section:
            correction_match = re.search(
                r"\*\*\(Self-Correction[^)]*\):\*\*\s*(.*?)(?=\n\n|\Z)",
                section,
                re.DOTALL,
            )
            if correction_match:
                correction = correction_match.group(1).strip()

        thought_step = ThoughtStep(
            number=step_number,
            title=title,
            content=content,
            sub_points=sub_points,
            correction=correction,
        )
        thought_steps.append(thought_step)

    return sorted(thought_steps, key=lambda x: x.number)


def format_thought_step(step: ThoughtStep) -> str:
    """
    Formats a thought step into a readable string representation.

    Args:
        step (ThoughtStep): The thought step to format

    Returns:
        str: Formatted string representation of the thought step
    """
    output = "<thought>"
    output += f"<title>{step.title}</title>\n"

    if step.sub_points:
        output += "<value>"

    for point in step.sub_points:
        if "title" in point:
            output += f"\n• {point['title']}\n"
            if "content" in point:
                output += f"  {point['content']}\n"
            for sub_point in point.get("sub_points", []):
                output += f"    - {sub_point}\n"
        elif "content" in point:
            output += f"{point['content']}\n"

    if step.correction:
        output += f"\nSelf-Correction:\n{step.correction}\n"

    if step.sub_points:
        output += "</value>"

    output += "</thought>"

    return output


def prepare_training_data_including_thoughts(
    df: pd.DataFrame,
    block_size: int,
    tokenizer,
) -> List[Dict]:
    """
    Prepare data for training with consistent padding and sequence lengths.
    """
    data_items = []
    max_thoughts = 5  # Set a maximum number of thought steps to handle
    
    for idx, row in df.iterrows():
        try:
            # Get context and question from DataFrame fields
            context = row["input"]
            question = row["prompt"]

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

            # Tokenize and pad inputs
            tokenized_question = torch.tensor(
                tokenizer.encode(
                    question,
                    truncation=True,
                    padding='max_length',
                    max_length=block_size // 2
                ),
                dtype=torch.long
            )
            
            tokenized_context = torch.tensor(
                tokenizer.encode(
                    context,
                    truncation=True,
                    padding='max_length',
                    max_length=block_size
                ),
                dtype=torch.long
            )
            
            tokenized_complete_answer = torch.tensor(
                tokenizer.encode(
                    final_answer,
                    truncation=True,
                    padding='max_length',
                    max_length=block_size
                ),
                dtype=torch.long
            )

            # Tokenize and pad thought steps with consistent size
            tokenized_thoughts = []
            for thought in formatted_thoughts[:max_thoughts]:  # Limit number of thoughts
                thought_tokens = torch.tensor(
                    tokenizer.encode(
                        thought,
                        truncation=True,
                        padding='max_length',
                        max_length=block_size // 4
                    ),
                    dtype=torch.long
                )
                tokenized_thoughts.append(thought_tokens)
            
            # Pad thoughts list if fewer than max_thoughts
            while len(tokenized_thoughts) < max_thoughts:
                pad_tokens = torch.zeros(block_size // 4, dtype=torch.long)
                tokenized_thoughts.append(pad_tokens)
            
            # Stack thoughts into a single tensor
            tokenized_thoughts = torch.stack(tokenized_thoughts)

            # Create data item
            data_item = {
                "question": tokenized_question,
                "context": tokenized_context,
                "answer": tokenized_complete_answer,  # Changed name to match training loop
                "intermediate_answers": tokenized_thoughts,  # Changed name to match training loop
            }

            data_items.append(data_item)

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
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


def generate_text(
    model, tokenizer, device, prompt="The quick brown fox", max_new_tokens=30
):
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

    print(query)
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
        print("Executing SQL query:", formatted_query)

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
