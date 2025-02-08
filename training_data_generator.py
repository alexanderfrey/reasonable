import requests
from typing import List, Dict, Any
import json
from dataclasses import dataclass
import random
import time


class OllamaAPI:
    def __init__(
        self, model_name: str = "llama2", base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama API client

        Args:
            model_name: Name of the model to use (e.g., "llama2", "mistral")
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"

        # Verify connection
        try:
            self._test_connection()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama API: {e}")

    def _test_connection(self):
        """Test connection to Ollama API"""
        response = requests.get(f"{self.base_url}/api/tags")
        if response.status_code != 200:
            raise ConnectionError(
                f"Failed to connect to Ollama API. Status: {response.status_code}"
            )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """
        Generate text using Ollama API with retry logic

        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            str: Generated text response
        """
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "format": "json",  # Request JSON format
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.generate_url, json=data)
                response.raise_for_status()

                # Parse response
                try:
                    # Ollama streams responses, we need to take the last complete response
                    lines = response.text.strip().split("\n")
                    last_response = json.loads(lines[-1])
                    return last_response["response"]
                except (json.JSONDecodeError, KeyError) as e:
                    if attempt == max_retries - 1:
                        raise ValueError(f"Failed to parse Ollama response: {e}")

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise ConnectionError(
                        f"Failed to generate after {max_retries} attempts: {e}"
                    )

                time.sleep(retry_delay)
                continue

        raise RuntimeError("Failed to generate response")


@dataclass
class LanguageTask:
    grade_level: str
    skill_category: str
    task_type: str
    prompt: str
    thoughts: List[str]  # Intermediate thinking steps
    correct_response: str
    difficulty: str
    skills_tested: List[str]


class EducationalLevelFramework:
    def __init__(self):
        self.grade_levels = {
            "kindergarten": {
                "age_range": "5-6",
                "skill_categories": [
                    "letter_recognition",
                    "phonics",
                    "basic_writing",
                    "sight_words",
                    "listening_comprehension",
                ],
            },
            "first_grade": {
                "age_range": "6-7",
                "skill_categories": [
                    "word_building",
                    "sentence_structure",
                    "basic_punctuation",
                    "vocabulary",
                    "reading_comprehension",
                ],
            },
            "second_grade": {
                "age_range": "7-8",
                "skill_categories": [
                    "grammar_basics",
                    "sentence_types",
                    "compound_words",
                    "story_elements",
                    "writing_process",
                ],
            },
            "third_grade": {
                "age_range": "8-9",
                "skill_categories": [
                    "parts_of_speech",
                    "verb_tenses",
                    "paragraph_writing",
                    "reading_strategies",
                    "vocabulary_expansion",
                ],
            },
            "fourth_grade": {
                "age_range": "9-10",
                "skill_categories": [
                    "advanced_grammar",
                    "writing_styles",
                    "complex_sentences",
                    "literary_devices",
                    "research_skills",
                ],
            },
        }

        self.task_types = {
            "multiple_choice": {
                "format": "question with 4 options",
                "response_type": "single correct answer",
            },
            "fill_in_blank": {
                "format": "sentence with ___ for missing word",
                "response_type": "single word or phrase",
            },
            "short_answer": {
                "format": "open-ended question",
                "response_type": "2-3 sentence response",
            },
            "matching": {
                "format": "two columns of related items",
                "response_type": "paired matches",
            },
            "correction": {
                "format": "text with errors",
                "response_type": "corrected version",
            },
        }

    def generate_prompt_template(
        self, grade_level: str, skill_category: str, task_type: str
    ) -> str:
        """Generate an LLM prompt template for a specific educational context"""
        template = f"""
        Create a {task_type} question for {grade_level} students that tests {skill_category}.
        
        Question Requirements:
        1. Be age-appropriate for {self.grade_levels[grade_level]['age_range']} year olds
        2. Focus specifically on {skill_category}
        3. Be engaging and relate to common childhood experiences
        4. Use clear, simple language appropriate for the grade level
        5. Connect to everyday situations students might encounter
        
        Thinking Process Requirements:
        1. Include 3-5 step-by-step thoughts that show the reasoning process
        2. Make thoughts explicit and age-appropriate
        3. Model good problem-solving strategies
        4. Show connections to prior knowledge
        5. Demonstrate metacognitive strategies appropriate for the grade level
        
        Answer Requirements:
        1. Provide a complete answer that goes beyond a single word
        2. Include a clear explanation of why the answer is correct
        3. Give an example or make a connection to something familiar
        4. Use age-appropriate language for {grade_level} level
        5. Reinforce the key concept being taught
        
        Language Guidelines by Grade:
        - Kindergarten: Very simple sentences, concrete examples, familiar vocabulary
        - First Grade: Short, clear sentences, common sight words, basic explanations
        - Second Grade: Compound sentences okay, growing vocabulary, clear connections
        - Third Grade: More complex sentences, academic vocabulary introduction
        - Fourth Grade: Multiple connected ideas, expanded vocabulary, deeper explanations
        
        Format the response as a JSON object with:
        {{
            "prompt": "the actual question or task",
            "correct_response": "the answer or solution",
            "difficulty": "easy/medium/hard",
            "skills_tested": ["specific skills being tested"]
        }}
        """
        return template

    def generate_tasks_with_llm(
        self,
        llm_api,
        num_tasks: int = 10,
        specific_grade: str = None,
        specific_category: str = None,
    ) -> List[LanguageTask]:
        """Generate tasks using the provided LLM API"""
        tasks = []
        for _ in range(num_tasks):
            # Select random grade and category if not specified
            grade = specific_grade or random.choice(list(self.grade_levels.keys()))
            category = specific_category or random.choice(
                self.grade_levels[grade]["skill_categories"]
            )
            task_type = random.choice(list(self.task_types.keys()))

            # Generate prompt for LLM
            prompt = self.generate_prompt_template(grade, category, task_type)

            # Call LLM API (implementation depends on specific API being used)
            response = llm_api.generate(prompt)

            # Parse response and create LanguageTask object
            try:
                task_data = json.loads(response)
                task = LanguageTask(
                    grade_level=grade,
                    skill_category=category,
                    task_type=task_type,
                    prompt=task_data["prompt"],
                    thoughts=task_data["thoughts"],
                    correct_response=task_data["correct_response"],
                    difficulty=task_data["difficulty"],
                    skills_tested=task_data["skills_tested"],
                )
                tasks.append(task)
            except json.JSONDecodeError:
                print(f"Failed to parse LLM response: {response}")
                continue

        return tasks

    def save_tasks_to_jsonl(self, tasks: List[LanguageTask], base_filename: str):
        """
        Save generated tasks to a JSONL file with appropriate naming

        Args:
            tasks: List of generated tasks
            base_filename: Base name for the output file
        """
        # Extract metadata for filename
        grade_levels = sorted(set(task.grade_level for task in tasks))
        num_tasks = len(tasks)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create descriptive filename
        grades_str = (
            "_".join(grade_levels)
            if len(grade_levels) <= 3
            else f"{len(grade_levels)}grades"
        )
        filename = f"language_tasks_{grades_str}_{num_tasks}tasks_{timestamp}.jsonl"

        # If base_filename includes a directory, preserve it
        if os.path.dirname(base_filename):
            filename = os.path.join(os.path.dirname(base_filename), filename)

        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(filename) if os.path.dirname(filename) else ".",
            exist_ok=True,
        )

        # Save tasks in JSONL format
        with open(filename, "w", encoding="utf-8") as f:
            for task in tasks:
                # Convert task to dictionary
                task_dict = {
                    "grade_level": task.grade_level,
                    "skill_category": task.skill_category,
                    "task_type": task.task_type,
                    "prompt": task.prompt,
                    "thoughts": task.thoughts,
                    "correct_response": task.correct_response,
                    "difficulty": task.difficulty,
                    "skills_tested": task.skills_tested,
                }
                # Write as single JSON line
                f.write(json.dumps(task_dict, ensure_ascii=False) + "\n")

        return filename

    def save_checkpoint(self, tasks: List[LanguageTask], base_filename: str):
        """Save checkpoint with appropriate naming"""
        checkpoint_file = f"{base_filename}.checkpoint.jsonl"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(vars(task), ensure_ascii=False) + "\n")

    def load_tasks_from_jsonl(self, filename: str) -> List[LanguageTask]:
        """Load tasks from a JSONL file"""
        tasks = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                task_dict = json.loads(line)
                tasks.append(LanguageTask(**task_dict))
        return tasks


def main(
    output_file: str,
    llm_api_key: str,
    num_tasks_per_grade: int = 100,
    grades: List[str] = None,
    categories: List[str] = None,
    difficulties: List[str] = None,
    task_types: List[str] = None,
    balance_categories: bool = True,
    validate_outputs: bool = True,
    max_retries: int = 3,
    batch_size: int = 10,
    save_frequency: int = 50,
    verbose: bool = False,
) -> None:
    """
    Generate educational language tasks using an LLM API.

    Args:
        output_file: Path to save the generated tasks
        llm_api_key: API key for the LLM service
        num_tasks_per_grade: Number of tasks to generate per grade level
        grades: Specific grades to generate tasks for. If None, uses all grades
        categories: Specific skill categories to focus on. If None, uses all categories
        difficulties: Specific difficulties to generate. If None, generates a mix
        task_types: Specific task types to generate. If None, uses all types
        balance_categories: Whether to ensure equal distribution of skill categories
        validate_outputs: Whether to validate LLM outputs against grade-level criteria
        max_retries: Maximum number of retries for failed generations
        batch_size: Number of tasks to generate in parallel
        save_frequency: How often to save progress to disk
        verbose: Whether to print detailed progress information
    """
    # Initialize framework
    framework = EducationalLevelFramework()

    # Initialize LLM API client
    llm_api = initialize_llm_client(llm_api_key)

    # Set up grade levels to generate
    if grades is None:
        grades = list(framework.grade_levels.keys())

    # Track generation progress
    total_tasks = len(grades) * num_tasks_per_grade
    generated_tasks = []
    failed_generations = []

    try:
        for grade in grades:
            if verbose:
                print(f"\nGenerating tasks for {grade}...")

            # Calculate tasks per category if balancing
            grade_categories = (
                categories or framework.grade_levels[grade]["skill_categories"]
            )

            if balance_categories:
                tasks_per_category = num_tasks_per_grade // len(grade_categories)

            # Generate tasks for each category
            for category in grade_categories:
                remaining_tasks = (
                    tasks_per_category if balance_categories else num_tasks_per_grade
                )

                while remaining_tasks > 0:
                    # Generate in batches
                    current_batch = min(batch_size, remaining_tasks)

                    # Generate batch of tasks
                    new_tasks = []
                    retries = 0

                    while len(new_tasks) < current_batch and retries < max_retries:
                        try:
                            batch_tasks = framework.generate_tasks_with_llm(
                                llm_api,
                                num_tasks=current_batch - len(new_tasks),
                                specific_grade=grade,
                                specific_category=category,
                            )

                            # Validate tasks if requested
                            if validate_outputs:
                                batch_tasks = [
                                    task
                                    for task in batch_tasks
                                    if validate_task(task, grade)
                                ]

                            new_tasks.extend(batch_tasks)

                        except Exception as e:
                            retries += 1
                            if verbose:
                                print(f"Retry {retries}/{max_retries} due to: {e}")

                    # Track failed generations
                    if len(new_tasks) < current_batch:
                        failed_generations.append(
                            {
                                "grade": grade,
                                "category": category,
                                "attempted": current_batch,
                                "succeeded": len(new_tasks),
                            }
                        )

                    # Add successful generations to results
                    generated_tasks.extend(new_tasks)
                    remaining_tasks -= len(new_tasks)

                    # Save progress periodically
                    if len(generated_tasks) % save_frequency == 0:
                        checkpoint_file = framework.save_checkpoint(
                            generated_tasks, output_file
                        )

                    if verbose:
                        print(
                            f"Generated {len(generated_tasks)}/{total_tasks} tasks..."
                        )

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user. Saving progress...")

    finally:
        # Save all successfully generated tasks
        final_file = framework.save_tasks_to_jsonl(generated_tasks, output_file)

        # Save failed generation log if any failures occurred
        if failed_generations:
            failed_log_file = f"{os.path.splitext(final_file)[0]}_failed_log.json"
            with open(failed_log_file, "w") as f:
                json.dump(failed_generations, f, indent=2)

        if verbose:
            print(f"\nGeneration complete:")
            print(f"- Successfully generated {len(generated_tasks)} tasks")
            print(f"- Failed generations: {len(failed_generations)}")
            print(f"- Results saved to {final_file}")
            if failed_generations:
                print(f"- Failed log saved to {failed_log_file}")


def validate_task(task: LanguageTask, grade: str) -> bool:
    """
    Validate that a generated task meets grade-level requirements.

    Args:
        task: The task to validate
        grade: The grade level to validate against

    Returns:
        bool: Whether the task meets all validation criteria
    """
    # Implement validation logic here
    # Example checks:
    # - Appropriate vocabulary level
    # - Correct grammar and spelling
    # - Grade-appropriate complexity
    # - Complete and well-formed responses
    # - Proper thought process steps
    return True  # Placeholder


def initialize_llm_client(model_name: str = "llama2"):
    """
    Initialize the Ollama API client with the specified model.

    Args:
        model_name: Name of the Ollama model to use
    Returns:
        OllamaAPI: Initialized Ollama client
    """
    return OllamaAPI(model_name=model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate educational language tasks using Ollama LLM"
    )
    parser.add_argument("output_file", help="Path to save generated tasks")
    parser.add_argument(
        "--model", default="llama2", help="Ollama model name to use (default: llama2)"
    )
    parser.add_argument(
        "--tasks-per-grade",
        type=int,
        default=100,
        help="Number of tasks to generate per grade",
    )
    parser.add_argument(
        "--grades", nargs="*", help="Specific grades to generate tasks for"
    )
    parser.add_argument(
        "--categories", nargs="*", help="Specific skill categories to focus on"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (0.0 to 1.0)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Ensure equal distribution of skill categories",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate outputs against grade-level criteria",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information"
    )

    args = parser.parse_args()

    main(
        output_file=args.output_file,
        llm_api_key=args.llm_api_key,
        num_tasks_per_grade=args.tasks_per_grade,
        grades=args.grades,
        categories=args.categories,
        balance_categories=args.balance,
        validate_outputs=args.validate,
        verbose=args.verbose,
    )
