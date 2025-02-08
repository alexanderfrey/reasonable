#!/usr/bin/env python3
import os
import json
import time
import random
from typing import List, Dict, Any
from dataclasses import dataclass
import argparse
import logging
from openai import OpenAI


@dataclass
class LanguageTask:
    grade_level: str
    skill_category: str
    task_type: str
    prompt: str
    thoughts: List[str]
    correct_response: str
    difficulty: str
    skills_tested: List[str]


class VLLMAPI:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen-7B-Chat",
        openai_api_base: str = "http://localhost:8000/v1",
        max_tokens: int = 4096,
        show_logs: bool = True,
    ):
        self.model_id = model_id
        self.openai_api_base = openai_api_base
        self.max_tokens = max_tokens
        self.show_logs = show_logs

        try:
            self._test_connection()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to VLLM API: {e}")

    def _test_connection(self):
        try:
            client = OpenAI(api_key="EMPTY", base_url=self.openai_api_base)
            # Simple test completion
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to VLLM API: {e}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        for attempt in range(max_retries):
            try:
                client = OpenAI(api_key="EMPTY", base_url=self.openai_api_base)
                completion = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                )
                return completion.choices[0].message.content

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ConnectionError(
                        f"Failed to generate after {max_retries} attempts: {e}"
                    )
                time.sleep(retry_delay)
                continue

        raise RuntimeError("Failed to generate response")


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
                    "rhyming_words",
                    "letter_sounds",
                    "uppercase_lowercase",
                    "word_families",
                    "oral_vocabulary",
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
                    "phonemic_awareness",
                    "spelling_patterns",
                    "story_elements",
                    "descriptive_language",
                    "verb_tenses",
                ],
            },
            "second_grade": {
                "age_range": "7-8",
                "skill_categories": [
                    "compound_words",
                    "prefixes_suffixes",
                    "grammar_basics",
                    "reading_fluency",
                    "main_idea",
                    "context_clues",
                    "paragraph_writing",
                    "synonyms_antonyms",
                    "contractions",
                    "comprehension_strategies",
                ],
            },
            "third_grade": {
                "age_range": "8-9",
                "skill_categories": [
                    "reading_strategies",
                    "writing_process",
                    "vocabulary_context",
                    "text_features",
                    "parts_of_speech",
                    "figurative_language",
                    "summarizing",
                    "editing_revising",
                    "research_skills",
                    "advanced_punctuation",
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
        }

    def generate_prompt_template(
        self, grade_level: str, skill_category: str, task_type: str
    ) -> str:
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

        Format the response as a JSON object with:
        {{
            "prompt": "the actual question or task",
            "thoughts": [
                "step-by-step thinking process",
                "analysis of the problem",
                "consideration of relevant rules or patterns"
            ],
            "correct_response": "complete answer with explanation",
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
        temperature: float = 0.7,
    ) -> List[LanguageTask]:
        tasks = []
        for _ in range(num_tasks):
            grade = specific_grade or random.choice(list(self.grade_levels.keys()))
            category = specific_category or random.choice(
                self.grade_levels[grade]["skill_categories"]
            )
            task_type = random.choice(list(self.task_types.keys()))

            prompt = self.generate_prompt_template(grade, category, task_type)

            try:
                response = llm_api.generate(prompt, temperature=temperature)
                # Clean the response - remove any leading/trailing whitespace and non-JSON text
                response = response.strip()
                # Try to find JSON content between curly braces if there's other text
                if not response.startswith("{"):
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    if start != -1 and end != 0:
                        response = response[start:end]

                try:
                    task_data = json.loads(response)
                    # Validate required fields
                    required_fields = [
                        "prompt",
                        "thoughts",
                        "correct_response",
                        "difficulty",
                        "skills_tested",
                    ]
                    if all(field in task_data for field in required_fields):
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
                    else:
                        logging.warning(
                            f"Missing required fields in response: {response[:100]}..."
                        )
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse LLM response as JSON: {str(e)}")
                    logging.error(f"Response received: {response[:100]}...")
            except Exception as e:
                logging.error(f"Error generating task: {str(e)}")
                continue

        return tasks

    def create_output_filename(
        self, base_filename: str, grade_levels: List[str]
    ) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        grades_str = (
            "_".join(grade_levels)
            if len(grade_levels) <= 3
            else f"{len(grade_levels)}grades"
        )
        filename = f"language_tasks_{grades_str}_{timestamp}.jsonl"
        output_dir = "training_data"
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        return output_path

    def save_task(self, task: LanguageTask, file_handle) -> None:
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
        file_handle.write(json.dumps(task_dict, ensure_ascii=False) + "\n")
        file_handle.flush()


def main(
    output_file: str,
    model_id: str = "Qwen/Qwen-7B-Chat",
    openai_api_base: str = "http://localhost:8000/v1",
    num_tasks_per_grade: int = 100,
    grades: List[str] = None,
    categories: List[str] = None,
    temperature: float = 0.7,
    verbose: bool = False,
):
    # Configure logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    framework = EducationalLevelFramework()
    llm_api = VLLMAPI(
        model_id=model_id, openai_api_base=openai_api_base, show_logs=verbose
    )

    if grades is None:
        grades = list(framework.grade_levels.keys())

    total_tasks = len(grades) * num_tasks_per_grade
    tasks_generated = 0

    output_filename = framework.create_output_filename(output_file, grades)
    logging.info(f"Writing output to: {output_filename}")

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            for grade in grades:
                if verbose:
                    logging.info(f"\nGenerating tasks for {grade}...")

                grade_categories = (
                    categories or framework.grade_levels[grade]["skill_categories"]
                )

                base_tasks_per_category = max(
                    1, num_tasks_per_grade // len(grade_categories)
                )
                remaining_tasks = num_tasks_per_grade - (
                    base_tasks_per_category * len(grade_categories)
                )

                for i, category in enumerate(grade_categories):
                    current_category_tasks = (
                        base_tasks_per_category + 1
                        if i < remaining_tasks
                        else base_tasks_per_category
                    )

                    if verbose:
                        logging.info(
                            f"Generating {current_category_tasks} tasks for {category}..."
                        )

                    for _ in range(current_category_tasks):
                        try:
                            tasks = framework.generate_tasks_with_llm(
                                llm_api,
                                num_tasks=1,
                                specific_grade=grade,
                                specific_category=category,
                                temperature=temperature,
                            )

                            if tasks:
                                framework.save_task(tasks[0], f)
                                tasks_generated += 1

                                if verbose:
                                    logging.info(
                                        f"Generated {tasks_generated}/{total_tasks} tasks..."
                                    )

                        except Exception as e:
                            logging.error(f"Error generating task: {str(e)}")
                            continue

    except KeyboardInterrupt:
        logging.warning(
            "\nGeneration interrupted by user. Tasks saved up to this point."
        )

    finally:
        if verbose:
            logging.info(f"\nGeneration complete:")
            logging.info(f"- Successfully generated {tasks_generated} tasks")
            logging.info(f"- Results saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate educational language tasks using VLLM"
    )
    parser.add_argument("output_file", help="Path to save generated tasks")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen-7B-Chat",
        help="VLLM model ID to use (default: Qwen/Qwen-7B-Chat)",
    )
    parser.add_argument(
        "--api-base", default="http://localhost:8000/v1", help="VLLM API base URL"
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
        help="Sampling temperature (0.0 to 1.0)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information"
    )

    args = parser.parse_args()

    main(
        output_file=args.output_file,
        model_id=args.model_id,
        openai_api_base=args.api_base,
        num_tasks_per_grade=args.tasks_per_grade,
        grades=args.grades,
        categories=args.categories,
        temperature=args.temperature,
        verbose=args.verbose,
    )
