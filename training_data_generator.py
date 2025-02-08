#!/usr/bin/env python3
import os
import json
import time
import random
import requests
from typing import List, Dict, Any
from dataclasses import dataclass
import argparse


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


class OllamaAPI:
    def __init__(
        self, model_name: str = "llama2", base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"

        try:
            self._test_connection()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama API: {e}")

    def _test_connection(self):
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
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "format": "json",
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.generate_url, json=data)
                response.raise_for_status()

                try:
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
            response = llm_api.generate(prompt, temperature=temperature)

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
        """Save generated tasks to a JSONL file with appropriate naming"""
        grade_levels = sorted(set(task.grade_level for task in tasks))
        num_tasks = len(tasks)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        grades_str = (
            "_".join(grade_levels)
            if len(grade_levels) <= 3
            else f"{len(grade_levels)}grades"
        )
        filename = f"language_tasks_{grades_str}_{num_tasks}tasks_{timestamp}.jsonl"

        if os.path.dirname(base_filename):
            filename = os.path.join(os.path.dirname(base_filename), filename)

        os.makedirs(
            os.path.dirname(filename) if os.path.dirname(filename) else ".",
            exist_ok=True,
        )

        with open(filename, "w", encoding="utf-8") as f:
            for task in tasks:
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
                f.write(json.dumps(task_dict, ensure_ascii=False) + "\n")

        return filename


def main(
    output_file: str,
    model_name: str = "llama2",
    num_tasks_per_grade: int = 100,
    grades: List[str] = None,
    categories: List[str] = None,
    temperature: float = 0.7,
    verbose: bool = False,
):

    framework = EducationalLevelFramework()
    llm_api = OllamaAPI(model_name=model_name)

    if grades is None:
        grades = list(framework.grade_levels.keys())

    total_tasks = len(grades) * num_tasks_per_grade
    generated_tasks = []

    try:
        for grade in grades:
            if verbose:
                print(f"\nGenerating tasks for {grade}...")

            grade_categories = (
                categories or framework.grade_levels[grade]["skill_categories"]
            )
            tasks_per_category = num_tasks_per_grade // len(grade_categories)

            for category in grade_categories:
                if verbose:
                    print(f"Generating {tasks_per_category} tasks for {category}...")

                new_tasks = framework.generate_tasks_with_llm(
                    llm_api,
                    num_tasks=tasks_per_category,
                    specific_grade=grade,
                    specific_category=category,
                    temperature=temperature,
                )

                generated_tasks.extend(new_tasks)

                if verbose:
                    print(f"Generated {len(generated_tasks)}/{total_tasks} tasks...")

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user. Saving progress...")

    finally:
        if generated_tasks:
            final_file = framework.save_tasks_to_jsonl(generated_tasks, output_file)
            if verbose:
                print(f"\nGeneration complete:")
                print(f"- Successfully generated {len(generated_tasks)} tasks")
                print(f"- Results saved to {final_file}")


if __name__ == "__main__":
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
        help="Sampling temperature (0.0 to 1.0)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information"
    )

    args = parser.parse_args()

    main(
        output_file=args.output_file,
        model_name=args.model,
        num_tasks_per_grade=args.tasks_per_grade,
        grades=args.grades,
        categories=args.categories,
        temperature=args.temperature,
        verbose=args.verbose,
    )
