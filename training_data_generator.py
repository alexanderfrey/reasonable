#!/usr/bin/env python3
import os
import json
import time
import random
from typing import List, Dict, Any
from dataclasses import dataclass
import argparse
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
from datetime import datetime, timedelta
from openai import OpenAI
import pandas as pd

from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from google import genai


def generate_response_with_google(input_message: str) -> str:
    """
    Sends a message to the Google Generative AI model and returns the response.

    Args:
        input_message (str): The input text to send to the model.

    Returns:
        str: The text response from the model.
    """
    # key = "AIzaSyB1Xl1sADBc6CgqUNU7aCQ1nstWFlkvcKE"
    key = "AIzaSyCvL-__lNp4bTHRMWvFuwrV435-ot-2jM0"
    client = genai.Client(api_key=key, http_options={"api_version": "v1alpha"})

    response = client.models.generate_content(
        model="gemini-2.0-pro-exp-02-05",
        contents=input_message,
        # config={"thinking_config": {"include_thoughts": True}},
    )

    # Return the text response
    return response.text


def create_connection_pool():
    """Create a database connection pool using environment variables."""
    return ThreadedConnectionPool(
        1,  # minconn
        40,  # maxconn
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        port=os.getenv("DB_PORT"),
        password=os.getenv("DB_PASSWORD"),
    )


def load_recent_news(
    conn_pool, start_date: str, end_date: str, limit: int = 10000
) -> pd.DataFrame:
    """
    Load the most recent news articles.

    Args:
        conn_pool: Database connection pool
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        limit: Maximum number of articles to fetch

    Returns:
        DataFrame containing the news articles
    """
    conn = None
    try:
        conn = conn_pool.getconn()
        cur = conn.cursor()

        query = f"""
        SELECT 
            raw.uuid as news_uuid,
            raw.title, 
            article, 
            raw.link,
            published
        FROM 
            msh_raw_news raw
        WHERE 
            raw.published >= '{start_date}' 
            AND raw.published <= '{end_date}'
        ORDER BY 
            raw.published DESC
        LIMIT {limit};
        """

        cur.execute(query)
        rows = cur.fetchall()

        return pd.DataFrame(
            rows, columns=["news_uuid", "title", "article", "link", "published"]
        )

    except psycopg2.Error as e:
        logging.error(f"Database error in news load: {e}")
        raise
    finally:
        if conn is not None:
            conn_pool.putconn(conn)


@dataclass
class LanguageTask:
    grade_level: str
    skill_category: str
    task_type: str
    prompt: str
    context: str
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
            "children": {  # NEW GRADE LEVEL
                "age_range": "25-35",
                "skill_categories": [
                    # Foundational Language Understanding
                    "reading_comprehension",
                    # "following_directions",
                    "basic_concept_understanding",  # colors, shapes, numbers, prepositions
                    "intermediate_concept_understanding",
                    "advanced_concept_understanding",
                    "vocabulary_building_early",  # basic nouns, verbs, adjectives
                    "question_understanding_simple",  # who, what, where
                    # Early Language Acquisition Skills
                    "imitation_and_repetition",  # repeating words and phrases
                    "turn_taking_communication",  # responding in simple exchanges
                    "story_comprehension_simple",  # understanding very short, simple stories
                    "categorization_basic",  # grouping objects and words by simple categories
                    # Scenario Understanding and Prediction
                    "cause_effect_basic",  # understanding simple if-then relationships
                    "sequence_prediction",  # predicting what comes next in simple sequences
                    "outcome_anticipation",  # anticipating likely outcomes in familiar situations
                    "scenario_completion",  # completing partial scenarios with logical endings
                    "character_intention_basic",  # understanding basic motivations and goals
                    # Advanced Scenario Development
                    "narrative_continuation",  # extending stories with logical developments
                    "alternative_outcomes",  # generating plausible alternative scenarios
                    "context_extrapolation",  # predicting developments based on context clues
                    "consequence_chain",  # understanding chains of events and their implications
                    "scenario_adaptation",  # modifying scenarios based on changing conditions
                    # Complex Scenario Analysis
                    "multipath_prediction",  # analyzing multiple possible future developments
                    "probability_reasoning",  # understanding likelihood of different outcomes
                    "scenario_synthesis",  # combining multiple contexts to predict outcomes
                    "temporal_projection",  # understanding how scenarios might develop over time
                    "variable_interaction",  # analyzing how different factors affect outcomes
                ],
            },
            # "kindergarten": {
            #     "age_range": "5-6",
            #     "skill_categories": [
            #         # Reading Fundamentals
            #         "letter_sound_relationships",
            #         "phonological_awareness",
            #         "print_concepts",
            #         "sight_words",
            #         # Basic Writing
            #         "basic_writing",
            #         "name_writing",
            #         # Word Work
            #         "rhyming_words",
            #         "word_families",
            #         "syllable_awareness",
            #         # Comprehension
            #         "story_sequencing",
            #         "prediction_skills",
            #         # Language Skills
            #         "question_answering",
            #     ],
            # },
            # "first_grade": {
            #     "age_range": "6-7",
            #     "skill_categories": [
            #         # Reading Skills
            #         "decoding_strategies",
            #         "word_building",
            #         "fluency_practice",
            #         "phonemic_awareness",
            #         "reading_accuracy",
            #         # Writing Development
            #         "sentence_structure",
            #         "basic_punctuation",
            #         "capitalization",
            #         "spacing_and_formatting",
            #         # Vocabulary and Word Study
            #         "vocabulary_acquisition",
            #         "word_relationships",
            #         "spelling_patterns",
            #         "compound_words_intro",
            #         # Comprehension Skills
            #         "reading_comprehension",
            #         "story_elements",
            #         "retelling_stories",
            #         "making_connections",
            #         "asking_questions",
            #         # Grammar Foundations
            #         "parts_of_speech_basic",
            #         "verb_tenses",
            #         "singular_plural",
            #         "descriptive_language",
            #     ],
            # },
            # "second_grade": {
            #     "age_range": "7-8",
            #     "skill_categories": [
            #         # Word Analysis
            #         "compound_words",
            #         "prefixes_suffixes",
            #         "word_patterns",
            #         "syllable_types",
            #         "irregular_words",
            #         # Reading Strategies
            #         "context_clues",
            #         "main_idea",
            #         "supporting_details",
            #         "text_features",
            #         "comprehension_monitoring",
            #         # Writing Skills
            #         "paragraph_writing",
            #         "topic_sentences",
            #         "narrative_writing",
            #         "informative_writing",
            #         "opinion_writing",
            #         # Language Use
            #         "grammar_basics",
            #         "punctuation_usage",
            #         "synonyms_antonyms",
            #         "homophones",
            #         "contractions",
            #         # Research and Study
            #         "dictionary_skills",
            #         "alphabetical_order",
            #         "note_taking_basic",
            #         # Critical Thinking
            #         "compare_contrast",
            #         "cause_effect",
            #         "fact_opinion",
            #         "drawing_conclusions",
            #     ],
            # },
            # "third_grade": {
            #     "age_range": "8-9",
            #     "skill_categories": [
            #         # Advanced Reading
            #         "reading_strategies",
            #         "inferencing",
            #         "author_purpose",
            #         "text_structure",
            #         "genre_characteristics",
            #         # Writing Process
            #         "writing_process",
            #         "essay_structure",
            #         "editing_revising",
            #         "peer_review",
            #         "writing_conventions",
            #         # Vocabulary Development
            #         "vocabulary_context",
            #         "word_relationships_advanced",
            #         "multiple_meaning_words",
            #         "academic_vocabulary",
            #         "root_words",
            #         # Language Skills
            #         "parts_of_speech",
            #         "sentence_types",
            #         "figurative_language",
            #         "dialogue_usage",
            #         "advanced_punctuation",
            #         # Research and Analysis
            #         "research_skills",
            #         "source_evaluation",
            #         "summarizing",
            #         "paraphrasing",
            #         "organizing_information",
            #         # Literary Analysis
            #         "character_analysis",
            #         "plot_development",
            #         "theme_identification",
            #         "point_of_view",
            #         "literary_devices",
            #         # Study and Reference
            #         "reference_materials",
            #         "note_taking_strategies",
            #         "test_taking_strategies",
            #     ],
            # },
        }

        # Updated task_types with only self-contained tasks
        self.task_types = {
            # Existing task types remain...
            # New Comprehension Tasks
            # "reading_prediction": {
            #     "format": "partial text passages with context clues",
            #     "response_type": "predicting logical story continuations",
            #     "skills": ["prediction", "context analysis", "story understanding"],
            # },
            # "cause_effect_analysis": {
            #     "format": "scenarios describing events and outcomes",
            #     "response_type": "identifying cause-effect relationships",
            #     "skills": [
            #         "logical reasoning",
            #         "sequence understanding",
            #         "relationship analysis",
            #     ],
            # },
            # "compare_contrast": {
            #     "format": "pairs of texts, objects, or concepts for comparison",
            #     "response_type": "identifying similarities and differences",
            #     "skills": [
            #         "analytical thinking",
            #         "detail observation",
            #         "structured comparison",
            #     ],
            # },
            # # Extended Writing Tasks
            # "dialogue_writing": {
            #     "format": "scenarios requiring character conversations",
            #     "response_type": "creating appropriate dialogue exchanges",
            #     "skills": ["conversation flow", "quotation marks", "character voice"],
            # },
            # "descriptive_writing": {
            #     "format": "prompts for sensory-rich descriptions",
            #     "response_type": "creating detailed descriptive passages",
            #     "skills": ["sensory language", "adjective use", "vivid writing"],
            # },
            # "story_continuation": {
            #     "format": "story beginnings requiring completion",
            #     "response_type": "writing logical story endings",
            #     "skills": ["creativity", "plot development", "narrative consistency"],
            # },
            # # Critical Thinking Tasks
            # "fact_opinion": {
            #     "format": "statements for classification",
            #     "response_type": "distinguishing facts from opinions",
            #     "skills": [
            #         "critical thinking",
            #         "evidence evaluation",
            #         "statement analysis",
            #     ],
            # },
            # "author_purpose": {
            #     "format": "text passages with clear author intent",
            #     "response_type": "identifying why the author wrote the text",
            #     "skills": ["purpose analysis", "text evaluation", "author perspective"],
            # },
            # "bias_identification": {
            #     "format": "age-appropriate texts with subtle biases",
            #     "response_type": "recognizing and explaining potential biases",
            #     "skills": [
            #         "critical reading",
            #         "perspective taking",
            #         "fairness evaluation",
            #     ],
            # },
            # # Language Structure Tasks
            # "compound_complex": {
            #     "format": "sentence elements for advanced combination",
            #     "response_type": "creating compound-complex sentences",
            #     "skills": ["advanced syntax", "conjunction use", "sentence crafting"],
            # },
            # "paragraph_organization": {
            #     "format": "topic sentences with supporting details",
            #     "response_type": "organizing coherent paragraphs",
            #     "skills": [
            #         "paragraph structure",
            #         "idea organization",
            #         "topic development",
            #     ],
            # },
            # "transitional_phrases": {
            #     "format": "sentences requiring logical connections",
            #     "response_type": "using appropriate transition words",
            #     "skills": ["flow", "coherence", "connection words"],
            # },
            # # Vocabulary Enhancement Tasks
            # "context_creation": {
            #     "format": "vocabulary words needing context",
            #     "response_type": "creating sentences showing word meaning",
            #     "skills": ["vocabulary use", "context building", "sentence creation"],
            # },
            # "word_connotation": {
            #     "format": "words with similar meanings but different implications",
            #     "response_type": "analyzing emotional meanings of words",
            #     "skills": ["connotation", "word choice", "emotional language"],
            # },
            "word_networks": {
                "format": "central concepts for word association",
                "response_type": "building related word groups",
                "skills": [
                    "word relationships",
                    "semantic networks",
                    "vocabulary expansion",
                ],
            },
            # # Metacognitive Tasks
            "strategy_explanation": {
                "format": "reading or writing problems to solve",
                "response_type": "explaining thought process and strategies",
                "skills": ["metacognition", "strategy use", "process explanation"],
            },
            # "error_analysis": {
            #     "format": "texts with deliberate mistakes",
            #     "response_type": "explaining why something is incorrect",
            #     "skills": [
            #         "error recognition",
            #         "rule understanding",
            #         "correction reasoning",
            #     ],
            # },
            # "self_correction": {
            #     "format": "drafts requiring revision",
            #     "response_type": "identifying and fixing own mistakes",
            #     "skills": ["editing", "self-review", "improvement strategies"],
            # },
            # # Document Tasks
            # "text_features": {
            #     "format": "texts with various structural elements",
            #     "response_type": "identifying and using text features",
            #     "skills": [
            #         "text navigation",
            #         "structure understanding",
            #         "information location",
            #     ],
            # },
            # "genre_analysis": {
            #     "format": "texts of different types",
            #     "response_type": "identifying genre characteristics",
            #     "skills": [
            #         "genre recognition",
            #         "text structure",
            #         "purpose identification",
            #     ],
            # },
            "news_vocab_in_context": {
                "format": "news article snippet with highlighted word",
                "response_type": "explaining the meaning of the highlighted word in the context of the article snippet",
                "skills": [
                    "vocabulary_building",
                    "context_clues",
                    "reading_comprehension",
                ],
            },
            "news_simple_question_answer": {
                "format": "simplified news article and simple questions (who, what, where, when)",
                "response_type": "answering simple questions based on the simplified news article",
                "skills": [
                    "reading_comprehension",
                    "following_directions",
                    "story_comprehension_simple",
                    "question_understanding_simple",
                ],  # skills adjusted for general_children
            },
            "news_main_idea_simple": {
                "format": "short, simplified news article",
                "response_type": "identifying the main idea of the simplified news article in one sentence",
                "skills": [
                    "main_idea",
                    "summarizing",
                    "reading_comprehension",
                    "story_comprehension_simple",
                ],  # skills adjusted
            },
            "news_event_sequence_simple": {
                "format": "very short, simplified news story with jumbled sentences",
                "response_type": "ordering the sentences to reconstruct the news story",
                "skills": [
                    "story_sequencing",
                    "logical_reasoning",
                    "story_comprehension_simple",
                ],  # skills adjusted
            },
            "news_category_identification": {
                "format": "news article snippet",
                "response_type": "identifying the category or topic of the news (e.g., sports, weather, politics, animal news)",
                "skills": [
                    "categorization_basic",
                    "vocabulary_building",
                    "basic_concept_understanding",
                ],  # skills adjusted
            },
            "news_emotion_identification_simple": {
                "format": "very short, simplified news story describing an event (positive or negative)",
                "response_type": "identifying the overall emotion or feeling of the news story (e.g., happy, sad, exciting)",
                "skills": [
                    "nonverbal_communication_understanding",
                    "listening_comprehension",
                    "basic_concept_understanding",
                ],  # emotion as basic concept
            },
            "news_following_directions_simple": {
                "format": "simplified news snippet followed by a simple instruction related to the text (e.g., 'Find the name of the place mentioned.')",
                "response_type": "following the instruction and providing the requested information from the news snippet",
                "skills": [
                    "following_directions",
                    "reading_comprehension",
                    "attention_to_detail",
                ],  # skills adjusted
            },
        }

    # def generate_prompt_template(
    #     self,
    #     grade_level: str,
    #     skill_category: str,
    #     task_type: str,
    #     article_text: str = None,
    # ) -> str:
    #     # Base template with optional article inspiration
    #     article_context = (
    #         f"""
    #     Use this news article as inspiration for generating an engaging task:
    #     {article_text}

    #     Important:
    #     - You can modify or simplify the article's theme/content to be age-appropriate
    #     - The task doesn't need to use the exact article content
    #     - Create a self-contained task that's inspired by the article's topic
    #     - Adapt the complexity for {self.grade_levels[grade_level]['age_range']} year olds
    #     """
    #         if article_text
    #         else ""
    #     )

    #     template = f"""
    #     {article_context}
    #     Create a self-contained '{task_type}' question for {grade_level} students that tests {skill_category}.

    #     Question Requirements:
    #     1. Be age-appropriate for {self.grade_levels[grade_level]['age_range']} year olds
    #     2. Focus specifically on {skill_category}
    #     3. Be engaging and relate to common childhood experiences
    #     4. Use clear, simple language appropriate for the grade level
    #     5. Connect to everyday situations students might encounter
    #     6. Must be completely self-contained - all information needed to answer must be provided in the prompt
    #     7. Avoid references to external materials, images, or resources
    #     8. Include any necessary context or background information within the prompt

    #     Thinking Process Requirements:
    #     1. Include 3-5 step-by-step thoughts that demonstrate clear logical reasoning
    #     2. Make thoughts explicit and age-appropriate
    #     3. Model systematic problem-solving strategies
    #     4. Show connections to information provided in the prompt
    #     5. Use metacognitive strategies appropriate for the grade level
    #     6. Ensure reasoning only uses information from the prompt or common knowledge

    #     Answer Requirements:
    #     1. Provide a complete answer that goes beyond a single word
    #     2. Include a clear explanation using only information from the prompt
    #     3. Give examples that reference the context provided in the prompt
    #     4. Use age-appropriate language for {grade_level} level
    #     5. Reinforce the key concept being tested
    #     6. Verify that the answer can be derived purely from the prompt and reasoning
    #     7. Include any assumptions or background knowledge used in reaching the answer

    #     Format the response as a JSON object with:
    #     {{
    #         "prompt": "the actual question or task, including all necessary context",
    #         "thoughts": [
    #             "step-by-step thinking process",
    #             "analysis of the problem",
    #             "consideration of relevant information from prompt"
    #         ],
    #         "correct_response": "complete answer with explanation",
    #         "difficulty": "easy/medium/hard",
    #         "skills_tested": ["specific skills being tested"]
    #     }}

    #     - Do not reference the article in 'correct_response' or the 'prompt'. Only use the article and its context as an inspiration.
    #     """
    #     return template

    def generate_prompt_template_concise(
        self,
        grade_level: str,
        skill_category: str,
        task_type: str,
        article_text: str = None,
    ) -> str:
        article_context = (
            f"""News Article:
{article_text}
--------------

Create a task directly based on the key facts, themes, or concepts from this article, 
adapted appropriately for {self.grade_levels[grade_level]['age_range']} year olds.
The task must incorporate specific details from the article while being self-contained.
"""
            if article_text
            else ""
        )

        template = f"""{article_context}

Create a learning activity for {grade_level} (age {self.grade_levels[grade_level]['age_range']}) that develops {skill_category} through {task_type}.

Task Requirements:
- Must use specific facts/themes from the article while being self-contained
- Adapt complex concepts to age-appropriate language and context
- Focus on {skill_category} while preserving key article information
- Include necessary factual context required to answer the question accurately

Reasoning Steps (for 'thoughts' JSON field):
- Identify key article facts/concepts that align with {skill_category}
- Show how these are adapted for {grade_level} level understanding
- Demonstrate how the task type ({task_type}) practices the skill

Answer Requirements (for 'correct_response' JSON field):
- Base answer on adapted article information
- Show clear mastery of the targeted skill category
- Use age-appropriate explanations and vocabulary

Response Format (JSON):
{{
    "context": "Essential background information and facts needed to answer the question accurately",
    "prompt": "The question/task (self-contained)",
    "thoughts": ["Step 1: ...", "Step 2: ...", ...],
    "correct_response": "Complete answer with explanation",
    "difficulty": "easy/medium/hard",
    "skills_tested": ["List skills", "{skill_category}"]
}}

Important:
- Task should clearly reflect article content without direct reference
- Maintain factual accuracy while simplifying for target age
- Ensure task type and skill category align naturally
- Focus on one primary skill while supporting overall 
- CRITICAL: All responses must be based solely on information provided in context and prompt
"""
        return template

    def generate_university_scenario_template(
        self,
        course_level: str,
        discipline: str,
        analysis_type: str,
        article_text: str = None,
    ) -> str:
        article_context = (
            f""" Source Material: {article_text} -------------- """
            if article_text
            else ""
        )
        template = f"""{article_context}
    Generate an academic analysis task based on this source material. The task should require students to analyze specific aspects of the content. All necessary information to complete the task must be included in the prompt and context - no external sources should be required.

    Core Requirements:
    1. Task Construction:
    - Frame a specific analytical question or problem
    - Include all necessary background information and definitions

    2. Content Focus:
    - Incorporate concrete examples from the source material
    - Require explicit methodological application
    - Connect theory to empirical evidence
    - Provide necessary explanations

    3. Analysis Depth:
    - Require multi-level analysis (e.g., micro/macro perspectives)
    - Include comparative theoretical analysis
    - Demand evidence-based argumentation
    - Ensure all necessary contextual information is provided

    4. Academic Standards:
    - Include complete explanations of all required concepts

    Response Format (JSON):
    {{
        "prompt": "A clear, specific task that requires students to analyze the source material using methodological approaches. The task should include all necessary background information.",
        "context": "Essential concepts and comprehensive background information needed to complete the analysis.",
        "thoughts": [
            "1. Understand task requirements",
            "2. Identify key themes and evidence from source material",
            "3. Consider appropriate methodological approaches",
            "4. Connect evidence to theoretical arguments",
            "5. Synthesize analysis into coherent conclusions"
        ],
        "correct_response": "Give an example answer here that answers the task specified in the prompt.",
        "difficulty": "intermediate/advanced/expert",
        "skills_assessed": [
            "analytical_reasoning",
            "evidence-based argumentation"
        ]
    }}

    Important Requirements:
    - The prompt must specify a clear, concrete analytical task
    - All information needed to complete the task must be self-contained within the prompt and context
    - No external sources or prior knowledge should be required beyond what is provided
    - Correct response must demonstrate thorough analysis of all task components
"""
        return template

    def generate_counterfactual_reasoning_template(self, article_text: str) -> str:
        template = f"""Given the source material: {article_text}

Transform this real-world scenario into a counterfactual reasoning exercise that tests understanding of complex causal relationships. Create a prompt that presents an alternative scenario where key variables or decisions are changed, requiring analysis of potential outcomes and systemic impacts. The prompt should be completely standalone and not reference the original source material.

Core Requirements:
1. Scenario Construction:
- Present a modified version of real events where key variables are altered
- Maintain realistic constraints and system dynamics
- Include all necessary background context directly in the prompt
- Frame the scenario as a standalone hypothetical situation

2. Causal Analysis:
- Define all relevant stakeholders and their incentives within the prompt
- Include necessary historical context and system relationships
- Present the counterfactual as a natural "what-if" scenario
- Ensure all required information is self-contained

Response Format (JSON):
{{
    "prompt": "A completely standalone counterfactual scenario that contains all necessary background and context. For example, instead of referencing real events, use 'Consider a scenario where [alternative condition]. Given [relevant background and system dynamics], analyze the potential outcomes...'",
    "thoughts": [
        "1. First, I need to identify the baseline conditions and status quo described in the prompt",
        "2. Next, I should map out the key variables that have been altered and understand their initial state",
        "3. I need to analyze the immediate direct effects of these changes on different stakeholders",
        "4. Then, I should consider how these effects might cascade through the system over time",
        "5. Finally, I must evaluate potential feedback loops and long-term equilibrium states"
    ],
    "correct_response": "Provide the actual, complete analysis of the counterfactual scenario, only referencing information that was provided within the prompt itself.",
    "difficulty": "intermediate/advanced/expert",
    "skills_assessed": [
        "conceptual_understanding",
        "explanatory_clarity",
        "relationship_mapping"
    ]
}}

Critical Requirements:
- Prompts must be completely self-contained with no references to external sources
- All necessary background information must be embedded within the prompt
- No mention of source materials or articles in the prompt
- The counterfactual should appear as a natural, standalone scenario
- The correct_response must provide a complete analysis based solely on information in the prompt
- Both prompt and response should be independent of any external context
"""
        return template

    def generate_concept_explanation_template(
        self,
        article_text: str,
    ) -> str:
        template = f"""Given the source material: {article_text}

Generate an educational task that builds upon concepts from the source material, but formulate the prompt as a completely standalone question. The prompt should not reference or depend on any external sources, including the source material itself. All necessary information should be directly embedded within the prompt.

Core Requirements:
1. Task Construction:
- Create a completely self-contained prompt that includes all necessary context
- Incorporate key concepts and examples as part of the prompt's background
- Do not reference or mention any source materials
- Present the task as if the information is general knowledge in the field

2. Knowledge Integration:
- Embed necessary background information directly in the prompt
- Include relevant examples as part of the prompt's scenario
- Define any technical terms within the prompt itself
- Present relationships between concepts as part of the question

Response Format (JSON):
{{
    "prompt": "A completely standalone question that contains all necessary background, examples, and context. The prompt must not reference any external sources or materials. For example, instead of 'Based on the article...' use 'In the field of X, concept Y is known for...'",
    "thoughts": [
        "1. Understand the key concepts and their relationships in the question",
        "2. Identify the main principles and theories that apply",
        "3. Consider relevant examples and their implications",
        "4. Analyze how different components interact",
        "5. Synthesize information into a coherent explanation"
    ],
    "correct_response": "Provide the actual, complete answer to the specific question asked in the prompt. The answer should only reference information that was provided within the prompt itself.",
    "difficulty": "intermediate/advanced/expert",
    "skills_assessed": [
        "conceptual_understanding",
        "explanatory_clarity",
        "relationship_mapping"
    ]
}}

Critical Requirements:
- Prompts must be completely self-contained with no references to external sources
- All necessary information must be embedded within the prompt itself
- No mention of source materials or articles in the prompt
- The task should appear as a natural, standalone question
- The correct_response must be a complete, specific answer to the prompt's question
- Both prompt and response should be independent of any external context
"""
        return template

    def generate_tasks_with_llm(
        self,
        llm_api,
        num_tasks: int = 10,
        specific_grade: str = None,
        specific_category: str = None,
        temperature: float = 0.7,
        article_text: str = None,
    ) -> List[LanguageTask]:
        tasks = []
        for _ in range(num_tasks):
            grade = specific_grade or random.choice(list(self.grade_levels.keys()))
            skill_category = specific_category or random.choice(
                self.grade_levels[grade]["skill_categories"]
            )

            # Modified task type selection to use format strings
            response_types = [
                (key, data["response_type"]) for key, data in self.task_types.items()
            ]
            _, task_type = random.choice(response_types)
            skill_category = "scenario_completion_and_inference"
            task_type = "news_scenario_prediction"
            prompt = self.generate_counterfactual_reasoning_template(
                article_text=article_text + "..." if article_text else None,
            )
            try:
                # response = generate_response_with_google(prompt)
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
                        # "context",
                        "thoughts",
                        "correct_response",
                        "difficulty",
                        # "skills_tested",
                    ]
                    if all(field in task_data for field in required_fields):
                        task = LanguageTask(
                            grade_level=grade,
                            skill_category=skill_category,
                            task_type=task_type,
                            prompt=task_data["prompt"],
                            context=task_data.get("context"),
                            thoughts=task_data["thoughts"],
                            correct_response=task_data["correct_response"],
                            difficulty=task_data["difficulty"],
                            skills_tested=task_data.get("skills_tested", []),
                        )
                        tasks.append(task)
                    else:
                        logging.warning(
                            f"Missing required fields in response: {response[:100]}..."
                        )
                    # time.sleep(5)

                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse LLM response as JSON: {str(e)}")
                    logging.error(f"Response received: {response[:100]}...")
            except Exception as e:
                if "429" in str(e) or "resource exhausted" in str(e).lower():
                    logging.error(f"Resource exhausted (429 error): {str(e)}")
                    break
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
            "context": task.context,
            "thoughts": task.thoughts,
            "correct_response": task.correct_response,
            "difficulty": task.difficulty,
            "skills_tested": task.skills_tested,
        }
        file_handle.write(json.dumps(task_dict, ensure_ascii=False) + "\n")
        file_handle.flush()


def task_generator_worker(
    framework,
    llm_api,
    grade: str,
    category: str,
    num_tasks: int,
    news_queue: Queue,
    temperature: float,
    result_queue: Queue,
    verbose: bool = False,
):
    """Worker function for generating tasks in a thread."""
    try:
        for _ in range(num_tasks):
            try:
                # Get a news article from the queue with timeout
                article = news_queue.get(timeout=300)
            except Queue.empty:
                logging.warning(
                    f"No more news articles available for {grade} {category}"
                )
                break

            # Generate task
            tasks = framework.generate_tasks_with_llm(
                llm_api,
                num_tasks=1,
                specific_grade=grade,
                specific_category=category,
                temperature=temperature,
                article_text=article["article"][:2500],
            )

            if tasks:
                result_queue.put(tasks[0])

            # Note: We no longer put the article back in the queue

    except Exception as e:
        logging.error(f"Error in worker thread: {str(e)}")


def task_writer_worker(
    framework,
    output_file: str,
    result_queue: Queue,
    total_tasks: int,
    verbose: bool = False,
):
    """Worker function for writing tasks to file."""
    tasks_written = 0
    with open(output_file, "w", encoding="utf-8") as f:
        while tasks_written < total_tasks:
            try:
                task = result_queue.get(timeout=300)  # 1 minute timeout
                framework.save_task(task, f)
                tasks_written += 1

                if verbose and tasks_written % 10 == 0:
                    logging.info(f"Written {tasks_written}/{total_tasks} tasks...")

            except Queue.empty:
                logging.warning(
                    "No tasks received for 60 seconds, checking if generation is complete..."
                )
                if tasks_written >= total_tasks:
                    break
            except Exception as e:
                logging.error(f"Error writing task: {str(e)}")


def main(
    output_file: str,
    model_id: str = "microsoft/phi-4",
    openai_api_base: str = "http://localhost:8000/v1",
    num_tasks_per_grade: int = 100,
    grades: List[str] = None,
    categories: List[str] = None,
    temperature: float = 0.7,
    verbose: bool = False,
    news_limit: int = 50000,
    max_workers: int = 6,  # New parameter for controlling thread pool size
):
    """
    Multi-threaded main function that loads recent news articles and generates educational tasks.

    Parameters:
        max_workers: Maximum number of worker threads to use (default: 4)
        (other parameters remain the same)
    """
    # Configure logging
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize framework and API
    framework = EducationalLevelFramework()
    llm_api = VLLMAPI(
        model_id=model_id, openai_api_base=openai_api_base, show_logs=verbose
    )

    # Initialize thread-safe queues
    news_queue = Queue()
    result_queue = Queue()

    # Initialize database connection pool
    conn_pool = create_connection_pool()

    try:
        # Calculate date range for news
        end_date = (datetime.now() - timedelta(days=620)).strftime(
            "%Y-%m-%d"
        )  # datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=720)).strftime("%Y-%m-%d")

        # Load news articles
        if verbose:
            logging.info("Loading news articles...")

        news_df = load_recent_news(
            conn_pool, start_date=start_date, end_date=end_date, limit=news_limit
        )

        news_df["word_count"] = news_df["article"].apply(lambda x: len(str(x).split()))
        news_df = news_df[news_df["word_count"] >= 600]
        print(news_df)
        # Calculate total tasks needed
        if grades is None:
            grades = list(framework.grade_levels.keys())

        total_tasks = len(grades) * num_tasks_per_grade

        # Verify we have enough news articles
        if len(news_df) < total_tasks:
            logging.warning(
                f"Not enough unique news articles ({len(news_df)}) "
                f"for requested tasks ({total_tasks}). Some tasks may not be generated."
            )

        logging.info(
            f"Working with {len(news_df)} news articles for {total_tasks} tasks"
        )

        # Put news articles in the queue
        news_queue = Queue()
        for _, article in news_df.iterrows():
            news_queue.put(article)

        if verbose:
            logging.info(f"Loaded {len(news_df)} news articles")

        if grades is None:
            grades = list(framework.grade_levels.keys())

        # Calculate total tasks and prepare work distribution
        total_tasks = len(grades) * num_tasks_per_grade
        work_items = []

        # Prepare work items for thread pool
        for grade in grades:
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
                work_items.append((grade, category, current_category_tasks))

        # Start writer thread
        output_filename = framework.create_output_filename(output_file, grades)
        writer_thread = threading.Thread(
            target=task_writer_worker,
            args=(framework, output_filename, result_queue, total_tasks, verbose),
        )
        writer_thread.start()

        # Execute task generation in thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for grade, category, num_tasks in work_items:
                future = executor.submit(
                    task_generator_worker,
                    framework,
                    llm_api,
                    grade,
                    category,
                    num_tasks,
                    news_queue,
                    temperature,
                    result_queue,
                    verbose,
                )
                futures.append(future)

            # Wait for all generation to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    logging.error(f"Error in task generation thread: {str(e)}")

        # Wait for writer thread to complete
        writer_thread.join()

    except KeyboardInterrupt:
        logging.warning(
            "\nGeneration interrupted by user. Tasks saved up to this point."
        )

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

    finally:
        if verbose:
            logging.info(f"\nGeneration complete:")
            logging.info(f"- Results saved to {output_filename}")

        # Clean up database connection pool
        if "conn_pool" in locals():
            conn_pool.closeall()


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
