'''
Arguments:
    --batch_file: path to json file containing array of id and question fields
    --answer_dir: result directory

'''

prompt_with_options = """Please answer the question based on the options provided. Only choose one option. Give your answer in format ANSWER: [your_choice].

Question: {question}

Options:
{options}

"""

prompt_without_options = """Please answer the following question directly and concisely.

Question: {question}

"""

import os
import argparse
from openai import OpenAI
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import json
import traceback
import time
import random
import google.generativeai as generativeai
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, ThinkingConfig
from utils.token_calculator import tokenize


# Load environment variables from keys.env file
load_dotenv('keys.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
generativeai.configure(api_key=GEMINI_API_KEY)

CONCURRENT_NUM = 120
MODEL_ID = "gemini-2.5-pro"
MODEL_ID_Flash = "gemini-2.5-flash"

class SimpleLLMClient:
    def __init__(self, is_qwen: bool = True, is_flash: bool = True):
        self.is_qwen = is_qwen
        if is_qwen:
            self.client = OpenAI(
                api_key='EMPTY',
                base_url="http://localhost:8000/v1"
            )
            self.model_name = self.client.models.list().data[0].id
        else:
            # Use Gemini model
            if is_flash:
                self.model_name = MODEL_ID_Flash
            else:
                self.model_name = MODEL_ID
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        print(f"#######\nInit SimpleLLMClient with model {self.model_name}\n#######")
        time.sleep(5)

    def answer_question(self, prompt, question_id, answer_dir):
        """Directly answer a question using the model"""
        print(f"Running question {question_id}")

        try:
            # Directly call the model to get an answer
            if self.is_qwen:
                thought, response = self._query_qwen(prompt)
            else:
                thought, response = self._query_gemini(prompt)

            thought_length = tokenize(thought)

            answer_file = f"{answer_dir}/result_{question_id}.json"
            
            # Save results
            with open(answer_file, 'w', encoding='utf-8') as f:
                result = {
                    "model": self.model_name,
                    "answer": response,
                    "thought_length": thought_length
                }
                json.dump(result, f, indent=4)
                
            print(f"Question {question_id} result saved to {answer_dir}/result_{question_id}.json\n")
            
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            print(traceback.format_exc())

    def answer_questions_parallel(self, prompts, question_ids, answer_dir):
        """Process multiple questions in parallel"""
        print(f"Running {len(prompts)} questions in parallel with {CONCURRENT_NUM} workers")
        with ThreadPoolExecutor(max_workers=CONCURRENT_NUM) as executor:
            futures = [executor.submit(self.answer_question, prompt, question_id, answer_dir) 
                for prompt, question_id in zip(prompts, question_ids)]
            concurrent.futures.wait(futures)

    def _query_qwen(self, prompt):
        """Query the Qwen model"""
        max_try_times = 3
        for _ in range(max_try_times):
            try:
                qwen_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,
                    max_tokens=20480,
                )
                
                thought = qwen_response.choices[0].message.reasoning_content
                original_response = qwen_response.choices[0].message.content
                return thought, original_response

            except Exception as e:
                print(f"Error: {e}")
                if _ == max_try_times - 1:
                    raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")
                time.sleep(random.randint(1, 3))

        return "Error: Failed to get response"

    def _query_gemini(self, prompt):
        """Query the Gemini model"""
        max_try_times = 5
        for _ in range(max_try_times):
            try:
                gemini_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=GenerateContentConfig(
                        thinking_config=ThinkingConfig(include_thoughts=True),
                        max_output_tokens=20480
                    ),
                )

                original_response = None
                thought = None
                 
                for part in gemini_response.candidates[0].content.parts:
                    if not part.text:
                        continue
                    if part.thought:
                        thought = part.text
                    else:
                        original_response = part.text

                if original_response is not None:
                    return thought, original_response
                if original_response is None and thought is not None:
                    print(f"No response found, but thought is not None")
                    return thought, ""

            except Exception as e:
                print(f"Error: {e}")
                if _ == max_try_times - 1:
                    raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")
                time.sleep(random.randint(1, 3))

        return "Error: Failed to get response"

def load_questions_from_file(file_path):
    """Load questions from a JSON file (structured as an array of objects with 'id' and 'question' fields)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [item["question"] for item in data]
    ids = [item["id"] for item in data]
    options = []
    for item in data:
        if "options" in item:
            options.append(item["options"])
    return questions, ids, options

def filter_completed_questions(questions, ids, options, answer_dir):
    """Filter out questions that already have answer files"""
    filtered_questions = []
    filtered_ids = []
    filtered_options = []
    completed_count = 0
    
    for i, question_id in enumerate(ids):
        answer_file = f"{answer_dir}/result_{question_id}.json"
        if os.path.exists(answer_file):
            completed_count += 1
        else:
            filtered_questions.append(questions[i])
            filtered_ids.append(question_id)
            if i < len(options):
                filtered_options.append(options[i])
    
    return filtered_questions, filtered_ids, filtered_options, completed_count

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_file', type=str, help='Path to json file containing array of id and question fields')
    parser.add_argument('--answer_dir', type=str, default='results', help='Result directory')
    parser.add_argument('--is_qwen', action='store_true', help='Use Qwen model')
    parser.add_argument('--is_flash', action='store_true', help='Use Gemini Flash model')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
   
    answer_dir = args.answer_dir
    is_qwen = args.is_qwen
    is_flash = args.is_flash
   
    # make sure answer_dir exists
    os.makedirs(answer_dir, exist_ok=True)

    # Load questions from file
    questions, ids, options = load_questions_from_file(args.batch_file)
    total_questions = len(questions)
    print(f"Loaded {total_questions} questions from {args.batch_file}")
    
    # Filter out completed questions
    questions, ids, options, completed_count = filter_completed_questions(questions, ids, options, answer_dir)
    remaining_questions = len(questions)
    
    print(f"Total dataset: {total_questions} questions")
    print(f"Already completed: {completed_count} questions")
    print(f"Remaining to process: {remaining_questions} questions")
    
    # If no questions to process, exit
    if remaining_questions == 0:
        print("All questions have been completed!")
        exit(0)
    
    # Build prompts
    prompts = []
    if len(options) > 0 and all(opt for opt in options):
        # Use options format if options are available
        for question, options_list in zip(questions, options):
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            options_str = ""
            for i, option in enumerate(options_list):
                options_str += f"{letters[i]}) {option}\n"
            prompt = prompt_with_options.format(question=question, options=options_str)
            prompts.append(prompt)
    else:
        # Use no-options format for questions without options
        for question in questions:
            prompt = prompt_without_options.format(question=question)
            prompts.append(prompt)
    
    # Create a simplified client
    client = SimpleLLMClient(is_qwen=is_qwen, is_flash=is_flash)

    print(f"Model: {client.model_name}")
    if is_qwen:
        print(f"Using Qwen model...")
    else:
        print(f"Using Gemini {'Flash' if is_flash else 'Pro'} model...")
    print(f"Direct answer mode...")

    # Process all questions in parallel
    client.answer_questions_parallel(prompts, ids, answer_dir)