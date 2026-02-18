#!/usr/bin/env python3
"""
Context Length and Pass@K Evaluation Script for MathHay 3s3d Task

Evaluates models across multiple context lengths and calculates pass@K scores.
Uses the same evaluation logic as the base MathHay repository.
"""

import os
import sys
import json
import math
import time
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import tiktoken
import requests

# AWS Bedrock
import boto3
from botocore.exceptions import ClientError

# Google Gemini
import google.generativeai as genai

# HuggingFace Inference Client
try:
    from huggingface_hub import InferenceClient
    HF_INFERENCE_AVAILABLE = True
except ImportError:
    HF_INFERENCE_AVAILABLE = False

# OpenAI for LLM verification
from openai import OpenAI

# Local imports
from bench_generation.utils.tools import load_json_file, save_json_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize tiktoken for token counting
encoding = tiktoken.encoding_for_model("gpt-4o")

def tokenization(text):
    """Calculate tokens using tiktoken."""
    return encoding.encode(text, disallowed_special=())

def decode_tokens(tokens):
    """Decode tokens back to text."""
    return encoding.decode(tokens)

# Model configurations
MODEL_CONFIGS = {
    'gemini-2.5-flash': {
        'provider': 'google',
        'model_id': 'gemini-2.5-flash',
        'input_price': 0.30,
        'output_price': 2.50,
        'family': 'gemini'
    },
    'gpt-oss-120b': {
        'provider': 'bedrock',
        'model_id': 'openai.gpt-oss-120b-1:0',
        'region': 'us-east-1',  # Will be overridden per region
        'input_price': 0.15,  # $0.00015 per 1K tokens = $0.15 per 1M tokens
        'output_price': 0.60,  # $0.0006 per 1K tokens = $0.60 per 1M tokens
        'family': 'openai'
    },
    'deepseek-v3.2': {
        'provider': 'huggingface',
        'model_id': 'deepseek-ai/DeepSeek-V3.2-Exp',
        'hf_provider': 'novita',
        'input_price': 0.28,
        'output_price': 0.42,
        'family': 'deepseek'
    },
    'qwen3-235b': {
        'provider': 'bedrock',
        'model_id': 'qwen.qwen3-235b-a22b-2507-v1:0',
        'region': 'us-west-2',
        'input_price': 0.22,
        'output_price': 0.88,
        'family': 'qwen'
    },
    'qwen3-next': {
        'provider': 'huggingface',
        'model_id': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
        'hf_provider': 'together',
        'input_price': 0.15,
        'output_price': 1.50,
        'family': 'qwen'
    }
}

# GPT-OSS regions (only us-east-1 as requested)
GPT_OSS_REGIONS = [
    'us-east-1'
]

# Context lengths to evaluate
CONTEXT_LENGTHS = [4000, 8000, 12000, 16000, 22000, 32000, 64000, 128000]

# Pass@K values for 128K context
PASS_AT_K_VALUES = [1, 2, 4, 8]

class ContextLengthEvaluator:
    def __init__(self, model_name, haystack_len, placement='middle', region=None):
        self.model_name = model_name
        self.haystack_len = haystack_len
        self.placement = placement
        self.task = '3s3d'  # Only 3s3d task
        
        # Get model config
        if model_name == 'gpt-oss-120b' and region:
            self.config = MODEL_CONFIGS['gpt-oss-120b'].copy()
            self.config['region'] = region
            self.model_display_name = f'gpt-oss-120b-{region}'
        else:
            self.config = MODEL_CONFIGS[model_name].copy()
            self.model_display_name = model_name
        
        # Initialize client based on provider
        if self.config['provider'] == 'bedrock':
            # Priority: 1) Bedrock API key, 2) AWS profile
            bedrock_api_key = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
            aws_profile = os.getenv('AWS_PROFILE', 'bedrock-account')
            
            if bedrock_api_key:
                logger.info("Using Amazon Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)")
                self.bedrock_client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self.config['region']
                )
            else:
                logger.info(f"Using AWS profile: {aws_profile}")
                session = boto3.Session(profile_name=aws_profile)
                self.bedrock_client = session.client(
                    service_name="bedrock-runtime",
                    region_name=self.config['region']
                )
        elif self.config['provider'] == 'google':
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.client = genai.GenerativeModel(self.config['model_id'])
        elif self.config['provider'] == 'huggingface':
            hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not hf_api_key:
                raise ValueError("HUGGINGFACE_API_KEY environment variable must be set for HuggingFace models")
            self.hf_api_key = hf_api_key
            base_model_id = self.config['model_id']
            hf_provider = self.config.get('hf_provider', 'together')
            if hf_provider:
                self.hf_model_id = f"{base_model_id}:{hf_provider}"
            else:
                self.hf_model_id = base_model_id
            
            # Use OpenAI client format with HuggingFace router
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_api_key
            )
            logger.info(f"Using HuggingFace OpenAI client for {self.hf_model_id} (provider: {hf_provider})")
        
        # Initialize OpenAI client for LLM verification
        self.use_llm_verification = False
        self.verification_client = None
        # Check if LLM verification should be disabled
        if not os.getenv("DISABLE_LLM_VERIFICATION"):
            try:
                self.verification_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.use_llm_verification = True
                logger.info(f"LLM verification enabled (GPT-4o)")
            except:
                logger.warning(f"OpenAI API key not set - LLM verification disabled")
                self.use_llm_verification = False
        else:
            logger.info("LLM verification disabled via DISABLE_LLM_VERIFICATION env var")
        
        logger.info(f"Initialized {self.model_display_name} for context length {haystack_len}")
    
    def create_context(self, row, document_data):
        """Create long-context input exactly like the original evaluation (3s3d only)."""
        question = row["Task"]['question']
        longest_irrelevant_documents_indexs = row["Irrelevant_Documents_Indexs"]
        longest_irrelevant_documents = [document_data[doc_idx]["Document"] for doc_idx in longest_irrelevant_documents_indexs]
        joint_longest_irrelevant_documents = "\n\n".join(longest_irrelevant_documents)
        joint_longest_irrelevant_documents_tokens = tokenization(joint_longest_irrelevant_documents)
        
        # Calculate prompt length
        prompt_len = len(tokenization(f"Question: {question}")) + 500  # Buffer for instructions
        
        # Adjust haystack length
        set_haystack_len = self.haystack_len
        if self.haystack_len == 128000:
            set_haystack_len = set_haystack_len - 2000  # Buffer for safety
        
        # Handle 3s3d task (3-step, 3-document)
        relevant_document1 = row['Documents'][0]
        relevant_document2 = row['Documents'][1]
        relevant_document3 = row['Documents'][2]
        relevant_document1_tokens = tokenization(relevant_document1)
        relevant_document2_tokens = tokenization(relevant_document2)
        relevant_document3_tokens = tokenization(relevant_document3)
        relevant_document_tokens = relevant_document1_tokens + relevant_document2_tokens + relevant_document3_tokens
        
        rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
        irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
        
        # Placement (middle-middle-middle by default)
        if self.placement == 'middle' or self.placement == 'middle-middle-middle':
            placement_index1 = rest_tokens // 2
            placement_index2 = rest_tokens // 2
            placement_index3 = rest_tokens // 2
        elif self.placement == 'first' or self.placement == 'first-first-first':
            placement_index1 = 0
            placement_index2 = 0
            placement_index3 = 0
        elif self.placement == 'last' or self.placement == 'last-last-last':
            placement_index1 = rest_tokens
            placement_index2 = rest_tokens
            placement_index3 = rest_tokens
        else:
            # Default: middle-middle-middle
            placement_index1 = rest_tokens // 2
            placement_index2 = rest_tokens // 2
            placement_index3 = rest_tokens // 2
        
        irrelevant_document_tokens_ = irrelevant_document_tokens[:]
        irrelevant_document_tokens_[placement_index3:placement_index3] = relevant_document3_tokens
        irrelevant_document_tokens_[placement_index2:placement_index2] = relevant_document2_tokens
        irrelevant_document_tokens_[placement_index1:placement_index1] = relevant_document1_tokens
        
        # Decode tokens back to text
        long_context_input = decode_tokens(irrelevant_document_tokens_)
        
        return long_context_input
    
    def create_prompt(self, question, long_context_input):
        """Create prompt exactly like base MathHay repo."""
        prompt_template = """Long-Context Documents:
{long_context_input}

You are tasked with solving a mathematical reasoning question using information from Long-Context Documents. Follow these steps to ensure accurate extraction and calculation:

**Instructions:**
1. **Extract Relevant Numerical Information**: Carefully read through the provided Long-Context Documents to identify and list all relevant numerical details. These could include objects, their attributes, numerical values, dates, locations, or any other quantitative data.
   
2. **Analyze and Solve the Question**: Use the identified numerical details to solve the given question. Ensure your solution involves a single computational step based on the relevant data extracted. Focus on logical or arithmetic operations as required by the question.

Question:
{question}

The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object should not be wrapped in triple backticks.

Here is the output schema:
```
{{"properties": {{"reasoning": {{"title": "Reasoning", "description": "Solution process.", "type": "string"}}, "answer": {{"title": "Answer", "description": "The final numerical answer to the question, deduced through reasoning.", "type": "number"}}}}, "required": ["reasoning", "answer"]}}
```"""
        
        return prompt_template.format(question=question, long_context_input=long_context_input)
    
    def invoke_model(self, prompt):
        """Invoke model and return response, tokens, and inference time."""
        start_time = time.time()
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        
        if self.config['provider'] == 'bedrock':
            # Bedrock invocation
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Check if using inference profile (us. prefix) or direct model ID
                    model_id = self.config['model_id']
                    is_inference_profile = model_id.startswith('us.')
                    
                    if is_inference_profile:
                        # Use converse API for inference profiles
                        conversation = [
                            {
                                "role": "user",
                                "content": [{"text": prompt}]
                            }
                        ]
                        response = self.bedrock_client.converse(
                            modelId=model_id,
                            messages=conversation,
                            inferenceConfig={
                                "maxTokens": 16384,
                                "temperature": 0.0
                            }
                        )
                        response_text = response["output"]["message"]["content"][0]["text"]
                        usage = response.get('usage', {})
                        input_tokens = usage.get('inputTokens', 0)
                        output_tokens = usage.get('outputTokens', 0)
                    else:
                        # Use invoke_model for direct model IDs
                        if self.config['family'] == 'anthropic':
                            body = json.dumps({
                                "anthropic_version": "bedrock-2023-05-31",
                                "max_tokens": 16384,
                                "temperature": 0.0,
                                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                            })
                        elif self.config['family'] in ['deepseek', 'qwen', 'openai']:
                            body = json.dumps({
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 16384,
                                "temperature": 0.0
                            })
                        else:
                            body = json.dumps({
                                "prompt": prompt,
                                "max_tokens": 16384,
                                "temperature": 0.0
                            })
                        
                        response = self.bedrock_client.invoke_model(
                            modelId=model_id,
                            body=body,
                            contentType='application/json',
                            accept='application/json'
                        )
                        # Handle response body (could be bytes or already parsed)
                        if hasattr(response['body'], 'read'):
                            response_body = json.loads(response['body'].read())
                        else:
                            response_body = response['body'] if isinstance(response['body'], dict) else json.loads(response['body'])
                        
                        if self.config['family'] == 'anthropic':
                            response_text = response_body['content'][0]['text']
                            usage = response_body.get('usage', {})
                            input_tokens = usage.get('input_tokens', 0)
                            output_tokens = usage.get('output_tokens', 0)
                        elif self.config['family'] in ['deepseek', 'qwen', 'openai']:
                            response_text = response_body['choices'][0]['message']['content']
                            usage = response_body.get('usage', {})
                            if not usage:
                                # Try alternative keys
                                usage = response_body.get('usage_metadata', {})
                            input_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
                            output_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
                            # Fallback to tiktoken if no usage info
                            if input_tokens == 0:
                                input_tokens = len(tokenization(prompt))
                            if output_tokens == 0:
                                output_tokens = len(tokenization(response_text))
                        else:
                            response_text = response_body.get('completion', response_body.get('generated_text', ''))
                            input_tokens = len(tokenization(prompt))
                            output_tokens = len(tokenization(response_text))
                    
                    break
                except Exception as e:
                    error_str = str(e)
                    if 'ThrottlingException' in error_str or 'TooManyRequestsException' in error_str:
                        if attempt < max_retries - 1:
                            delay = (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Throttled, retrying in {delay:.1f}s...")
                            time.sleep(delay)
                        else:
                            raise
                    else:
                        raise
        
        elif self.config['provider'] == 'google':
            # Google Gemini
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.generate_content(prompt)
                    response_text = response.text
                    input_tokens = len(tokenization(prompt))
                    output_tokens = len(tokenization(response_text))
                    break
                except Exception as e:
                    if 'ResourceExhausted' in str(e) and attempt < max_retries - 1:
                        delay = 2 ** attempt + 1
                        logger.warning(f"Rate limited, retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        raise
        
        elif self.config['provider'] == 'huggingface':
            # HuggingFace OpenAI client format
            max_retries = 5
            base_delay = 2
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.hf_model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=16384,
                        temperature=0.0
                    )
                    response_text = response.choices[0].message.content
                    input_tokens = response.usage.prompt_tokens if response.usage else len(tokenization(prompt))
                    output_tokens = response.usage.completion_tokens if response.usage else len(tokenization(response_text))
                    break
                except Exception as e:
                    error_str = str(e)
                    if '503' in error_str or 'loading' in error_str.lower() or 'timeout' in error_str.lower():
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            raise
                    else:
                        raise
        
        inference_time = time.time() - start_time
        return response_text, input_tokens, output_tokens, inference_time
    
    def parse_answer(self, response_text):
        """Extract answer and reasoning from response."""
        try:
            # Try to find JSON in response
            if '```json' in response_text:
                json_start = response_text.index('```json') + 7
                json_end = response_text.rindex('```')
                json_text = response_text[json_start:json_end].strip()
            elif '{' in response_text and '}' in response_text:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                return None, response_text[:1000]
            
            parsed = json.loads(json_text)
            answer = float(parsed.get('answer', -999999))
            reasoning = parsed.get('reasoning', '')
            return answer, reasoning
        except:
            return None, response_text[:500]
    
    def compare_answers(self, expected, predicted):
        """Compare answers with tolerance (numerical comparison)."""
        try:
            if predicted is None:
                return False
            return math.isclose(float(expected), float(predicted), rel_tol=1e-9)
        except:
            return False
    
    def llm_verification(self, solution1, solution2):
        """Use GPT-4o to verify if two solutions are equivalent."""
        if not self.use_llm_verification:
            return "None", "LLM verification not available"
        
        prompt = f"""Your task is to determine if the two given solutions are equivalent in terms of reasoning and final answer.

Solution 1:
{solution1}

Solution 2:
{solution2}

Criteria for equivalence:
1. Both solutions should have the same reasoning steps leading to the final answer.
2. The final numerical answers should be identical.

Please analyze the two solutions and state whether they are the same or different. If different, provide a brief explanation of the discrepancies.

The output should be formatted as a JSON instance that conforms to the JSON schema below.

Here is the output schema:
```
{{"properties": {{"reasoning": {{"title": "Reasoning", "description": "Verification process.", "type": "string"}}, "output": {{"title": "Output", "description": "Yes or No. Yes means the two solutions are equivalent. No means the two solutions are different.", "type": "string"}}}}, "required": ["reasoning", "output"]}}
```"""

        try:
            response = self.verification_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024
            )
            response_text = response.choices[0].message.content
            
            # Parse JSON from response
            if '```json' in response_text:
                json_start = response_text.index('```json') + 7
                json_end = response_text.rindex('```')
                json_text = response_text[json_start:json_end].strip()
            elif '{' in response_text and '}' in response_text:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                return "None", response_text[:200]
            
            parsed = json.loads(json_text)
            return parsed.get('output', 'None'), parsed.get('reasoning', '')
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return "None", f"Error: {str(e)}"
    
    def evaluate_question(self, row, document_data):
        """Evaluate a single question and return result."""
        question = row["Task"]['question']
        expected_answer = row["Task"]['answer']
        
        # Create context and prompt
        long_context_input = self.create_context(row, document_data)
        prompt = self.create_prompt(question, long_context_input)
        
        # Invoke model
        try:
            response_text, input_tokens, output_tokens, inference_time = self.invoke_model(prompt)
            
            # Parse answer
            predicted_answer, reasoning = self.parse_answer(response_text)
            
            # Numerical comparison
            numerical_match = self.compare_answers(expected_answer, predicted_answer)
            
            # LLM verification (with timeout and error handling)
            llm_judge = "None"
            judge_reasoning = ""
            if self.use_llm_verification:
                try:
                    solution1 = row["Task"]["solution"] + "\nAnswer1: " + str(expected_answer)
                    solution2 = reasoning + "\nAnswer2: " + str(predicted_answer if predicted_answer is not None else "None")
                    llm_judge, judge_reasoning = self.llm_verification(solution1, solution2)
                except Exception as e:
                    logger.warning(f"LLM verification failed for question: {e}")
                    llm_judge = "None"
                    judge_reasoning = f"Error: {str(e)}"
            
            # Combined evaluation (like base repo)
            correct = numerical_match or ("yes" in llm_judge.lower())
            
            # Calculate cost
            cost = (input_tokens / 1_000_000) * self.config['input_price'] + \
                   (output_tokens / 1_000_000) * self.config['output_price']
            
            return {
                'correct': correct,
                'numerical_match': numerical_match,
                'llm_judge': llm_judge,
                'predicted_answer': predicted_answer,
                'expected_answer': expected_answer,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost': cost,
                'inference_time': inference_time,
                'response_text': response_text[:1000]  # Store first 1000 chars
            }
        except Exception as e:
            logger.error(f"Error evaluating question: {e}")
            return {
                'correct': False,
                'error': str(e),
                'predicted_answer': None,
                'expected_answer': expected_answer
            }
    
    def evaluate_dataset(self, data_rows, document_data):
        """Evaluate entire dataset and return results."""
        results = []
        correct_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        
        logger.info(f"Starting evaluation: {len(data_rows)} questions")
        
        for idx, row in enumerate(data_rows, 1):
            logger.info(f"Question {idx}/{len(data_rows)}")
            
            result = self.evaluate_question(row, document_data)
            result['question_idx'] = idx
            
            if result.get('correct', False):
                correct_count += 1
            
            if 'input_tokens' in result:
                total_input_tokens += result['input_tokens']
                total_output_tokens += result['output_tokens']
                total_cost += result.get('cost', 0.0)
            
            results.append(result)
            
            # Log progress
            status = "PASS" if result.get('correct', False) else "FAIL"
            logger.info(f"  {status} | {result.get('input_tokens', 0):,} in / {result.get('output_tokens', 0):,} out | ${result.get('cost', 0):.4f}")
            
            # Small delay between questions
            time.sleep(0.5)
        
        accuracy = correct_count / len(data_rows) if data_rows else 0
        
        summary = {
            'model_name': self.model_display_name,
            'context_length': self.haystack_len,
            'task': self.task,
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_questions': len(data_rows),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_cost': total_cost,
            'avg_input_tokens': total_input_tokens / len(data_rows) if data_rows else 0,
            'avg_output_tokens': total_output_tokens / len(data_rows) if data_rows else 0
        }
        
        return results, summary


def evaluate_pass_at_k(evaluator, data_rows, document_data, output_dir, timestamp):
    """Evaluate pass@K by running 8 attempts once and calculating pass@2, pass@4, pass@8."""
    max_k = 8  # Run 8 attempts for all questions
    logger.info(f"Evaluating pass@K: running {max_k} attempts per question for {len(data_rows)} questions")
    
    # Create trajectory directory
    trajectory_dir = output_dir / f"trajectories/{evaluator.model_display_name}_128k_passatk_{timestamp}"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    
    all_questions_results = []
    pass_counts = {1: 0, 2: 0, 4: 0, 8: 0}
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    for idx, row in enumerate(data_rows, 1):
        logger.info(f"Question {idx}/{len(data_rows)} (running {max_k} attempts)")
        
        question = row["Task"]['question']
        expected_answer = row["Task"]['answer']
        
        # Create context and prompt once (same for all attempts)
        long_context_input = evaluator.create_context(row, document_data)
        prompt = evaluator.create_prompt(question, long_context_input)
        
        # Run max_k attempts
        attempts = []
        for attempt_num in range(max_k):
            logger.info(f"  Attempt {attempt_num + 1}/{max_k}")
            
            # Invoke model
            try:
                response_text, input_tokens, output_tokens, inference_time = evaluator.invoke_model(prompt)
                
                # Parse answer
                predicted_answer, reasoning = evaluator.parse_answer(response_text)
                
                # Numerical comparison
                numerical_match = evaluator.compare_answers(expected_answer, predicted_answer)
                
                # LLM verification
                llm_judge = "None"
                judge_reasoning = ""
                if evaluator.use_llm_verification:
                    try:
                        solution1 = row["Task"]["solution"] + "\nAnswer1: " + str(expected_answer)
                        solution2 = reasoning + "\nAnswer2: " + str(predicted_answer if predicted_answer is not None else "None")
                        llm_judge, judge_reasoning = evaluator.llm_verification(solution1, solution2)
                    except Exception as e:
                        logger.warning(f"LLM verification failed: {e}")
                        llm_judge = "None"
                        judge_reasoning = f"Error: {str(e)}"
                
                # Combined evaluation
                correct = numerical_match or ("yes" in llm_judge.lower())
                
                # Calculate cost
                cost = (input_tokens / 1_000_000) * evaluator.config['input_price'] + \
                       (output_tokens / 1_000_000) * evaluator.config['output_price']
                
                attempt_result = {
                    'attempt': attempt_num + 1,
                    'correct': correct,
                    'numerical_match': numerical_match,
                    'llm_judge': llm_judge,
                    'predicted_answer': predicted_answer,
                    'expected_answer': expected_answer,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost': cost,
                    'inference_time': inference_time,
                    'response_text': response_text,  # Full response
                    'reasoning': reasoning,
                    'judge_reasoning': judge_reasoning
                }
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cost += cost
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt_num + 1}: {e}")
                attempt_result = {
                    'attempt': attempt_num + 1,
                    'correct': False,
                    'error': str(e),
                    'predicted_answer': None,
                    'expected_answer': expected_answer
                }
            
            attempts.append(attempt_result)
            time.sleep(0.5)  # Small delay between attempts
        
        # Calculate pass@K for k=1, 2, 4, 8
        for k in [1, 2, 4, 8]:
            k_attempts = attempts[:k]
            if any(attempt.get('correct', False) for attempt in k_attempts):
                pass_counts[k] += 1
        
        # Save trajectory for this question
        question_result = {
            'question_idx': idx,
            'question': question,
            'expected_answer': expected_answer,
            'attempts': attempts,
            'pass@1': attempts[0].get('correct', False) if len(attempts) > 0 else False,
            'pass@2': any(attempt.get('correct', False) for attempt in attempts[:2]),
            'pass@4': any(attempt.get('correct', False) for attempt in attempts[:4]),
            'pass@8': any(attempt.get('correct', False) for attempt in attempts[:8])
        }
        all_questions_results.append(question_result)
        
        # Save individual question trajectory
        with open(trajectory_dir / f"q{idx:03d}_trajectory.json", 'w') as f:
            json.dump(question_result, f, indent=2)
        
        # Save prompt
        with open(trajectory_dir / f"q{idx:03d}_prompt.txt", 'w') as f:
            f.write(prompt)
    
    # Calculate final pass@K metrics
    pass_at_k_results = {}
    for k in [1, 2, 4, 8]:
        pass_at_k = pass_counts[k] / len(data_rows) if data_rows else 0
        pass_at_k_results[f'pass@{k}'] = {
            'pass_at_k': pass_at_k,
            'k': k,
            'pass_count': pass_counts[k],
            'total_questions': len(data_rows),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_cost': total_cost
        }
    
    # Save all results
    with open(trajectory_dir / "all_questions.json", 'w') as f:
        json.dump(all_questions_results, f, indent=2)
    
    logger.info(f"Trajectories saved to {trajectory_dir}")
    
    return pass_at_k_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate models across context lengths and pass@K')
    parser.add_argument('--models', nargs='+', required=True, help='Models to evaluate')
    parser.add_argument('--context-lengths', nargs='+', type=int, default=CONTEXT_LENGTHS,
                       help='Context lengths to evaluate')
    parser.add_argument('--placement', default='middle', help='Document placement strategy')
    parser.add_argument('--pass-at-k', action='store_true', help='Also evaluate pass@K for 128K')
    parser.add_argument('--output-dir', default='outputs/results/context-length-evaluation',
                       help='Output directory for results')
    parser.add_argument('--no-llm-verification', action='store_true',
                       help='Skip LLM verification (faster, uses only numerical comparison)')
    
    args = parser.parse_args()
    
    # Disable LLM verification if requested
    if args.no_llm_verification:
        os.environ["DISABLE_LLM_VERIFICATION"] = "1"
        logger.info("LLM verification disabled (--no-llm-verification flag)")
    
    # Load data (same paths as run_multimodel_evaluation.py)
    input_dir = Path("./outputs/data/March-2024-to-September-2024/")
    task_data_file = input_dir / "full_haystack_question_3s3d.json"
    document_data_file = input_dir / "documents.json"
    
    if not task_data_file.exists() or not document_data_file.exists():
        logger.error(f"Data files not found:")
        logger.error(f"  Task data: {task_data_file} (exists: {task_data_file.exists()})")
        logger.error(f"  Documents: {document_data_file} (exists: {document_data_file.exists()})")
        sys.exit(1)
    
    logger.info(f"Loading task data from {task_data_file}")
    task_data = load_json_file(str(task_data_file))
    
    logger.info(f"Loading document data from {document_data_file}")
    document_data = load_json_file(str(document_data_file))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = {}
    
    # Evaluate each model
    for model_name in args.models:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*80}")
        
        model_results = {}
        
        # Handle GPT-OSS with multiple regions
        if model_name == 'gpt-oss-120b':
            regions_to_evaluate = GPT_OSS_REGIONS
        else:
            regions_to_evaluate = [None]
        
        for region in regions_to_evaluate:
            if region:
                display_name = f"{model_name}-{region}"
            else:
                display_name = model_name
            
            logger.info(f"\nEvaluating {display_name}")
            
            # Evaluate each context length
            for context_len in args.context_lengths:
                logger.info(f"\n{'-'*80}")
                logger.info(f"Context length: {context_len}")
                logger.info(f"{'-'*80}")
                
                evaluator = ContextLengthEvaluator(
                    model_name=model_name,
                    haystack_len=context_len,
                    placement=args.placement,
                    region=region
                )
                
                results, summary = evaluator.evaluate_dataset(task_data, document_data)
                
                logger.info(f"Accuracy: {summary['accuracy']:.2%} ({summary['correct_count']}/{summary['total_questions']})")
                logger.info(f"Total cost: ${summary['total_cost']:.2f}")
                
                # Store results
                key = f"{display_name}_{context_len}"
                model_results[key] = {
                    'summary': summary,
                    'results': results
                }
                
                # Save individual result file
                result_file = output_dir / f"{key}_{timestamp}.json"
                with open(result_file, 'w') as f:
                    json.dump({
                        'summary': summary,
                        'results': results
                    }, f, indent=2)
                
                logger.info(f"Results saved to {result_file}")
            
            # Evaluate pass@K for 128K context
            if args.pass_at_k and 128000 in args.context_lengths:
                logger.info(f"\n{'-'*80}")
                logger.info(f"Evaluating pass@K for {display_name} (128K context)")
                logger.info(f"{'-'*80}")
                
                evaluator = ContextLengthEvaluator(
                    model_name=model_name,
                    haystack_len=128000,
                    placement=args.placement,
                    region=region
                )
                
                # Run 8 attempts once and calculate pass@2, pass@4, pass@8
                pass_at_k_results = evaluate_pass_at_k(evaluator, task_data, document_data, output_dir, timestamp)
                
                # Log results
                for k in [1, 2, 4, 8]:
                    result = pass_at_k_results[f'pass@{k}']
                    logger.info(f"Pass@{k}: {result['pass_at_k']:.2%} ({result['pass_count']}/{result['total_questions']})")
                
                # Save pass@K results summary
                pass_at_k_file = output_dir / f"{display_name}_128k_passatk_{timestamp}.json"
                with open(pass_at_k_file, 'w') as f:
                    json.dump(pass_at_k_results, f, indent=2)
                
                logger.info(f"Pass@K results saved to {pass_at_k_file}")
                
                model_results[f"{display_name}_128k_passatk"] = pass_at_k_results
        
        all_results[model_name] = model_results
    
    # Save combined results
    combined_file = output_dir / f"all_results_{timestamp}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"All evaluations complete!")
    logger.info(f"Combined results saved to {combined_file}")
    logger.info(f"{'='*80}")
    
    # Print summary
    logger.info("\nSummary:")
    for model_name, model_results in all_results.items():
        logger.info(f"\n{model_name}:")
        for key, data in model_results.items():
            if 'summary' in data:
                summary = data['summary']
                logger.info(f"  {key}: {summary['accuracy']:.2%} accuracy, ${summary['total_cost']:.2f} cost")
            elif 'pass@' in key:
                for k, result in data.items():
                    logger.info(f"  {k}: {result['pass_at_k']:.2%}")


if __name__ == "__main__":
    main()

