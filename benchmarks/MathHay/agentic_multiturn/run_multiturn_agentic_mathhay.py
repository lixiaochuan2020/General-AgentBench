#!/usr/bin/env python3
"""
Self-Correcting Agentic MathHay Evaluation

This implements a self-correcting agentic version of MathHay where:
- Agent solves the problem with all documents provided upfront
- Agent self-reflects on whether the answer might be wrong
- If uncertain, agent retries with feedback about what might be wrong
- Test with different max retries (0, 2, 4)

The agent uses self-correction to improve answers rather than asking for more documents.
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
import tiktoken
import requests

# HuggingFace Inference Client
try:
    from huggingface_hub import InferenceClient
    HF_INFERENCE_AVAILABLE = True
except ImportError:
    HF_INFERENCE_AVAILABLE = False

# Add parent directory to path for imports
script_dir = Path(__file__).parent
mathhay_root = script_dir.parent
sys.path.insert(0, str(mathhay_root))

# AWS Bedrock
import boto3
from botocore.exceptions import ClientError

# Google Gemini
import google.generativeai as genai

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

# Model configurations with pricing (same as main script)
MODEL_CONFIGS = {
    'claude-sonnet-4.5': {
        'provider': 'bedrock',
        'model_id': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'region': 'us-east-1',
        'input_price': 3.00,
        'output_price': 15.00,
        'family': 'anthropic'
    },
    'deepseek-r1': {
        'provider': 'bedrock',
        'model_id': 'us.deepseek.r1-v1:0',
        'region': 'us-west-2',
        'input_price': 1.35,
        'output_price': 5.40,
        'family': 'deepseek'
    },
    'deepseek-v3.1': {
        'provider': 'bedrock',
        'model_id': 'deepseek.v3-1-v1:0',
        'region': 'us-east-1',
        'input_price': 0.58,
        'output_price': 1.68,
        'family': 'deepseek'
    },
    'gemini-2.5-flash': {
        'provider': 'google',
        'model_id': 'gemini-2.5-flash',
        'input_price': 0.30,
        'output_price': 2.50,
        'family': 'gemini'
    },
    'gemini-2.5-pro': {
        'provider': 'google',
        'model_id': 'gemini-2.5-pro',
        'input_price': 1.25,  # ≤200K tokens
        'output_price': 10.00,  # ≤200K tokens
        'family': 'gemini'
    },
    'qwen3-32b': {
        'provider': 'bedrock',
        'model_id': 'qwen.qwen3-32b-v1:0',
        'region': 'us-west-2',
        'input_price': 0.15,
        'output_price': 0.60,
        'family': 'qwen'
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
        'hf_provider': 'together',  # Options: 'together', 'hyperbolic', 'novita'
        'input_price': 0.15,
        'output_price': 1.50,
        'family': 'qwen'
    },
    'deepseek-v3.2': {
        'provider': 'huggingface',
        'model_id': 'deepseek-ai/DeepSeek-V3.2-Exp',
        'hf_provider': 'novita',  # DeepSeek V3.2 is only supported on novita
        'input_price': 0.28,
        'output_price': 0.42,
        'family': 'deepseek'
    },
    'gpt-oss-120b': {
        'provider': 'bedrock',
        'model_id': 'openai.gpt-oss-120b-1:0',
        'region': 'us-east-1',
        'input_price': 0.15,  # $0.00015 per 1K tokens = $0.15 per 1M tokens
        'output_price': 0.60,  # $0.0006 per 1K tokens = $0.60 per 1M tokens
        'family': 'openai'
    }
}

class SelfCorrectingMathHayEvaluator:
    def __init__(self, model_name, task, max_retries=2, haystack_len=128000, placement='middle'):
        """
        Initialize self-correcting MathHay evaluator.
        
        Args:
            model_name: Model to use
            task: Task type (2ssd, 3ssd, 2s2d, 3s2d, 3s3d)
            max_retries: Maximum number of retries after initial attempt (0 = no retries)
            haystack_len: Maximum context length in tokens (default: 128000)
            placement: Document placement strategy (default: 'middle')
        """
        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.task = task
        self.max_retries = max_retries
        self.haystack_len = int(haystack_len)
        self.placement = placement
        
        # Initialize the appropriate client
        if self.config['provider'] == 'bedrock':
            bedrock_api_key = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
            aws_profile = os.getenv('AWS_PROFILE', 'bedrock-account')
            
            if bedrock_api_key:
                logger.info("Using Amazon Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)")
                self.client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=self.config['region']
                )
            else:
                logger.info(f"Using AWS profile: {aws_profile}")
                session = boto3.Session(profile_name=aws_profile)
                self.client = session.client(
                    service_name="bedrock-runtime",
                    region_name=self.config['region']
                )
        elif self.config['provider'] == 'google':
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable not set. "
                    "Please set it with: export GEMINI_API_KEY='your_key'"
                )
            genai.configure(api_key=gemini_api_key)
            self.client = genai.GenerativeModel(self.config['model_id'])
            logger.info("Using Google Gemini API")
        elif self.config['provider'] == 'huggingface':
            hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not hf_api_key:
                raise ValueError("HUGGINGFACE_API_KEY environment variable must be set for HuggingFace models")
            self.hf_api_key = hf_api_key
            base_model_id = self.config['model_id']
            # Append provider suffix if specified (e.g., :together, :hyperbolic, :novita)
            hf_provider = self.config.get('hf_provider', 'together')  # Default to 'together' since novita has issues
            if hf_provider:
                self.hf_model_id = f"{base_model_id}:{hf_provider}"
            else:
                self.hf_model_id = base_model_id
            
            # Use OpenAI client format with HuggingFace router (supports :provider suffix)
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_api_key
            )
            logger.info(f"Using HuggingFace OpenAI client for {self.hf_model_id} (provider: {hf_provider})")
        
        # Initialize OpenAI client for LLM verification
        try:
            self.verification_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.use_llm_verification = True
            logger.info(f"LLM verification enabled (GPT-4o)")
        except:
            logger.warning(f"OpenAI API key not set - LLM verification disabled")
            self.verification_client = None
            self.use_llm_verification = False
        
        logger.info(f"Initialized {model_name} for self-correcting evaluation")
        logger.info(f"Max retries: {max_retries} (total attempts: {max_retries + 1})")
    
    def create_context(self, row, document_data):
        """Create long-context input with 128K token limit (same as run_multimodel_evaluation.py)."""
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
        
        # Handle three-document task (3s3d) - most common for this evaluation
        if self.task in ["3s3d"]:
            relevant_document1 = row['Documents'][0]
            relevant_document2 = row['Documents'][1]
            relevant_document3 = row['Documents'][2]
            relevant_document1_tokens = tokenization(relevant_document1)
            relevant_document2_tokens = tokenization(relevant_document2)
            relevant_document3_tokens = tokenization(relevant_document3)
            relevant_document_tokens = relevant_document1_tokens + relevant_document2_tokens + relevant_document3_tokens
            
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
            
            # Placement
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
                # Default: evenly spaced
                placement_index1 = rest_tokens // 4
                placement_index2 = rest_tokens // 2
                placement_index3 = rest_tokens * 3 // 4
            
            irrelevant_document_tokens_ = irrelevant_document_tokens[:]
            irrelevant_document_tokens_[placement_index3:placement_index3] = relevant_document3_tokens
            irrelevant_document_tokens_[placement_index2:placement_index2] = relevant_document2_tokens
            irrelevant_document_tokens_[placement_index1:placement_index1] = relevant_document1_tokens
        
        # Handle two-document tasks (2s2d, 3s2d)
        elif self.task in ["2s2d", "3s2d"]:
            relevant_document1 = row['Documents'][0]
            relevant_document2 = row['Documents'][1]
            relevant_document1_tokens = tokenization(relevant_document1)
            relevant_document2_tokens = tokenization(relevant_document2)
            relevant_document_tokens = relevant_document1_tokens + relevant_document2_tokens
            
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
            
            # Placement
            if self.placement == 'middle' or self.placement == 'middle-middle':
                placement_index1 = rest_tokens // 2
                placement_index2 = rest_tokens // 2
            elif self.placement == 'first' or self.placement == 'first-first':
                placement_index1 = 0
                placement_index2 = 0
            elif self.placement == 'last' or self.placement == 'last-last':
                placement_index1 = rest_tokens
                placement_index2 = rest_tokens
            else:
                placement_index1 = rest_tokens // 2
                placement_index2 = rest_tokens // 2
            
            irrelevant_document_tokens_ = irrelevant_document_tokens[:]
            irrelevant_document_tokens_[placement_index2:placement_index2] = relevant_document2_tokens
            irrelevant_document_tokens_[placement_index1:placement_index1] = relevant_document1_tokens
        
        # Handle single-document tasks (2ssd, 3ssd)
        elif self.task in ["2ssd", "3ssd"]:
            relevant_document = row['Documents'][0]
            relevant_document_tokens = tokenization(relevant_document)
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
            
            # Placement
            if self.placement == 'first':
                placement_index = 0
            elif self.placement == 'middle':
                placement_index = rest_tokens // 2
            elif self.placement == 'last':
                placement_index = rest_tokens
            else:
                placement_index = rest_tokens // 2
            
            irrelevant_document_tokens_ = irrelevant_document_tokens[:]
            irrelevant_document_tokens_[placement_index:placement_index] = relevant_document_tokens
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        
        long_context_input = decode_tokens(irrelevant_document_tokens_)
        return long_context_input
    
    def create_initial_prompt(self, question, documents):
        """Create prompt for initial attempt to solve the question."""
        
        prompt = f"""You are solving a mathematical reasoning question using information from provided documents.

Long-Context Documents:
{documents}

Question:
{question}

Please solve this question step by step and provide your final numerical answer.

The output should be formatted as a JSON instance:
{{"reasoning": "your detailed solution process", "answer": <numerical_answer>}}"""
        
        return prompt
    
    def create_self_reflection_prompt(self, question, previous_reasoning, previous_answer):
        """Create prompt for self-reflection on the previous answer."""
        
        prompt = f"""You previously solved this question:

Question: {question}

Your previous reasoning:
{previous_reasoning}

Your previous answer: {previous_answer}

Now, critically evaluate your previous solution. Consider:
1. Did you use the correct information from the documents?
2. Were your calculations accurate?
3. Did you interpret the question correctly?
4. Are there any logical errors in your reasoning?

The output should be formatted as a JSON instance:
{{"confident": true/false, "feedback": "brief explanation of what might be wrong or what to reconsider"}}

Set confident to true if you believe your answer is correct, or false if you think there might be issues."""
        
        return prompt
    
    def create_retry_prompt(self, question, documents, previous_attempts, feedback):
        """Create prompt for retry attempt with feedback."""
        
        # Build history of previous attempts
        history_text = "[Previous Attempts]:\n"
        for i, attempt in enumerate(previous_attempts):
            history_text += f"\nAttempt {i+1}:\n"
            history_text += f"Reasoning: {attempt['reasoning'][:300]}...\n"
            history_text += f"Answer: {attempt['answer']}\n"
        
        prompt = f"""You are solving a mathematical reasoning question using information from provided documents.

{history_text}

Self-Critique Feedback:
{feedback}

Long-Context Documents:
{documents}

Question:
{question}

Based on the feedback about your previous attempt(s), solve this question again. Pay special attention to the issues identified in the feedback.

The output should be formatted as a JSON instance:
{{"reasoning": "your detailed solution process", "answer": <numerical_answer>}}"""
        
        return prompt
    
    def parse_answer_response(self, response_text):
        """Parse response containing reasoning and answer."""
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
                return None, None
            
            parsed = json.loads(json_text)
            
            answer = parsed.get('answer')
            reasoning = parsed.get('reasoning', '')
            
            if answer is not None:
                try:
                    answer = float(answer)
                except:
                    answer = None
            
            return reasoning, answer
            
        except Exception as e:
            logger.warning(f"Failed to parse answer response: {e}")
            return response_text[:500], None
    
    def parse_reflection_response(self, response_text):
        """Parse self-reflection response."""
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
                return False, response_text[:500]
            
            parsed = json.loads(json_text)
            
            confident = parsed.get('confident', True)
            feedback = parsed.get('feedback', '')
            
            return confident, feedback
            
        except Exception as e:
            logger.warning(f"Failed to parse reflection response: {e}")
            # Default to not confident if parsing fails
            return False, response_text[:500]
    
    def invoke_model(self, prompt):
        """Invoke the model and return response with token counts."""
        start_time = time.time()
        
        if self.config['provider'] == 'bedrock':
            max_retries = 5
            base_delay = 2
            for attempt in range(max_retries):
                try:
                    is_inference_profile = self.config['model_id'].startswith('us.')
                    
                    if is_inference_profile:
                        conversation = [
                            {
                                "role": "user",
                                "content": [{"text": prompt}]
                            }
                        ]
                        response = self.client.converse(
                            modelId=self.config['model_id'],
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
                        if self.config['family'] == 'anthropic':
                            body = {
                                "anthropic_version": "bedrock-2023-05-31",
                                "max_tokens": 16384,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [{"type": "text", "text": prompt}]
                                    }
                                ],
                                "temperature": 0.0
                            }
                        elif self.config['family'] == 'deepseek':
                            body = {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": prompt
                                    }
                                ],
                                "max_tokens": 16384,
                                "temperature": 0.0
                            }
                        elif self.config['family'] in ['qwen', 'openai']:
                            body = {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": prompt
                                    }
                                ],
                                "max_tokens": 16384,
                                "temperature": 0.0
                            }
                        
                        response_obj = self.client.invoke_model(
                            modelId=self.config['model_id'],
                            body=json.dumps(body)
                        )
                        response_body = json.loads(response_obj['body'].read())
                        
                        if self.config['family'] == 'anthropic':
                            response_text = response_body['content'][0]['text']
                            usage = response_body.get('usage', {})
                            input_tokens = usage.get('input_tokens', 0)
                            output_tokens = usage.get('output_tokens', 0)
                        elif self.config['family'] in ['deepseek', 'qwen', 'openai']:
                            response_text = response_body['choices'][0]['message']['content']
                            usage = response_body.get('usage', {})
                            if not usage:
                                usage = response_body
                            
                            input_tokens = (
                                usage.get('input_tokens') or
                                usage.get('prompt_tokens') or
                                usage.get('inputTokens') or
                                usage.get('promptTokens') or
                                0
                            )
                            output_tokens = (
                                usage.get('output_tokens') or
                                usage.get('completion_tokens') or
                                usage.get('outputTokens') or
                                usage.get('completionTokens') or
                                0
                            )
                            
                            if input_tokens == 0:
                                input_tokens = len(tokenization(prompt))
                            if output_tokens == 0:
                                output_tokens = len(tokenization(response_text))
                    
                    break
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                    error_message = e.response.get('Error', {}).get('Message', str(e))
                    
                    if 'ThrottlingException' in error_code or 'Too many tokens' in error_message:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            raise
                    else:
                        raise
                except Exception as e:
                    error_str = str(e)
                    if 'ThrottlingException' in error_str or 'Too many tokens' in error_str:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            raise
                    else:
                        raise
        
        elif self.config['provider'] == 'google':
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
            max_retries = 5
            base_delay = 2
            hf_provider = self.config.get('hf_provider', 'together')
            for attempt in range(max_retries):
                try:
                    # Use OpenAI client format (supports :provider suffix via router)
                    response = self.client.chat.completions.create(
                        model=self.hf_model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=16384,
                        temperature=0.0
                    )
                    
                    # Extract response text
                    response_text = response.choices[0].message.content
                    
                    # Get token counts
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
    
    def compare_answers(self, expected, predicted):
        """Compare answers with tolerance."""
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

Please analyze the two solutions and state whether they are the same or different. If different, provide a brief explanation.

The output should be formatted as a JSON instance:
{{"output": "Yes or No", "reasoning": "your explanation"}}"""

        try:
            response = self.verification_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024
            )
            response_text = response.choices[0].message.content
            
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
    
    def run_self_correcting_evaluation(self, question, row, document_data, expected_answer):
        """
        Run self-correcting evaluation for a single question.
        
        Returns:
            dict with: attempts_made, final_answer, final_reasoning, correct, 
                      attempt_details (list of attempt info), total_input_tokens, 
                      total_output_tokens, total_cost
        """
        # Create context with 128K token limit (like multimodel evaluation)
        documents = self.create_context(row, document_data)
        
        total_input_tokens = 0
        total_output_tokens = 0
        attempt_details = []
        previous_attempts = []
        final_answer = None
        final_reasoning = None
        
        # Initial attempt
        attempt_num = 0
        logger.info(f"  Attempt {attempt_num + 1}: Initial solution")
        
        prompt = self.create_initial_prompt(question, documents)
        response_text, input_tokens, output_tokens, inference_time = self.invoke_model(prompt)
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        reasoning, answer = self.parse_answer_response(response_text)
        
        attempt_detail = {
            'attempt_num': attempt_num + 1,
            'attempt_type': 'initial',
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'response_text': response_text,
            'reasoning': reasoning,
            'answer': answer,
            'inference_time': inference_time
        }
        attempt_details.append(attempt_detail)
        
        final_answer = answer
        final_reasoning = reasoning
        previous_attempts.append({'reasoning': reasoning, 'answer': answer})
        
        # Self-correction loop
        for retry_num in range(self.max_retries):
            # Check if we should retry
            if answer is None:
                logger.info(f"  Attempt {attempt_num + 1}: No valid answer, forcing retry")
                should_retry = True
                feedback = "Previous attempt did not produce a valid numerical answer. Please solve the question and provide a numerical answer."
            else:
                # Self-reflection
                logger.info(f"  Self-reflection after attempt {attempt_num + 1}")
                
                reflection_prompt = self.create_self_reflection_prompt(question, reasoning, answer)
                reflection_response, refl_input_tokens, refl_output_tokens, refl_inference_time = self.invoke_model(reflection_prompt)
                
                total_input_tokens += refl_input_tokens
                total_output_tokens += refl_output_tokens
                
                confident, feedback = self.parse_reflection_response(reflection_response)
                
                # Store reflection details
                reflection_detail = {
                    'attempt_num': attempt_num + 1,
                    'attempt_type': 'reflection',
                    'input_tokens': refl_input_tokens,
                    'output_tokens': refl_output_tokens,
                    'response_text': reflection_response,
                    'confident': confident,
                    'feedback': feedback,
                    'inference_time': refl_inference_time
                }
                attempt_details.append(reflection_detail)
                
                should_retry = not confident
                
                if confident:
                    logger.info(f"  Model is confident in answer: {answer}")
                    break
                else:
                    logger.info(f"  Model is uncertain, will retry. Feedback: {feedback[:100]}...")
            
            # Retry if needed and we have retries left
            if should_retry and retry_num < self.max_retries:
                attempt_num += 1
                logger.info(f"  Attempt {attempt_num + 1}: Retry with feedback")
                
                retry_prompt = self.create_retry_prompt(question, documents, previous_attempts, feedback)
                retry_response, retry_input_tokens, retry_output_tokens, retry_inference_time = self.invoke_model(retry_prompt)
                
                total_input_tokens += retry_input_tokens
                total_output_tokens += retry_output_tokens
                
                reasoning, answer = self.parse_answer_response(retry_response)
                
                retry_detail = {
                    'attempt_num': attempt_num + 1,
                    'attempt_type': 'retry',
                    'input_tokens': retry_input_tokens,
                    'output_tokens': retry_output_tokens,
                    'response_text': retry_response,
                    'reasoning': reasoning,
                    'answer': answer,
                    'feedback_used': feedback,
                    'inference_time': retry_inference_time
                }
                attempt_details.append(retry_detail)
                
                final_answer = answer
                final_reasoning = reasoning
                previous_attempts.append({'reasoning': reasoning, 'answer': answer})
        
        # Evaluate correctness
        numerical_match = self.compare_answers(expected_answer, final_answer) if final_answer is not None else False
        
        llm_judge = "None"
        judge_reasoning = ""
        if self.use_llm_verification and final_answer is not None:
            solution1 = row["Task"]["solution"] + "\nAnswer1: " + str(expected_answer)
            solution2 = (final_reasoning or "") + "\nAnswer2: " + str(final_answer)
            llm_judge, judge_reasoning = self.llm_verification(solution1, solution2)
        
        correct = numerical_match or ("yes" in str(llm_judge).lower())
        
        # Calculate cost
        input_cost = (total_input_tokens / 1_000_000) * self.config['input_price']
        output_cost = (total_output_tokens / 1_000_000) * self.config['output_price']
        total_cost = input_cost + output_cost
        
        return {
            'attempts_made': len([a for a in attempt_details if a['attempt_type'] in ['initial', 'retry']]),
            'final_answer': final_answer,
            'final_reasoning': final_reasoning,
            'expected_answer': expected_answer,
            'correct': correct,
            'numerical_match': numerical_match,
            'llm_judge': llm_judge,
            'judge_reasoning': judge_reasoning,
            'attempt_details': attempt_details,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_cost': total_cost
        }
    
    def evaluate_dataset(self, data_rows, document_data, output_dir):
        """Evaluate entire dataset with self-correcting approach."""
        results = []
        correct_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_dir = Path(output_dir) / f"selfcorrect_trajectories/{self.model_name}_{self.task}_{self.max_retries}retries_{timestamp}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting self-correcting evaluation: {len(data_rows)} questions, max {self.max_retries} retries")
        logger.info(f"Trajectories: {trajectory_dir}")
        
        for idx, row in enumerate(data_rows):
            question = row["Task"]['question']
            expected_answer = row["Task"]["answer"]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {idx+1}/{len(data_rows)}")
            logger.info(f"{'='*80}")
            
            try:
                # Run self-correcting evaluation
                result = self.run_self_correcting_evaluation(question, row, document_data, expected_answer)
                
                total_input_tokens += result['total_input_tokens']
                total_output_tokens += result['total_output_tokens']
                
                if result['correct']:
                    correct_count += 1
                
                # Add question metadata
                result['question_idx'] = idx
                result['question'] = question
                result['model_name'] = self.model_name
                result['task'] = self.task
                result['max_retries'] = self.max_retries
                
                results.append(result)
                
                # Save trajectory
                with open(trajectory_dir / f"q{idx:03d}_trajectory.json", 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Log result
                status = "PASS" if result['correct'] else "FAIL"
                logger.info(f"  {status} | {result['attempts_made']} attempts | {result['total_input_tokens']:,} in / {result['total_output_tokens']:,} out | ${result['total_cost']:.4f}")
                logger.info(f"  Expected: {expected_answer}, Predicted: {result['final_answer']}")
                
                time.sleep(1)  # Small delay between questions
                
            except Exception as e:
                logger.error(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'question_idx': idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'error': str(e)
                })
                
                with open(trajectory_dir / f"q{idx:03d}_error.json", 'w') as f:
                    json.dump({'error': str(e), 'traceback': traceback.format_exc()}, f, indent=2)
        
        # Calculate summary
        accuracy = correct_count / len(data_rows) if data_rows else 0
        total_cost = (total_input_tokens / 1_000_000) * self.config['input_price'] + \
                     (total_output_tokens / 1_000_000) * self.config['output_price']
        
        summary = {
            'model_name': self.model_name,
            'task': self.task,
            'max_retries': self.max_retries,
            'total_questions': len(data_rows),
            'correct': correct_count,
            'accuracy': accuracy,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_cost': total_cost,
            'avg_attempts_per_question': sum(r.get('attempts_made', 0) for r in results) / len(results) if results else 0,
            'trajectory_directory': str(trajectory_dir),
            'results': results
        }
        
        # Save summary
        output_file = Path(output_dir) / f"selfcorrect_results_{self.model_name}_{self.task}_{self.max_retries}retries_{timestamp}.json"
        save_json_file(str(output_file), summary)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(data_rows)})")
        logger.info(f"Total cost: ${total_cost:.2f}")
        logger.info(f"Avg attempts: {summary['avg_attempts_per_question']:.1f}")
        logger.info(f"{'='*80}\n")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Self-correcting agentic MathHay evaluation')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['qwen3-32b', 'qwen3-next', 'deepseek-v3.2', 'gemini-2.5-flash', 'deepseek-v3.1'],
                       help='Models to test (defaults to 5 cheaper models)')
    parser.add_argument('--tasks', type=str, nargs='+',
                       default=['3s3d'],
                       help='Tasks to run (start with 3s3d as suggested)')
    parser.add_argument('--max_retries', type=int, nargs='+',
                       default=[2, 4],
                       help='Maximum number of retries to test (default: [2, 4])')
    parser.add_argument('--input_dir', type=str, 
                       default='./outputs/data/March-2024-to-September-2024/',
                       help='Input directory')
    parser.add_argument('--output_dir', type=str,
                       default='./outputs/results/March-2024-to-September-2024/',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load documents
    document_path = os.path.join(args.input_dir, 'documents.json')
    document_data = load_json_file(document_path)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SELF-CORRECTING AGENTIC MATHHAY EVALUATION")
    logger.info(f"{'='*80}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Max retries: {args.max_retries}")
    logger.info(f"{'='*80}\n")
    
    all_summaries = {}
    
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            continue
        
        for task in args.tasks:
            # Load task data
            data_file = f"full_haystack_question_{task}.json"
            data_path = os.path.join(args.input_dir, data_file)
            data_rows = load_json_file(data_path)
            
            logger.info(f"Loaded {len(data_rows)} questions from {task}")
            
            # Test each max_retries configuration
            for max_retries in args.max_retries:
                logger.info(f"\n{'#'*80}")
                logger.info(f"# {model_name} on {task} - {max_retries} retries (max {max_retries+1} attempts)")
                logger.info(f"{'#'*80}\n")
                
                evaluator = SelfCorrectingMathHayEvaluator(
                    model_name,
                    task,
                    max_retries=max_retries,
                    haystack_len=128000,
                    placement='middle'
                )
                
                summary = evaluator.evaluate_dataset(
                    data_rows,
                    document_data,
                    args.output_dir
                )
                
                all_summaries[f"{model_name}_{task}_{max_retries}retries"] = summary
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"ALL EVALUATIONS COMPLETE")
    logger.info(f"{'='*80}\n")
    
    for key, summary in all_summaries.items():
        logger.info(f"{key}:")
        logger.info(f"  Accuracy: {summary['accuracy']:.1%}")
        logger.info(f"  Cost: ${summary['total_cost']:.2f}")
        logger.info(f"  Avg attempts: {summary['avg_attempts_per_question']:.1f}")

if __name__ == '__main__':
    main()

