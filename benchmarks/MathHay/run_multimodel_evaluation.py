#!/usr/bin/env python3
"""
Multi-Model MathHay Evaluation Script
Runs inference on 2-step and 3-step questions for multiple models.

Supported models:
- Claude Sonnet 4.5 (Bedrock)
- DeepSeek R1 (Bedrock)
- DeepSeek V3.1 (Bedrock) 
- Gemini 2.5 Flash (Google API)
- Qwen3 32B (Bedrock)
- Qwen3 235B (Bedrock)
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

# Model configurations with pricing
MODEL_CONFIGS = {
    'claude-sonnet-4.5': {
        'provider': 'bedrock',
        'model_id': 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
        'region': 'us-east-1',  # Inference profile available in us-east-1, us-east-2, us-west-2
        'input_price': 3.00,
        'output_price': 15.00,
        'family': 'anthropic'
    },
    'deepseek-r1': {
        'provider': 'bedrock',
        'model_id': 'us.deepseek.r1-v1:0',  # Inference profile (requires us. prefix)
        'region': 'us-east-1',  # DeepSeek R1 is in us-east-1
        'input_price': 1.35,  # $0.00135 per 1K tokens = $1.35 per 1M tokens
        'output_price': 5.40,  # $0.0054 per 1K tokens = $5.40 per 1M tokens
        'family': 'deepseek'
    },
    'claude-haiku-4.5': {
        'provider': 'bedrock',
        'model_id': 'us.anthropic.claude-haiku-4-5-20251001-v1:0',  # Inference profile (requires us. prefix)
        'region': 'us-east-1',  # Claude Haiku 4.5 is in us-east-1
        'input_price': 1.00,  # $0.001 per 1K tokens = $1.00 per 1M tokens
        'output_price': 5.00,  # $0.005 per 1K tokens = $5.00 per 1M tokens
        'family': 'anthropic'
    },
    'deepseek-v3.1': {
        'provider': 'bedrock',
        'model_id': 'deepseek.v3-1-v1:0',  # NO us. prefix
        'region': 'us-east-1',  # DeepSeek V3.1 is in us-east-1
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
        'hf_provider': 'together',  # Options: 'together', 'hyperbolic', 'novita' (novita doesn't support text-generation)
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
    'gpt-5': {
        'provider': 'openai',
        'model_id': 'gpt-5-2025-08-07',
        'input_price': 1.25,  # Official pricing: $1.25 per 1M input tokens
        'output_price': 10.00,  # Official pricing: $10.00 per 1M output tokens
        'family': 'openai'
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

class MultiModelEvaluator:
    def __init__(self, model_name, task, haystack_len='128000', placement='middle'):
        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.task = task
        self.haystack_len = int(haystack_len)
        self.placement = placement
        
        # Initialize the appropriate client
        if self.config['provider'] == 'bedrock':
            # Priority: 1) Bedrock API key, 2) AWS profile
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
                # Verify which account is being used
                try:
                    sts = session.client('sts')
                    identity = sts.get_caller_identity()
                    account_id = identity.get('Account', 'Unknown')
                    logger.info(f"Using AWS Account: {account_id}")
                    if account_id != '970547356481':
                        logger.warning(f"WARNING: Expected account 970547356481, but using {account_id}")
                except Exception as e:
                    logger.warning(f"Could not verify AWS account: {e}")
        elif self.config['provider'] == 'google':
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.client = genai.GenerativeModel(self.config['model_id'])
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
        elif self.config['provider'] == 'openai':
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set for OpenAI models")
            self.client = OpenAI(api_key=openai_api_key)
            logger.info(f"Using OpenAI client for {self.config['model_id']}")
        
        # Initialize OpenAI client for LLM verification (like base repo does)
        try:
            self.verification_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.use_llm_verification = True
            logger.info(f"LLM verification enabled (GPT-4o)")
        except:
            logger.warning(f"OpenAI API key not set - LLM verification disabled")
            self.verification_client = None
            self.use_llm_verification = False
        
        logger.info(f"Initialized {model_name} in region {self.config.get('region', 'google')}")
    
    def create_context(self, row, document_data):
        """Create long-context input exactly like the original evaluation."""
        question = row["Task"]['question']
        longest_irrelevant_documents_indexs = row["Irrelevant_Documents_Indexs"]
        longest_irrelevant_documents = [document_data[doc_idx]["Document"] for doc_idx in longest_irrelevant_documents_indexs]
        joint_longest_irrelevant_documents = "\n\n".join(longest_irrelevant_documents)
        joint_longest_irrelevant_documents_tokens = tokenization(joint_longest_irrelevant_documents)
        
        # Calculate prompt length (like base repo uses prompt_len_cal)
        prompt_len = len(tokenization(f"Question: {question}")) + 500  # Add buffer for instructions
        
        # Adjust haystack length (like base repo line 352, 382, 432)
        set_haystack_len = self.haystack_len
        if self.haystack_len == 128000:
            set_haystack_len = set_haystack_len - 2000  # Buffer for safety (like base repo)
        
        # Handle single-document tasks (2ssd, 3ssd) - like base repo lines 347-372
        if self.task in ["2ssd", "3ssd"]:
            relevant_document = row['Documents'][0]
            relevant_document_tokens = tokenization(relevant_document)
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
            
            # Placement (like base repo lines 361-368)
            if self.placement == 'first':
                placement_index = 0
            elif self.placement == 'middle':
                placement_index = rest_tokens // 2
            elif self.placement == 'last':
                placement_index = rest_tokens
            else:
                placement_index = rest_tokens // 10 * int(self.placement[1:])
            
            irrelevant_document_tokens_ = irrelevant_document_tokens[:]
            irrelevant_document_tokens_[placement_index:placement_index] = relevant_document_tokens
        
        # Handle two-document tasks (2s2d, 3s2d) - like base repo lines 374-419
        elif self.task in ["2s2d", "3s2d"]:
            relevant_document1 = row['Documents'][0]
            relevant_document2 = row['Documents'][1]
            relevant_document1_tokens = tokenization(relevant_document1)
            relevant_document2_tokens = tokenization(relevant_document2)
            relevant_document = relevant_document1 + relevant_document2
            relevant_document_tokens = relevant_document1_tokens + relevant_document2_tokens
            
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
            
            # Placement (like base repo lines 392-415)
            if self.placement == 'middle' or self.placement == 'middle-middle':
                placement_index1 = rest_tokens // 2
                placement_index2 = rest_tokens // 2
            elif self.placement == 'first' or self.placement == 'first-first':
                placement_index1 = 0
                placement_index2 = 0
            elif self.placement == 'last' or self.placement == 'last-last':
                placement_index1 = rest_tokens
                placement_index2 = rest_tokens
            elif self.placement == 'first-middle':
                placement_index1 = 0
                placement_index2 = rest_tokens // 2
            elif self.placement == 'middle-last':
                placement_index1 = rest_tokens // 2
                placement_index2 = rest_tokens
            elif self.placement == 'first-last':
                placement_index1 = 0
                placement_index2 = rest_tokens
            else:
                # Default: middle-middle
                placement_index1 = rest_tokens // 2
                placement_index2 = rest_tokens // 2
            
            irrelevant_document_tokens_ = irrelevant_document_tokens[:]
            irrelevant_document_tokens_[placement_index2:placement_index2] = relevant_document2_tokens
            irrelevant_document_tokens_[placement_index1:placement_index1] = relevant_document1_tokens
        
        # Handle three-document task (3s3d) - like base repo lines 421-469
        elif self.task in ["3s3d"]:
            relevant_document1 = row['Documents'][0]
            relevant_document2 = row['Documents'][1]
            relevant_document3 = row['Documents'][2]
            relevant_document1_tokens = tokenization(relevant_document1)
            relevant_document2_tokens = tokenization(relevant_document2)
            relevant_document3_tokens = tokenization(relevant_document3)
            relevant_document_tokens = relevant_document1_tokens + relevant_document2_tokens + relevant_document3_tokens
            
            rest_tokens = max(0, set_haystack_len - len(relevant_document_tokens) - prompt_len)
            irrelevant_document_tokens = joint_longest_irrelevant_documents_tokens[:rest_tokens]
            
            # Placement (like base repo lines 441-464)
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
            elif self.placement == 'first-middle-last':
                placement_index1 = 0
                placement_index2 = rest_tokens // 2
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
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        
        long_context_input = decode_tokens(irrelevant_document_tokens_)
        return long_context_input
    
    def create_prompt(self, question, long_context_input):
        """Create the exact prompt format used in original evaluation."""
        format_instructions = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object should not be wrapped in triple backticks.

Here is the output schema:
```
{{"properties": {{"reasoning": {{"title": "Reasoning", "description": "Solution process.", "type": "string"}}, "answer": {{"title": "Answer", "description": "The final numerical answer to the question, deduced through reasoning.", "type": "number"}}}}, "required": ["reasoning", "answer"]}}
```"""
        
        prompt = f"""Long-Context Documents:
{long_context_input}

You are tasked with solving a mathematical reasoning question using information from Long-Context Documents. Follow these steps to ensure accurate extraction and calculation:

**Instructions:**
1. **Extract Relevant Numerical Information**: Carefully read through the provided Long-Context Documents to identify and list all relevant numerical details. These could include objects, their attributes, numerical values, dates, locations, or any other quantitative data.
   
2. **Analyze and Solve the Question**: Use the identified numerical details to solve the given question. Ensure your solution involves a single computational step based on the relevant data extracted. Focus on logical or arithmetic operations as required by the question.

Question:
{question}

{format_instructions}
"""
        return prompt
    
    def invoke_model(self, prompt):
        """Invoke the model and return response with token counts."""
        start_time = time.time()
        
        if self.config['provider'] == 'bedrock':
            # Retry logic for throttling (matching your working LLMAgent code)
            max_retries = 5
            base_delay = 2
            for attempt in range(max_retries):
                try:
                    # Inference profiles (us. prefix) use converse API (required for Claude 4.5)
                    # Direct model IDs use invoke_model
                    is_inference_profile = self.config['model_id'].startswith('us.')
                    
                    if is_inference_profile:
                        # Use converse API for inference profiles (per AWS docs)
                        logger.info(f"Using converse API for inference profile: {self.config['model_id']}")
                        conversation = [
                            {
                                "role": "user",
                                "content": [{"text": prompt}]
                            }
                        ]
                        logger.debug(f"Calling converse API with {len(tokenization(prompt)):,} token prompt")
                        response = self.client.converse(
                            modelId=self.config['model_id'],
                            messages=conversation,
                            inferenceConfig={
                                "maxTokens": 16384,  # Output tokens: increased for long reasoning (was 8192)
                                "temperature": 0.0
                                # Note: Cannot use both temperature and topP for Claude 4.5
                            }
                        )
                        logger.debug(f"Converse API call successful")
                        response_text = response["output"]["message"]["content"][0]["text"]
                        usage = response.get('usage', {})
                        input_tokens = usage.get('inputTokens', 0)
                        output_tokens = usage.get('outputTokens', 0)
                    else:
                        # Use invoke_model for direct model IDs
                        # Construct body based on model family
                        if self.config['family'] == 'anthropic':
                            # Per AWS docs: content should be array with type: "text"
                            body = {
                                "anthropic_version": "bedrock-2023-05-31",
                                "max_tokens": 16384,  # Output tokens: increased for long reasoning (was incorrectly 200000)
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
                                "max_tokens": 16384,  # Output tokens: increased for long reasoning
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
                                "max_tokens": 16384,  # Output tokens: increased for long reasoning
                                "temperature": 0.0
                            }
                        
                        response_obj = self.client.invoke_model(
                            modelId=self.config['model_id'],
                            body=json.dumps(body)
                        )
                        
                        response_body = json.loads(response_obj['body'].read())
                        
                        # Parse response based on family (matching your working LLMAgent code)
                        if self.config['family'] == 'anthropic':
                            response_text = response_body['content'][0]['text']
                            usage = response_body.get('usage', {})
                            input_tokens = usage.get('input_tokens', 0)
                            output_tokens = usage.get('output_tokens', 0)
                        elif self.config['family'] in ['deepseek', 'qwen', 'openai']:
                            response_text = response_body['choices'][0]['message']['content']
                            # Try multiple possible keys for token counts
                            usage = response_body.get('usage', {})
                            if not usage:
                                # Check if tokens are at top level
                                usage = response_body
                            
                            # Try all possible token key names
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
                            
                            # If still 0, fallback to tiktoken counting
                            if input_tokens == 0:
                                input_tokens = len(tokenization(prompt))
                                logger.warning(f"Token extraction failed, using tiktoken for input: {input_tokens}")
                            if output_tokens == 0:
                                output_tokens = len(tokenization(response_text))
                                logger.warning(f"Token extraction failed, using tiktoken for output: {output_tokens}")
                    
                    break
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                    error_message = e.response.get('Error', {}).get('Message', str(e))
                    error_str = f"{error_code}: {error_message}"
                    
                    # Check for use case form requirement (specific to Claude 4.5)
                    if error_code == 'ResourceNotFoundException' and 'use case details' in error_message.lower():
                        logger.error(f"ERROR: Claude 4.5 requires Anthropic use case form to be submitted.")
                        logger.error(f"   API used: {'converse' if is_inference_profile else 'invoke_model'}")
                        logger.error(f"   Model: {self.config['model_id']}")
                        logger.error(f"   Region: {self.config['region']}")
                        logger.error(f"   Account: 970547356481")
                        logger.error(f"   Steps:")
                        logger.error(f"   1. Go to AWS Bedrock Console → Foundation models → Claude Sonnet 4.5")
                        logger.error(f"   2. Click 'Submit use case details' or 'Request access'")
                        logger.error(f"   3. Fill out and submit the form")
                        logger.error(f"   4. Wait 15+ minutes for approval")
                        logger.error(f"   Error: {error_message}")
                        raise
                    
                    # Check for throttling
                    if error_code == 'ThrottlingException' or 'Too many tokens' in error_message:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            raise
                    else:
                        # For other errors, log details and raise
                        logger.error(f"Error (API: {'converse' if is_inference_profile else 'invoke_model'}, Code: {error_code}): {error_message}")
                        raise
                        
                except Exception as e:
                    error_str = str(e)
                    error_type = type(e).__name__
                    
                    # Check for use case form requirement in generic exceptions
                    if 'use case details' in error_str.lower():
                        logger.error(f" Claude 4.5 requires Anthropic use case form to be submitted.")
                        logger.error(f"   API used: {'converse' if is_inference_profile else 'invoke_model'}")
                        logger.error(f"   Error type: {error_type}")
                        logger.error(f"   Error: {error_str}")
                        raise
                    
                    # Check for throttling
                    if 'ThrottlingException' in error_str or 'Too many tokens' in error_str:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                        else:
                            raise
                    else:
                        # For other errors, log details and raise
                        logger.error(f"Unexpected error (API: {'converse' if is_inference_profile else 'invoke_model'}, Type: {error_type}): {error_str}")
                        raise
        
        elif self.config['provider'] == 'google':
            # Google Gemini
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.generate_content(prompt)
                    response_text = response.text
                    # Approximate token counts for Gemini
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
        
        elif self.config['provider'] == 'openai':
            # OpenAI API (GPT models)
            max_retries = 5
            base_delay = 2
            for attempt in range(max_retries):
                try:
                    # GPT-5 uses max_completion_tokens instead of max_tokens
                    # GPT-5 only supports temperature=1 (default), so don't set it
                    response = self.client.chat.completions.create(
                        model=self.config['model_id'],
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=16384
                        # Note: temperature parameter removed - GPT-5 only supports default value of 1
                    )
                    
                    # Extract response text
                    response_text = response.choices[0].message.content
                    
                    # Get token counts
                    input_tokens = response.usage.prompt_tokens if response.usage else len(tokenization(prompt))
                    output_tokens = response.usage.completion_tokens if response.usage else len(tokenization(response_text))
                    
                    break
                except Exception as e:
                    error_str = str(e)
                    if 'rate_limit' in error_str.lower() or 'timeout' in error_str.lower():
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
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
        """
        Use GPT-4o to verify if two solutions are equivalent (like base repo does).
        Returns (verification_result, reasoning)
        """
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

Example:
Solution 1:
def solve():
    current_value = 45e9  # $45 billion
    projected_value = 400e9  # $400 billion
    answer = projected_value - current_value
    return answer
Answer1: 355000000000.0

Solution 2:
The current value of the AI chip market is projected to be $45 billion, and it is expected to rise to $400 billion by 2027. To find the difference, we subtract the current value from the projected value: $400 billion - $45 billion = $355 billion.
Answer2: 355.0

Output: Yes

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
    
    def evaluate_dataset(self, data_rows, document_data, output_dir):
        """Evaluate entire dataset."""
        results = []
        correct_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trajectory_dir = Path(output_dir) / f"trajectories/{self.model_name}_{self.task}_{self.haystack_len}_{timestamp}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        # Create README in trajectory directory
        readme_content = f"""# Trajectory Files for {self.model_name} on {self.task}

Generated: {datetime.now().isoformat()}
Model: {self.model_name} ({self.config['model_id']})
Provider: {self.config['provider']}
Region: {self.config.get('region', 'N/A')}

## Files Structure

For each question (numbered 000, 001, 002, etc.), you'll find:

1. `qXXX_input.json` - Question metadata and task info
2. `qXXX_prompt.txt` - The EXACT full prompt sent to the model (including all context)
3. `qXXX_trajectory.json` - Complete response details including:
   - Full model response (raw_response)
   - Parsed reasoning (reasoning, reasoning_full)
   - Predicted vs expected answer
   - Token counts (input/output)
   - Cost for this question
   - Inference time
   - Model metadata
   - Original task data
   - Documents used

4. `qXXX_error.json` - Error details (only if question failed)

## Usage

To analyze a specific question:
```bash
# See the exact prompt sent
cat q000_prompt.txt

# See metadata
cat q000_input.json | jq

# See full response and analysis
cat q000_trajectory.json | jq
```

To find all incorrect answers:
```bash
jq 'select(.correct == false) | {{idx: .question_idx, expected: .expected_answer, predicted: .predicted_answer}}' q*_trajectory.json
```

To calculate average tokens:
```bash
jq -s 'map(.input_tokens) | add / length' q*_trajectory.json
```
"""
        
        with open(trajectory_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Starting evaluation: {len(data_rows)} questions")
        logger.info(f"Trajectories: {trajectory_dir}")
        
        for idx, row in enumerate(data_rows):
            question = row["Task"]['question']
            expected_answer = row["Task"]["answer"]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {idx+1}/{len(data_rows)}")
            logger.info(f"{'='*80}")
            
            try:
                # Create context
                long_context_input = self.create_context(row, document_data)
                prompt = self.create_prompt(question, long_context_input)
                
                # Save input with full details
                input_data = {
                    'question_idx': idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'task_info': {
                        'task': self.task,
                        'haystack_len': self.haystack_len,
                        'placement': self.placement,
                        'model_name': self.model_name,
                        'model_id': self.config['model_id']
                    },
                    'prompt_length_tokens': len(tokenization(prompt)),
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(trajectory_dir / f"q{idx:03d}_input.json", 'w') as f:
                    json.dump(input_data, f, indent=2)
                
                # Save full prompt separately for easy review
                with open(trajectory_dir / f"q{idx:03d}_prompt.txt", 'w') as f:
                    f.write(prompt)
                
                # Invoke model
                response_text, input_tokens, output_tokens, inference_time = self.invoke_model(prompt)
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Parse answer
                predicted_answer, reasoning = self.parse_answer(response_text)
                
                # Numerical comparison (like base repo line 479)
                numerical_match = self.compare_answers(expected_answer, predicted_answer)
                
                # LLM verification (like base repo lines 475-477)
                llm_judge = "None"
                judge_reasoning = ""
                if self.use_llm_verification:
                    # Construct solutions like base repo (lines 475-476)
                    solution1 = row["Task"]["solution"] + "\nAnswer1: " + str(expected_answer)
                    solution2 = reasoning + "\nAnswer2: " + str(predicted_answer)
                    llm_judge, judge_reasoning = self.llm_verification(solution1, solution2)
                
                # Final decision (like base repo line 480): correct if numerical match OR llm says yes
                correct = numerical_match or ("yes" in llm_judge.lower())
                
                if correct:
                    correct_count += 1
                
                # Calculate cost
                input_cost = (input_tokens / 1_000_000) * self.config['input_price']
                output_cost = (output_tokens / 1_000_000) * self.config['output_price']
                total_cost = input_cost + output_cost
                
                result = {
                    'question_idx': idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'predicted_answer': predicted_answer,
                    'correct': correct,
                    'numerical_match': numerical_match,
                    'llm_judge': llm_judge,
                    'reasoning': reasoning[:500] if reasoning else '',
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost': total_cost,
                    'inference_time': inference_time
                }
                results.append(result)
                
                # Save complete trajectory with all details
                trajectory_data = {
                    **result,
                    'reasoning_full': reasoning,
                    'raw_response': response_text,
                    'prompt_used': prompt,  # FULL PROMPT
                    'long_context_length': len(long_context_input),
                    'judge_reasoning': judge_reasoning,  # LLM verification reasoning
                    'model_info': {
                        'model_name': self.model_name,
                        'model_id': self.config['model_id'],
                        'provider': self.config['provider'],
                        'region': self.config.get('region', 'N/A'),
                        'input_price_per_1M': self.config['input_price'],
                        'output_price_per_1M': self.config['output_price']
                    },
                    'task_metadata': {
                        'task': self.task,
                        'haystack_len': self.haystack_len,
                        'placement': self.placement
                    },
                    'timestamp': datetime.now().isoformat(),
                    'original_task_data': row.get('Task', {}),
                    'documents_used': row.get('Documents', [])
                }
                
                with open(trajectory_dir / f"q{idx:03d}_trajectory.json", 'w') as f:
                    json.dump(trajectory_data, f, indent=2)
                
                # Log result with verification details
                verification_status = ""
                if self.use_llm_verification:
                    if numerical_match and "yes" in llm_judge.lower():
                        verification_status = " [NUM+LLM]"
                    elif numerical_match:
                        verification_status = " [NUM-ONLY]"
                    elif "yes" in llm_judge.lower():
                        verification_status = " [LLM-ONLY]"
                else:
                    verification_status = " [NUM-ONLY]"
                
                logger.info(f"  [{'PASS' if correct else 'FAIL'}]{verification_status} | {input_tokens:,} in / {output_tokens:,} out | ${total_cost:.4f} | {inference_time:.1f}s")
                logger.info(f"  Expected: {expected_answer}, Predicted: {predicted_answer}")
                
                # Small delay between questions
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"  Error: {e}")
                results.append({
                    'question_idx': idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'error': str(e)
                })
                
                with open(trajectory_dir / f"q{idx:03d}_error.json", 'w') as f:
                    json.dump({'error': str(e)}, f, indent=2)
        
        # Calculate summary
        accuracy = correct_count / len(data_rows) if data_rows else 0
        total_cost = (total_input_tokens / 1_000_000) * self.config['input_price'] + \
                     (total_output_tokens / 1_000_000) * self.config['output_price']
        
        summary = {
            'model_name': self.model_name,
            'model_id': self.config['model_id'],
            'provider': self.config['provider'],
            'region': self.config.get('region', 'N/A'),
            
            'task': self.task,
            'haystack_len': self.haystack_len,
            'placement': self.placement,
            
            'total_questions': len(data_rows),
            'correct': correct_count,
            'incorrect': len(data_rows) - correct_count,
            'accuracy': accuracy,
            
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'total_cost': total_cost,
            
            'avg_input_tokens': total_input_tokens / len(data_rows) if data_rows else 0,
            'avg_output_tokens': total_output_tokens / len(data_rows) if data_rows else 0,
            'avg_cost_per_question': total_cost / len(data_rows) if data_rows else 0,
            
            'pricing': {
                'input_price_per_1M_tokens': self.config['input_price'],
                'output_price_per_1M_tokens': self.config['output_price']
            },
            
            'timestamp': timestamp,
            'trajectory_directory': str(trajectory_dir),
            'evaluation_metadata': {
                'temperature': 0.0,
                'max_tokens': 8192,
                'retry_logic': 'exponential_backoff',
                'comparison_tolerance': 1e-9,
                'llm_verification_enabled': self.use_llm_verification,
                'llm_verification_model': 'gpt-4o' if self.use_llm_verification else None
            },
            
            'results': results
        }
        
        # Save results
        output_file = Path(output_dir) / f"results_{self.model_name}_{self.task}_{self.haystack_len}_{timestamp}.json"
        save_json_file(str(output_file), summary)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(data_rows)})")
        logger.info(f"Total cost: ${total_cost:.2f}")
        logger.info(f"Avg tokens: {summary['avg_input_tokens']:.0f} in / {summary['avg_output_tokens']:.0f} out")
        logger.info(f"Results: {output_file}")
        logger.info(f"{'='*80}\n")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Multi-model MathHay evaluation')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['qwen3-32b', 'qwen3-235b'],
                       help='Models to test')
    parser.add_argument('--tasks', type=str, nargs='+',
                       default=['2ssd', '3ssd', '2s2d', '3s2d', '3s3d'],
                       help='Tasks to run (2ssd, 3ssd, 2s2d, 3s2d, 3s3d)')
    parser.add_argument('--haystack_len', type=str, default='128000',
                       help='Haystack length')
    parser.add_argument('--placement', type=str, default='middle',
                       help='Placement of relevant docs (middle, first, last, or task-specific like middle-middle for 2-doc tasks)')
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
    logger.info(f"MULTI-MODEL MATHHAY EVALUATION")
    logger.info(f"{'='*80}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Haystack length: {args.haystack_len}")
    logger.info(f"{'='*80}\n")
    
    # Run evaluation for each model and task
    all_summaries = {}
    
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            continue
        
        for task in args.tasks:
            logger.info(f"\n{'#'*80}")
            logger.info(f"# Starting: {model_name} on {task}")
            logger.info(f"{'#'*80}\n")
            
            # Load task data
            data_file = f"full_haystack_question_{task}.json"
            data_path = os.path.join(args.input_dir, data_file)
            data_rows = load_json_file(data_path)
            
            logger.info(f"Loaded {len(data_rows)} questions from {task}")
            
            # Create evaluator
            evaluator = MultiModelEvaluator(
                model_name, 
                task, 
                args.haystack_len,
                args.placement
            )
            
            # Run evaluation
            summary = evaluator.evaluate_dataset(
                data_rows,
                document_data,
                args.output_dir
            )
            
            all_summaries[f"{model_name}_{task}"] = summary
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"ALL EVALUATIONS COMPLETE")
    logger.info(f"{'='*80}\n")
    
    for key, summary in all_summaries.items():
        logger.info(f"{key}:")
        logger.info(f"  Accuracy: {summary['accuracy']:.1%}")
        logger.info(f"  Cost: ${summary['total_cost']:.2f}")
        logger.info(f"  Questions: {summary['correct']}/{summary['total_questions']}")
    
    logger.info(f"\n{'='*80}")

if __name__ == '__main__':
    main()

