'''
Arguments:
    --batch_file: path to json file containing array of id and question fields
    --long_report: generate long report (default is short answer)
    --answer_dir: result directory
    --log_dir: log directory

Logs:
- see all system messages in /logs
    - /logs/search_{question_id}.log: search query and results
    - /logs/trajectory_{question_id}.md: total trajectory of the agent

'''


import re, os, requests
import argparse
from datetime import datetime
from collections import defaultdict
import google.generativeai as generativeai
# import google.generativeai as genai
from openai import OpenAI
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
# from google import genai
import google.generativeai as genai
# from google.genai import types
# from google.genai.types import GenerateContentConfig, ThinkingConfig
from dotenv import load_dotenv
from retrieval import *
from prompt import *
import json
import traceback
import time
import random
from utils.token_calculator import tokenize
import boto3
from huggingface_hub import InferenceClient

# Load environment variables from keys.env file
load_dotenv('keys.env')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# For Bedrock, ensure the AWS_BEARER_TOKEN_BEDROCK environment variable is set.
generativeai.configure(api_key=GEMINI_API_KEY)


PROJECT_ID = "[deepresearch-llm-modeling]"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
GEMINI_MODEL_ID = "gemini-2.5-pro"
ACTIONS = ['search', 'answer', 'plan', 'scripts']
CONCURRENT_NUM = 3
MAX_CONTEXT_LENGTH = 8000

class LLMAgent:
    def __init__(self, config, log_dir: str, answer_dir: str, is_long_report: bool = False, verbose: bool = False, 
             is_qwen: bool = False, is_deepseek: bool = False, is_bedrock: bool = False, is_hf: bool = False, 
             is_openai: bool = False, openai_model: str = 'gpt-4o', 
             search_engine: str = 'clueweb', url: str = 'http://localhost:8000/v1', 
             baseline_contexts: dict = None, hf_provider: str = None): 
        self.is_long_report = is_long_report
        self.is_qwen = is_qwen
        self.is_deepseek = is_deepseek
        self.is_bedrock = is_bedrock
        self.is_hf = is_hf
        self.is_openai = is_openai
        
        if self.is_qwen or self.is_deepseek:
            api_key = 'EMPTY'
            base_url = url
            if self.is_deepseek:
                api_key = DEEPSEEK_API_KEY
                base_url = "https://api.deepseek.com/v1"
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )

            if self.is_deepseek:
                self.model_name = "deepseek-reasoner" 
            else: # Qwen
                self.model_name = self.client.models.list().data[0].id
        elif self.is_bedrock:
            # Initialize Bedrock client.
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-west-2" 
            )
            # self.model_name =  "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            # self.model_name = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
            # self.model_name =  "qwen.qwen3-235b-a22b-2507-v1:0"
            self.model_name =  "us.deepseek.r1-v1:0"
            # self.model_name =  "us.meta.llama3-3-70b-instruct-v1:0"
            # self.model_name = "openai.gpt-oss-120b-1:0"
        elif self.is_hf:
            if not HF_API_KEY:
                raise ValueError("HF_API_KEY not found in environment variables")
            
            self.model_name = url if url != 'http://localhost:8000/v1' else "meta-llama/Llama-3.1-70B-Instruct"
            self.hf_provider = hf_provider
            
            # Initialize client with provider if specified
            if self.hf_provider:
                self.client = InferenceClient(
                    provider=self.hf_provider,
                    token=HF_API_KEY
                )
                print(f"Initialized HuggingFace Inference Client with model: {self.model_name}")
                print(f"Using provider: {self.hf_provider}")
            else:
                self.client = InferenceClient(token=HF_API_KEY)
                print(f"Initialized HuggingFace Inference Client with model: {self.model_name}")
        elif self.is_openai:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = openai_model
            print(f"Initialized OpenAI Client with model: {self.model_name}")
        else: # Gemini
            self.model_name = GEMINI_MODEL_ID
            # self.client = genai.Client(api_key=GEMINI_API_KEY)
            generativeai.configure(api_key=GEMINI_API_KEY)
            self.client = generativeai.GenerativeModel(self.model_name)

        self.search_engine = search_engine
        self.consecutive_search_cnt = {} # number of consecutive search actions performed for each sample
        self.search_cnt = {} # number of total search actions performed for each sample
        self.script_cnt = {} # number of total script actions performed for each sample
        # self.summary_cnt = {} # number of total summary actions performed for each sample
        self.context_cnt = {} # number of total context length in each turn for each sample
        self.turn_id = {} # turn id for each question
        # self.summary_history = {} # history of summary actions performed for each sample
        self.need_format_reminder = {} # whether need format reminder prompt for each sample
        self.target_context_length = config.get("target_context_length")
        self.relative_scaling_factor = config.get("relative_scaling_factor")
        self.baseline_contexts = baseline_contexts or {}
        self.baseline_avg = sum(self.baseline_contexts.values()) / len(self.baseline_contexts) if self.baseline_contexts else None
        self.config = config
        self.verbose = verbose
        self.log_dir = log_dir
        self.answer_dir = answer_dir
        self.questions = {}
        print(f"#######\nInit LLMAgent with model {self.model_name}, mode: {'long' if self.is_long_report else 'short'}, search: {self.search_engine}\n#######")
        if self.target_context_length:
            print(f"Absolute scaling: target_context_length={self.target_context_length}")
        if self.relative_scaling_factor:
            avg_str = f"{self.baseline_avg:.0f}" if self.baseline_avg else "0"
            print(f"Relative scaling: factor={self.relative_scaling_factor}, baseline_avg={avg_str}")
        time.sleep(5)

    def _get_target_context(self, question_id):
        """Get target context length for a question (absolute or relative)"""
        if self.target_context_length:
            return self.target_context_length
        elif self.relative_scaling_factor:
            baseline = self.baseline_contexts.get(str(question_id), self.baseline_avg)  # Convert to string
            if baseline is None:
                return None
            return int(baseline * self.relative_scaling_factor)
        return None

    def run_llm_loop(self, prompt, question_id):
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        trajectory_log = f"{self.log_dir}/trajectory_{question_id}.md"
        trajectory_jsonl_log = f"{self.log_dir}/trajectory_{question_id}.jsonl"
        consice_trajectory_log = f"{self.log_dir}/consice_trajectory_{question_id}.md"
        search_log = f"{self.log_dir}/search_{question_id}.log"
        
        # Clear logs
        with open(trajectory_log, 'w', encoding='utf-8') as f:
            f.write('')
        if self.verbose:
            # with open(consice_trajectory_log, 'w', encoding='utf-8') as f:
            #     f.write('')
            # with open(search_log, 'w', encoding='utf-8') as f:
            #     f.write('')
            with open(trajectory_jsonl_log, 'w', encoding='utf-8') as f:
                f.write('')

        print(f"Running question {question_id}")

        done = False
        input = prompt
        self.consecutive_search_cnt[question_id] = 0
        self.search_cnt[question_id] = 0
        self.script_cnt[question_id] = 0
        # self.summary_cnt[question_id] = 0
        self.context_cnt[question_id] = []
        self.turn_id[question_id] = 0
        # self.summary_history[question_id] = ''
        self.need_format_reminder[question_id] = False
        try:
            for step in range(self.config["max_turns"]):
                self.turn_id[question_id] += 1
                print(f"=====turn {self.turn_id[question_id]}======")

                if self.is_qwen or self.is_deepseek or self.is_openai:
                    response, action = self._query_openai_compatible(input, question_id)
                elif self.is_bedrock:
                    response, action = self._query_bedrock(input, question_id)
                elif self.is_hf:
                    response, action = self._query_hf(input, question_id)
                else:
                    response, action = self._query_gemini(input, question_id, trajectory_log)
                # execute actions (search or answer) and get observations
                done, need_update_history, next_obs = self._execute_response(
                    action, self.config["num_docs"], question_id, search_log
                )
                self._record_trajectory(input, response, next_obs, trajectory_log, trajectory_jsonl_log, question_id)
                # if self.verbose:
                #     self._record_consice_trajectory(response, next_obs, consice_trajectory_log, question_id)

                if done:
                    print("=====final response======")
                    break
                input = self._update_input(
                    input, response, next_obs, question_id, need_update_history, prompt
                )
            
            answer = self._compose_final_output(action)
            self._log_result(answer=answer, question_id=question_id)
                
            print(f"Question {question_id} result saved to {self.answer_dir}/result_{question_id}.json\n")
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

    def run_llm_loop_parallel(self, prompts, questions, question_ids):
        print(f"Running {len(prompts)} questions in parallel with {CONCURRENT_NUM} workers")

        for i, question_id in enumerate(question_ids):
            question = questions[i]
            self.questions[question_id] = question

        with ThreadPoolExecutor(max_workers=CONCURRENT_NUM) as executor:
            futures = [executor.submit(self.run_llm_loop, prompt, question_id) 
                for prompt, question_id in zip(prompts, question_ids)]
            concurrent.futures.wait(futures)

    def _query_openai_compatible(self, prompt, question_id):
        """Query an OpenAI-compatible API (Qwen, DeepSeek) with action format check."""
        thought = None
        max_try_times = 3
        for _ in range(max_try_times):
            try:
                response_obj = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,
                )

                thought = getattr(response_obj.choices[0].message, 'reasoning_content', None)
                response = response_obj.choices[0].message.content
                break

            except Exception as e:
                if "context" in str(e):
                    raise ValueError(f"Context length error: {e}")
                else:
                    print(f"Error: {e}")
                    print(traceback.format_exc())
                if _ == max_try_times - 1:
                    raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")

        action = self._postprocess_response(response) # if format is not correct, action is None

        if thought is not None:
            original_response = f'<think>{thought}</think>\n{response}'
        else:
            original_response = response
        
        input_tokens = tokenize(prompt)
        output_tokens = tokenize(original_response)
        self.context_cnt[question_id].append({
            'input': input_tokens,
            'output': output_tokens,
            'total': input_tokens + output_tokens
        })

        return original_response, action

    def _query_hf(self, prompt, question_id):
        """Query HuggingFace Inference API with specific provider."""
        # INCREASED RETRIES FROM 5 TO 10
        max_try_times = 10
        response = None
        
        for attempt in range(max_try_times):
            try:
                messages = [{"role": "user", "content": prompt}]
                
                response_obj = self.client.chat_completion(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=8192,
                    temperature=0.7,
                )
                
                response = response_obj.choices[0].message.content
                break

            except Exception as e:
                error_msg = str(e)
                
                # Combine 500 (Internal Error), 502 (Bad Gateway), 503 (Service Unavailable), 504 (Timeout)
                if any(code in error_msg for code in ["500", "502", "503", "504", "Internal Server Error", "timeout"]):
                    if attempt < max_try_times - 1:
                        # Exponential backoff: 4s, 8s, 16s, 32s, 60s...
                        wait_time = min(4 * (2 ** attempt), 60)
                        # Add jitter to prevent thundering herd
                        wait_time += random.uniform(0, 2)
                        
                        print(f"Server Error ({error_msg}), waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_try_times})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ValueError(f"Failed after {max_try_times} tries: {e}")
                
                elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    if attempt < max_try_times - 1:
                        wait_time = (2 ** attempt) + random.uniform(5, 10)
                        print(f"Rate limited, waiting {wait_time:.1f}s")
                        time.sleep(wait_time)
                    else:
                        raise ValueError(f"Rate limit exceeded: {e}")
                
                elif "context" in error_msg or "length" in error_msg:
                    raise ValueError(f"Context length error: {e}")
                else:
                    print(f"Error: {e}")
                    print(traceback.format_exc())
                    if attempt == max_try_times - 1:
                        raise ValueError(f"Failed after {max_try_times} tries: {e}")
                    time.sleep(random.randint(5, 10))

        action = self._postprocess_response(response)
        
        input_tokens = tokenize(prompt)
        output_tokens = tokenize(response)
        self.context_cnt[question_id].append({
            'input': input_tokens,
            'output': output_tokens,
            'total': input_tokens + output_tokens
        })

        return response, action

    def _query_bedrock(self, input, question_id):
        if 'claude' in self.model_name.lower() or 'anthropic' in self.model_name.lower():
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 32768,
                "messages": [
                    {
                        "role": "user",
                        "content": input
                    }
                ],
                "temperature": 0.7,
            }
        elif 'deepseek' in self.model_name.lower():
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": input
                    }
                ],
                "max_tokens": 32768,
                "temperature": 0.7,
            }
        elif 'qwen' in self.model_name.lower():
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": input
                    }
                ],
            }
        elif 'gpt' in self.model_name.lower() or 'openai' in self.model_name.lower():
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": input
                    }
                ],
                "max_tokens": 32768,
                "temperature": 0.7,
            }
        elif 'llama' in self.model_name.lower():
            body = {
                "prompt": input,
                "max_gen_len": 32768,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        else:
            body = {
                "prompt": input,
            }
        
        # Retry logic for throttling
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response_obj = self.client.invoke_model(
                    modelId=self.model_name,
                    body=json.dumps(body)
                )
                break  
                
            except Exception as e:
                if 'ThrottlingException' in str(e) or 'Too many tokens' in str(e):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        raise  
                else:
                    raise 
        
        response_body = json.loads(response_obj['body'].read())
        
        if 'claude' in self.model_name.lower() or 'anthropic' in self.model_name.lower():
            response_text = response_body['content'][0]['text']
        elif 'deepseek' in self.model_name.lower() or 'qwen' in self.model_name.lower() or 'gpt' in self.model_name.lower() or 'openai' in self.model_name.lower():
            response_text = response_body['choices'][0]['message']['content']
        elif 'llama' in self.model_name.lower():
            response_text = response_body.get('generation', response_body.get('outputs', [{}])[0].get('text', ''))
        else: 
            response_text = response_body.get('generation', response_body.get('text', ''))

        if not response_text:
            raise ValueError(f"Could not extract response text. Response keys: {response_body.keys()}")
        
        action = self._postprocess_response(response_text)
        input_tokens = tokenize(input)
        output_tokens = tokenize(response_text)
        self.context_cnt[question_id].append({
            'input': input_tokens,
            'output': output_tokens,
            'total': input_tokens + output_tokens
        })
        
        return response_text, action

    # def _query_gemini(self, prompt, question_id, trajectory_log):
    #     """Query Gemini with action format check. Only return the response with correct format.
    #     Args:
    #         prompt: prompt
    #     Returns:
    #         response: response with correct format and thought process
    #     """
    #     max_try_times = 5
    #     response = None  
    #     thought = ""
    #     action = None
        
    #     for _ in range(max_try_times):
    #         try:
    #             # gemini_response = self.client.models.generate_content(
    #             #     model=self.model_name,
    #             #     contents=prompt,
    #             #     config=GenerateContentConfig(
    #             #         thinking_config=ThinkingConfig(include_thoughts=True),
    #             #         max_output_tokens=6144
    #             #     ),
    #             # )
    #             # gemini_response = self.client.generate_content(prompt)
    #             # for part in gemini_response.candidates[0].content.parts:
    #             #     if not part.text:
    #             #         continue
    #             #     if part.thought:
    #             #         thought = part.text
    #             #     else:
    #             #         response = part.text
    #             gemini_response = self.client.generate_content(prompt)
    #             response = gemini_response.text  # single combined string
    #             thought = ""
                
    #             if response is not None:
    #                 action = self._postprocess_response(response)
    #                 if action is not None: # if format is correct, break
    #                     break
                
    #         except Exception as e:
    #             if "context" in str(e):
    #                 raise ValueError(f"Context length error: {e}")
    #             else:
    #                 print(f"Error: {e}")
    #                 print(traceback.format_exc())
    #             if _ == max_try_times - 1:
    #                 raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")
    #             else:
    #                 # random sleep 1-3 seconds
    #                 time.sleep(random.randint(1, 3))
           
    #     original_response = f'<think>{thought}</think>\n{response}'

    #     input_tokens = tokenize(prompt)
    #     output_tokens = tokenize(original_response)
    #     self.context_cnt[question_id].append({
    #         'input': input_tokens,
    #         'output': output_tokens,
    #         'total': input_tokens + output_tokens
    #     })

    #     return original_response, action

    def _query_gemini(self, prompt, question_id, trajectory_log):
        """Query Gemini with action format check. Only return the response with correct format.
        Args:
            prompt: prompt
        Returns:
            response: response with correct format and thought process
        """
        max_try_times = 5
        response = None  
        thought = ""
        action = None
        
        for _ in range(max_try_times):
            try:
                gemini_response = self.client.generate_content(prompt)
                
                # Check if response has valid parts before accessing text
                if not gemini_response.candidates or not gemini_response.candidates[0].content.parts:
                    finish_reason = gemini_response.candidates[0].finish_reason if gemini_response.candidates else "UNKNOWN"
                    print(f"No valid response parts. Finish reason: {finish_reason}")
                    if _ < max_try_times - 1:
                        time.sleep(random.randint(1, 3))
                        continue
                    else:
                        raise ValueError(f"Failed to get valid response after {max_try_times} tries. Finish reason: {finish_reason}")
                
                response = gemini_response.text  # single combined string
                thought = ""
                
                if response is not None:
                    action = self._postprocess_response(response)
                    if action is not None: # if format is correct, break
                        break
                
            except Exception as e:
                if "context" in str(e):
                    raise ValueError(f"Context length error: {e}")
                else:
                    print(f"Error: {e}")
                    print(traceback.format_exc())
                if _ == max_try_times - 1:
                    raise ValueError(f"Failed to get response after {max_try_times} tries: {e}")
                else:
                    # random sleep 1-3 seconds
                    time.sleep(random.randint(1, 3))
    
        original_response = f'<think>{thought}</think>\n{response}'

        input_tokens = tokenize(prompt)
        output_tokens = tokenize(original_response)
        self.context_cnt[question_id].append({
            'input': input_tokens,
            'output': output_tokens,
            'total': input_tokens + output_tokens
        })

        return original_response, action

    def _postprocess_response(self, response):
        """Make sure the response is in the correct format.
        Args:
            response: response text
        Returns:
            processed response, if the format is not correct, return None
        """
        if response is None:
            return None
        
        # Count occurrences of each tag
        tag_counts = {}
        for action in ACTIONS:
            start_tag = f'<{action}>'
            end_tag = f'</{action}>'
            start_count = response.count(start_tag)
            end_count = response.count(end_tag)
            tag_counts[action] = {'start': start_count, 'end': end_count}
        
            
            # check for <information> or </information> tag, this should not appear in the response
        if '<information>' in response or '</information>' in response:
            return None
        
        valid_actions = []
        for action in ACTIONS:
            start_count = tag_counts[action]['start']
            end_count = tag_counts[action]['end']
            
            # Tags must appear in pairs and at most once
            if start_count != end_count or start_count > 1:
                return None
            
            # If this action's tags appeared once, record as valid action
            if start_count == 1:
                valid_actions.append(action)
        
        # Only one action is allowed per response
        if len(valid_actions) != 1:
            return None
        
        # Extract content between valid action tags
        action = valid_actions[0]
        pattern = f'<{action}>(.*?)</{action}>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return f'<{action}>{content}</{action}>'
        
        return None

    def _execute_response(self, action, num_docs, question_id, search_log, do_search=True):
        """
        Args:
            action: action to be executed, None if format is not correct
            num_docs: number of docs to retrieve
            search_log: file to log search output
            do_search: whether to perform search
        Returns:
            done: whether the task is done
            need_update_history: whether need to update the history to agent summary
            next_obs: next observation
        """
        if action is None:
            self.need_format_reminder[question_id] = True
            next_obs = 'A invalid action, cannot be executed.'
            return False, False, next_obs

        action_type, content = self._parse_action(action)
        next_obs = ''
        done = False
        need_update_history = False

        if action_type == "answer":
            target_context = self._get_target_context(question_id)
            if target_context:
                last_turn_total = self.context_cnt[question_id][-1]['total'] if self.context_cnt[question_id] else 0
                
                if last_turn_total < target_context:
                    done = False
                    next_obs = "Before finalizing your answer, take additional time to verify your reasoning, consider alternative approaches, and search for any missing information that could strengthen your response."
                else:
                    done = True
                    next_obs = f'answer generated, the process is done.'
            else:
                done = True
                next_obs = f'answer generated, the process is done.'
        elif action_type == 'search':
            self.search_cnt[question_id] += 1
            self.consecutive_search_cnt[question_id] += 1
            search_query = content
            if do_search and search_query:
                search_results = self._search(search_query, num_docs, search_log, question_id)
                observation = f'<information>{search_results}</information>'
                next_obs = observation
            else:
                next_obs = '<information></information>'  # Return empty info if no query or search disabled
        elif action_type == 'plan':
            self.consecutive_search_cnt[question_id] = 0
            next_obs = '' # No observation for plan
        elif action_type == 'scripts':
            self.consecutive_search_cnt[question_id] = 0
            self.script_cnt[question_id] += 1
            next_obs = '' # No observation for scripts
        else:
            raise ValueError(f"Invalid action: {action_type}")

        return done, need_update_history, next_obs

    def _parse_action(self, action):
        """Parse the action to get the action type and content.
        Args:
            action: action, format ensured by postprocess_response
        Returns:
            action_type: action type
            content: action content
        """
        # Find the first occurrence of '<' and '>' to extract action_type
        start_tag_open = action.find('<')
        start_tag_close = action.find('>', start_tag_open)
        if start_tag_open == -1 or start_tag_close == -1:
            raise ValueError(f"Invalid action format: {action}")
        
        action_type = action[start_tag_open + 1:start_tag_close]

        # Find the last occurrence of '</' and '>' to locate the closing tag
        end_tag_open = action.rfind('</')
        end_tag_close = action.rfind('>', end_tag_open)
        if end_tag_open == -1 or end_tag_close == -1:
            raise ValueError(f"Invalid action format: {action}")

        # Extract content between the first '>' and last '</'
        content = action[start_tag_close + 1:end_tag_open].strip()

        return action_type, content

    def _record_trajectory(self, input, response, next_obs, trajectory_log, trajectory_jsonl_log, question_id):
        """Record the trajectory of the agent.
        Args:
            input: input
            response: response
            trajectory_log: path to trajectory log file
        """
        with open(trajectory_log, 'a', encoding='utf-8') as f:
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"## Turn {self.turn_id[question_id]} {time}\n\n")

            input_length = tokenize(input)
            response_length = tokenize(response if response else "")
            
            # Create patterns for all action types and truncate long contents
            for action in ['search', 'answer', 'plan', 'scripts', 'information']:
                pattern = f'<{action}>(.*?)</{action}>'
                
                def truncate_action_content(match):
                    """Truncate action content if it's too long"""
                    full_content = match.group(1)  # Content between action tags
                    if len(full_content) > 100:
                        truncated_content = full_content[:100] + '...'
                        return f'<{action}>{truncated_content}</{action}>'
                    else:
                        return match.group(0)  # Return original if short enough
                
                input_short = re.sub(pattern, truncate_action_content, input, flags=re.DOTALL)
            
            f.write(f"### Input:\n**length={input_length}**\n{input_short}\n\n")
            f.write(f"### Response:\n**length={response_length}**\n{response}\n\n--------------------------------\n\n")

        if self.verbose:
            with open(trajectory_jsonl_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "input": input,
                    "response": response,
                    "next_obs": next_obs,
                    "context_length": {"input": input_length, "output": response_length, "total": input_length + response_length}
                }) + '\n')

    def _record_consice_trajectory(self, response, next_obs, consice_trajectory_log, question_id):
        """Record the trajectory of the agent.
        Args:
            response: response
            next_obs: next observation
            consice_trajectory_log: path to consice trajectory log file
        """
        # if trajectory_log does not exist or is empty, write the header   
        if not os.path.exists(consice_trajectory_log) or os.path.getsize(consice_trajectory_log) == 0:
            with open(consice_trajectory_log, 'w', encoding='utf-8') as f:
                f.write(f"## Question: {self.questions[question_id]}\n\n")

        with open(consice_trajectory_log, 'a', encoding='utf-8') as f:
            f.write(f"## Turn {self.turn_id[question_id]}\n\n")
            f.write(f"### Agent Output:\n{response}\n\n")
            f.write(f"### Environment Observation:\n{next_obs}\n\n--------------------------------\n\n")


    def _update_input(self, input, cur_response, next_obs, question_id, need_update_history, original_prompt):
        """Update the input with the history.
        Args:
            input: input
            cur_response: current full response
            next_obs: next observation
            need_update_history: whether update the history to agent summary
            original_prompt: original prompt for the question
        Returns:
            updated input
        """
        if self.need_format_reminder[question_id]: # there is no valid action in this turn, need format reminder prompt
            context = f"[Turn {self.turn_id[question_id]}]:\n{cur_response}\n\n"
            context += format_reminder_prompt
            new_input = input + context
            self.need_format_reminder[question_id] = False
        else:
            if need_update_history:
                context = f"[Turn 0 - Turn {self.turn_id[question_id] - 1}]:\n{self.summary_history[question_id]}\n\n"
                context += f"[Turn {self.turn_id[question_id]}]:\n{next_obs}\n\n"
                new_input = original_prompt + context
            else:
                response_for_history = re.sub(r'<think>.*?</think>\n?', '', cur_response, flags=re.DOTALL)
                context = f"[Turn {self.turn_id[question_id]}]:\n{response_for_history}\n{next_obs}\n\n"
                new_input = input + context

        # add reminder for search and final report
        # Only add reminders if test-time scaling is not active
        if not self.target_context_length and not self.relative_scaling_factor:
            if self.consecutive_search_cnt[question_id] > self.config["search_reminder_turn"]:
                new_input += f'\nNote: You have performed {self.consecutive_search_cnt[question_id]} search actions. Please consider update your report scripts or output the final report. If you still want to search, make sure you check history search results and DO NOT perform duplicate search.'
            if self.turn_id[question_id] > self.config["final_report_reminder_turn"]:
                new_input += f'\nNote: You have performed {self.turn_id[question_id] + 1} turns. Please consider output the final report. If you still want to search, make sure you check history search results and DO NOT perform duplicate search.'
        
        target_context = self._get_target_context(question_id)
        if target_context:
            last_turn_total = self.context_cnt[question_id][-1]['total'] if self.context_cnt[question_id] else 0
            if last_turn_total >= target_context:
                new_input += '\n**CRITICAL: You MUST provide your final answer immediately. Do NOT perform any more searches or planning. Use the <answer>...</answer> format NOW.**'
        
        # add summary reminder prompt if context is too long
        input_length = tokenize(new_input)
        if input_length > MAX_CONTEXT_LENGTH:
            new_input += summary_reminder_prompt

        return new_input

    def _compose_final_output(self, response):
        if response is not None and '<answer>' in response and '</answer>' in response:
            return response.split('<answer>')[1].split('</answer>')[0]
        else:
            return 'did not find answer'

    def _log_result(self, answer, question_id):
        answer_file = f"{self.answer_dir}/result_{question_id}.json"
        total_input = sum(turn['input'] for turn in self.context_cnt[question_id])
        total_output = sum(turn['output'] for turn in self.context_cnt[question_id])
        total_tokens = sum(turn['total'] for turn in self.context_cnt[question_id])
        
        with open(answer_file, 'w', encoding='utf-8') as f:
            result = {
                    "model": self.model_name,
                    "question": self.questions[question_id],
                    "answer": answer,
                    "turns": self.turn_id[question_id],
                    "search count": self.search_cnt[question_id],
                    "script count": self.script_cnt[question_id],
                    "context lengths": self.context_cnt[question_id],
                    "total_input_tokens": total_input,
                    "total_output_tokens": total_output,
                    "total_tokens": total_tokens
                }
            json.dump(result, f, indent=4)

    def _search(self, query, num_docs, search_log, question_id):
        
        if self.search_engine == 'clueweb':
            documents = query_clueweb(query, num_docs=num_docs)
        elif self.search_engine == 'tavily':
            documents = query_tavility(query)
        elif self.search_engine == 'serper':
            documents = query_serper(query)
        elif self.search_engine == 'fineweb':
            documents = query_fineweb(query, num_docs=num_docs)
        else:
            raise ValueError(f"Invalid search engine: {self.search_engine}")
        info_retrieved = "\n\n".join(documents)

        # if self.verbose:
        #     with open(search_log, 'a', encoding='utf-8') as f:
        #         f.write(f"[turn={self.turn_id[question_id]}]\n")
        #         f.write(f"query:\n{query}\n\n")
        #         f.write(f"info_retrieved:\n{info_retrieved}\n\n\n")
        return info_retrieved

def load_questions_from_file(file_path):
    """Load questions from a JSON file (structured as an array of objects with 'id' and 'question' fields)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = [item["question"] for item in data]
    ids = [item["id"] for item in data]
    return questions, ids

def filter_completed_questions(questions, ids, answer_dir):
    """Filter out questions that already have answer files"""
    filtered_questions_dict = {}
    completed_count = 0
    
    for i, question_id in enumerate(ids):
        answer_file = f"{answer_dir}/result_{question_id}.json"
        if os.path.exists(answer_file):
            completed_count += 1
        else:
            filtered_questions_dict[question_id] = questions[i]
    
    return filtered_questions_dict, completed_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_file', type=str, help='Path to json file containing array of id and question fields')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--answer_dir', type=str, default='results', help='Result directory')
    parser.add_argument('--long_report', action='store_true', help='Generate long report (default is short answer)')
    parser.add_argument('--use_critique', action='store_true', help='Use critique to guide the model to improve the answer')
    parser.add_argument('--critique_dir', type=str, default='critique/critique_results', help='Path to critique results')
    parser.add_argument('--is_qwen', action='store_true', help='Use Qwen model')
    parser.add_argument('--is_deepseek', action='store_true', help='Use DeepSeek V2 model')
    parser.add_argument('--is_bedrock', action='store_true', help='Use AWS Bedrock model')
    parser.add_argument('--is_hf', action='store_true', help='Use HuggingFace Inference API')
    parser.add_argument('--is_openai', action='store_true', help='Use official OpenAI API')
    parser.add_argument('--openai_model', type=str, default='gpt-4o', help='Specific OpenAI model name')
    parser.add_argument('--url', type=str, default='http://localhost:8000/v1', help='URL to use (for Qwen) or model name (for HuggingFace)')
    parser.add_argument('--search_engine', type=str, default='clueweb', help='Search engine to use')
    parser.add_argument('--use_explicit_thinking', action='store_true', help='Whether is is a model with internal thinking. For a not thinking model, we use explicit think prompt to guide the model to think.')
    parser.add_argument('--target_context_length', type=int, default=None, help='Target context length for test-time scaling (e.g., 8000, 16000)')
    parser.add_argument('--relative_scaling_factor', type=float, default=None, help='Relative scaling factor (e.g., 2.0, 0.5, 0.25)')
    parser.add_argument('--baseline_contexts_file', type=str, default=None, help='Path to baseline contexts JSON file')
    parser.add_argument('--hf_provider', type=str, default=None, help='HuggingFace provider to use (e.g., together, hyperbolic, aws, azure)')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    is_long_report = args.long_report
    use_critique = args.use_critique
    critique_dir = args.critique_dir
    answer_dir = args.answer_dir
    log_dir = args.log_dir
    is_qwen = args.is_qwen
    is_deepseek = args.is_deepseek
    is_bedrock = args.is_bedrock
    is_hf = args.is_hf
    is_openai = args.is_openai
    openai_model = args.openai_model
    url = args.url
    search_engine = args.search_engine
    use_explicit_thinking = args.use_explicit_thinking
    hf_provider = args.hf_provider
    # make sure answer_dir and log_dir exist
    os.makedirs(answer_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load baseline contexts if relative scaling is enabled
    baseline_contexts = {}
    if args.relative_scaling_factor:
        if not args.baseline_contexts_file:
            raise ValueError("--baseline_contexts_file required when using --relative_scaling_factor")
        with open(args.baseline_contexts_file, 'r') as f:
            baseline_contexts = json.load(f)
        print(f"Loaded {len(baseline_contexts)} baseline contexts from {args.baseline_contexts_file}")

    # Load questions from file
    questions, ids = load_questions_from_file(args.batch_file)
    total_questions = len(questions)
    print(f"Loaded {total_questions} questions from {args.batch_file}")
    
    # Filter out completed questions
    filtered_questions_dict, completed_count = filter_completed_questions(questions, ids, answer_dir)
    remaining_questions_num = len(filtered_questions_dict)
    
    print(f"Total dataset: {total_questions} questions")
    print(f"Already completed: {completed_count} questions")
    print(f"Remaining to process: {remaining_questions_num} questions")
    
    # If no questions to process, exit
    if remaining_questions_num == 0:
        print("All questions have been completed!")
        exit(0)

    if use_critique:
        print(f"Loading critique results from {critique_dir}")
        critique_results = {}
        for id in ids:
            critique_file = f"{critique_dir}/{id}.json"
            if os.path.exists(critique_file):
                with open(critique_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    critique_result = data['critique']
                critique_results[id] = critique_result
            else:
                critique_results[id] = None
                print(f"Warning: No critique result found for question {id}")
        print(f"Loaded {len(critique_results)} critique results from {critique_dir}")
    

    prompts = []
    remaining_questions = []
    remaining_ids = []
    
    for id, question in filtered_questions_dict.items():
        remaining_questions.append(question)
        remaining_ids.append(id)
        
        if is_long_report:
            prompts.append(report_prompt.format(question=question))
        else:
            if args.use_explicit_thinking:
                if use_critique:
                    prompts.append(short_answer_prompt_explicit_thinking_with_critique.format(question=question, critique=critique_results[id]))
                else:
                    prompts.append(short_answer_prompt_explicit_thinking.format(question=question))
            else:
                if use_critique:
                    prompts.append(short_answer_prompt_internal_thinking_with_critique.format(question=question, critique=critique_results[id]))
                else:
                    prompts.append(short_answer_prompt_internal_thinking.format(question=question))

    config = {"max_turns": 100,
              "num_docs": 3,
              "search_reminder_turn": 5,
              "final_report_reminder_turn": 15,
              "target_context_length": args.target_context_length,
              "relative_scaling_factor": args.relative_scaling_factor
              } 
    
    agent = LLMAgent(config, log_dir=log_dir, answer_dir=answer_dir, 
                     is_long_report=is_long_report, verbose=True, 
                     is_qwen=is_qwen, is_deepseek=is_deepseek, 
                     is_bedrock=is_bedrock, is_hf=is_hf, is_openai=is_openai, openai_model=openai_model,
                     search_engine=search_engine, url=url, 
                     baseline_contexts=baseline_contexts,
                     hf_provider=hf_provider)

    if is_long_report:
        print(f"Generating long report mode...")
    else:
        print(f"Generating short answer mode...")

    agent.run_llm_loop_parallel(prompts, remaining_questions, remaining_ids)
