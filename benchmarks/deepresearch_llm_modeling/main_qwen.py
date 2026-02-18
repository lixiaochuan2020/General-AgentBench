'''
Arguments:
    --query: research question
    --long_report: generate long report (default is short answer)
    --answer_path: output file path for the answer
    --log_dir: log directory
    --can_click: enable click action
    --config_file: use custom config file (json file path)

Logs:
- see all system messages in /logs
    - /logs/search.log: search query and results
    - /logs/observation.log: observation of the environment
    - /logs/input.log: input to the model at each turn
    - /logs/response.log: response (after format check)
    - /logs/trajectory.md: total trajectory of the agent

'''

import torch
import re, os, requests
import argparse
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
from retrieval import *
from prompt import *
import json
import traceback
LOG_DIR = "logs"
ACTIONS = ['search', 'answer', 'plan', 'scripts', 'click', 'summary']
MAX_CONTEXT_LENGTH = 40000

class LLMAgent:
    def __init__(self, config, is_long_report: bool = False, verbose: bool = True, can_click: bool = False):
        self.is_long_report = is_long_report
        

        self.client = OpenAI(
            api_key='EMPTY',
            base_url="http://babel-8-17:8000/v1"
        )

        self.model_name = self.client.models.list().data[0].id
        self.consecutive_search_cnt = 0 # number of consecutive search actions performed for each sample
        self.search_cnt = 0 # number of total search actions performed for each sample
        self.script_cnt = 0
        self.turn_id = 0
        self.config = config
        self.verbose = verbose
        self.docs = {}
        self.doc_counter = 1 # for citations
        self.can_click = can_click

        self.search_stats = []
        self.click_stats = []

    def run_llm_loop(self, prompt):
        done = False
        input = prompt

        for step in range(self.config["max_turns"]):
            try:
                self.turn_id += 1

                print(f"=====turn {self.turn_id}======")
                if self.verbose:
                    self._log_input(input)
                
                thought, action = self.query_qwen(input)
                response_with_thought = f'<think>{thought}</think>\n\n{action}'
                self._record_trajectory(input, response_with_thought)

                # execute actions (search or answer) and get observations
                done, updated_history, next_obs = self.execute_response(
                    action, self.config["num_docs"], self.config["num_docs_to_read"]
                )

                if done:
                    print("=====final response======")
                    break

                input = self._update_input(
                    input, response_with_thought, next_obs, updated_history, prompt
                )
            except Exception as e:
                print(f"Error: {e}")
                print(traceback.format_exc())
                exit()

        answer = self._compose_final_output(action[0])
        return answer

    def query_qwen(self, prompt):
        # TODO: consider how to deal with a response without any action.
        """Query Qwen with action format check. Only return the response with correct format.
        Args:
            prompt: prompt
        Returns:
            response_with_thought: response with correct format and thought process
        """
        try_time = 0

        while try_time < self.config["max_try_time"]:
            try_time += 1

            # Initialize variables
            thought = ""
            original_response = ""

            try:
                qwen_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=38400,
                )
                
                thought = qwen_response.choices[0].message.reasoning_content
                original_response = qwen_response.choices[0].message.content

            except Exception as e:
                print(f"Error: {e}")
                continue

            actions = self.postprocess_response(original_response)

            if actions is None or len(actions) == 0:
                print(f"response with wrong format!")

                # add format reminder prompt for next try
                if self.is_long_report:
                    format_reminder_prompt = report_format_reminder_prompt
                else:
                    format_reminder_prompt = short_answer_format_reminder_prompt
                prompt = prompt + format_reminder_prompt

                # if max try time reached, raise error
                if try_time == self.config["max_try_time"]:
                    raise ValueError("Failed to generate response after max try time")
            else:
                response_with_thought = f'<think>{thought}</think>\n{actions}'

                if self.verbose:
                    with open(response_log, 'a', encoding='utf-8') as f:
                        length = len(response_with_thought)
                        f.write(f"[turn={self.turn_id}]\nresponse length={length}\n{response_with_thought}\n\n\n")

                return thought, actions

    def postprocess_response(self, response):
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
        
        # no summary action involved, normal case
        if tag_counts['summary']['start'] == 0:
            # Validate tag format rules
            valid_actions = []
            for action in ACTIONS:
                start_count = tag_counts[action]['start']
                end_count = tag_counts[action]['end']
                
                # Tags must appear in pairs
                if start_count != end_count:
                    return None
                
                # If this action's tags appeared more than once, record as valid action
                if start_count >= 1:
                    valid_actions.append(action)
            
            # At least one action per response
            if len(valid_actions) < 1:
                return None
            
            ret = []
            # Extract content between valid action tags
            for action in valid_actions:
                pattern = f'<{action}>(.*?)</{action}>'
                matches = re.findall(pattern, response, re.DOTALL)
                if len(matches) > 0:
                    ret += [f'<{action}>{content.strip()}</{action}>' for content in matches]

            if len(ret) > 0:
                return ret

            return None
                
        # special case for summary action, because the content in summary contains other tags
        else: 
            # Find the first occurrence of <summary>
            start_idx = response.find('<summary>')
            # Find the last occurrence of </summary>
            end_idx = response.rfind('</summary>')
            
            if start_idx == -1 or end_idx == -1:
                return None  # No <summary> or </summary> tag found
            
            # Extract content between the first <summary> and last </summary>
            content = response[start_idx + len('<summary>'):end_idx].strip()
            return [f'<summary>{content}</summary>']

        
        return None

    def execute_response(self, response, num_docs, num_docs_to_read, do_search=True):
        """
        Args:
            response: response
            num_docs: number of documents to retrieve
            num_docs_to_read: number of documents to fully expands
            do_search: whether to perform search
        Returns:
            done: whether the task is done
            observation: list of return information of this turn
        """
        actions_and_content = self.parse_action(response)
        next_obs = ''
        done = False
        answering = False
        info_retrieval = False
        planning = False
        scripting = False
        summarizing = False
        updated_history = False

        search_queries = []
        click_queries = []
        for action, content in actions_and_content:
            if action == 'search':
                search_queries.append(content)
                info_retrieval = True
            elif action == 'click':
                click_queries.append(content)
                info_retrieval = True
            elif action == 'answer':
                answering = True
            elif action == 'plan':
                planning = True
            elif action == 'scripts':
                scripting = True
            elif action == 'summary':
                summarizing = True

        docs = []
        potential_docs = []
        if do_search and len(search_queries) > 0:    
            docs_searched, potential_docs_searched = self.search(search_queries, 
                                                                num_docs, 
                                                                num_docs_to_read,
                                                                search_log)
            docs += docs_searched
            potential_docs += potential_docs_searched
            print('searched!')
            self.search_stats.append((self.turn_id, len(search_queries)))

        if do_search and len(click_queries) > 0:
            docs_clicked = self.click(click_queries, search_log)
            docs += docs_clicked
            print('clicked!')
            self.click_stats.append((self.turn_id, len(click_queries)))

        if answering:
            done = True
        elif info_retrieval:
            self.search_cnt += 1
            self.consecutive_search_cnt += 1
            docs_text = '</information>\n\n<information>'.join(docs)
            docs_text = '<information>' + docs_text + '</information>'
            if len(potential_docs) > 0:
                potential_docs = '\n\n'.join(potential_docs)
                potential_docs = '<information> Additional documents that might be useful: \n' + potential_docs + '</information>'
                observation = f'\n\n{docs_text.strip()}\n\n{potential_docs.strip()}\n\n'
            else:
                observation = f'\n\n{docs_text.strip()}\n\n'
            next_obs = observation
        elif planning:
            self.consecutive_search_cnt = 0
        elif scripting:
            self.consecutive_search_cnt = 0
            self.script_cnt += 1
        elif summarizing:
            next_obs = 'You performed a summary action in this turn. The content of this action is ignored since your history turns information has been updated according to it.\n'
            self.consecutive_search_cnt = 0
            self.summary_cnt += 1
            self.need_summary = False
            self.summary_history = content
            updated_history = True
        else:
            raise ValueError(f"Invalid action: {action}")

        return done, updated_history, next_obs

    def parse_action(self, actions):
        """Parse the action to get the action type and content.
        Args:
            action: action, format ensured by postprocess_response
        Returns:
            action_type: action type
            content: action content
        """
        actions_and_contents = []
        for action in actions:
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
            actions_and_contents.append((action_type, content))

        return actions_and_contents

    def _record_trajectory(self, input, response):
        """Record the trajectory of the agent.
        Args:
            input: input
            response: response
        """
        with open(trajectory_log, 'a', encoding='utf-8') as f:
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"## Turn {self.turn_id} {time}\n\n")

            input_length = len(input)
            response_length = len(response)            
            
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


    def _update_input(self, input, cur_response, next_obs, updated_history, original_prompt):
        """Update the input with the history.
        Args:
            input: input
            cur_response: current response
            next_obs: next observation
            updated_history: whether update the history to agent summary
            original_prompt: original prompt for the question
        Returns:
            updated input
        """


        if updated_history:
            context = f"[Turn 1 - Turn {self.turn_id - 1}]:\n{self.summary_history}\n\n"
            context += f"[Turn {self.turn_id}]:\n{next_obs}\n\n"
            new_input = original_prompt + context
        else:
            context = f"[Turn {self.turn_id}]:\n{cur_response}\n{next_obs}\n\n"
            new_input = input + context

        # add reminder for search and final report
        if self.consecutive_search_cnt > self.config["search_reminder_turn"]:
            new_input += f'\nNote: You have performed {self.consecutive_search_cnt} search actions. Please consider update your report scripts or output the final report. If you still want to search, make sure you check history search results and DO NOT perform duplicate search.'
        if self.turn_id > self.config["final_report_reminder_turn"]:
            new_input += f'\nNote: You have performed {self.turn_id} turns. Please consider output the final report. If you still want to search, make sure you check history search results and DO NOT perform duplicate search.'
        
        input_length = len(new_input)
        if input_length > MAX_CONTEXT_LENGTH:
            self.need_summary = True
            new_input = new_input + summary_reminder_prompt

        return new_input

    def _compose_final_output(self, response):
        if '</answer>' in response:
            response = response.split('<answer>')[1].split('</answer>')[0]
            return response
        else:
            return 'did not find answer'

    def _log_input(self, input):
        """Log the input to the log file.
        Args:
            input_str: input string
        """
        with open(input_log, 'a', encoding='utf-8') as f:
            length = len(input)
            
            # Truncate long content inside <information> tags to avoid huge logs
            pattern = r'<information>(.*?)</information>'
            
            def truncate_content(match):
                """Truncate information block content if it's too long"""
                full_content = match.group(1)  # Content between <information> tags
                if len(full_content) > 100:
                    truncated_content = full_content[:100] + '...'
                    return f'<information>{truncated_content}</information>'
                else:
                    return match.group(0)  # Return original if short enough
            
            input = re.sub(pattern, truncate_content, input, flags=re.DOTALL)
            f.write(f"[turn={self.turn_id}]\n**length={length}**\n{input}\n\n\n")
                

    def search(self, queries, num_docs, num_top_docs_to_read, search_log):
        doc_text = []
        potential_docs = []
        for query in queries:
            info_retrieved = ""
            num_docs_read = 0
            documents_and_ids = query_clueweb(query, num_docs=num_docs, num_top_docs_to_read=num_top_docs_to_read, with_id=True, with_url=self.can_click)  
            for id, url, doc in documents_and_ids:
                                
                # check how many to fully expand
                if num_docs_read < num_top_docs_to_read:
                    if id not in self.docs:
                        self.docs[id] = (self.doc_counter, url)
                        self.doc_counter += 1
                    else:
                        print("warning! duplicate document retrieved")

                    info_retrieved += f"doc_id {self.docs[id][0]}:"
                    info_retrieved += "\n"
                    info_retrieved += doc
                    info_retrieved += "\n\n"

                    num_docs_read += 1

            doc_text.append(info_retrieved)

            if self.verbose:
                with open(search_log, 'a', encoding='utf-8') as f:
                    f.write(f"[turn={self.turn_id}]\n")
                    f.write(f"query:\n{query}\n\n")
                    f.write(f"info_retrieved:\n{doc_text}\n\n\n")
        return doc_text, potential_docs
        
    # TODO: add clicked documents to citations list
    def click(self, links, search_log):
        # retreives document + outlink pairs
        document_texts = []
        for url in links:
            document_text, _ = query_clueweb_url(url)
            if document_text is None:
                document_text = f"Caution! This action is not valid. The url {url} does not appear in your history. Please check your history search results and DO NOT modify any URLs."
            document_texts.append(document_text)

        if self.verbose:
            with open(search_log, 'a', encoding='utf-8') as f:
                f.write(f"[turn={self.turn_id}]\n")
                f.write(f"clicked links:\n{links}\n\n")
                f.write(f"document text:\n{document_texts}\n\n\n")
        
        return document_texts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help='Research question or query')
    parser.add_argument('--long_report', action='store_true', help='Generate long report (default is short answer)')
    parser.add_argument('--answer_path', type=str, default='answer.md', help='Output file path for the answer')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--config_file', type=str, default=None, help="Path to config file")
    parser.add_argument('--can_click', action='store_true', help='can click on links')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    question = args.query
    is_long_report = args.long_report
    answer_path = args.answer_path
    log_dir = args.log_dir
    can_click = args.can_click
    
    # # Clear logs
    # os.makedirs(LOG_DIR, exist_ok=True)
    # if os.path.exists(LOG_DIR):
    #     for file in os.listdir(LOG_DIR):
    #         os.remove(os.path.join(LOG_DIR, file))

    search_log = f"{LOG_DIR}/search.log"
    observation_log = f"{LOG_DIR}/observation.log"
    response_log = f"{LOG_DIR}/response.log"
    raw_response_log = f"{LOG_DIR}/raw_response.log"
    input_log = f"{LOG_DIR}/input.log"
    trajectory_log = f"{LOG_DIR}/trajectory.md"
    
    if is_long_report:
        prompt = report_prompt.format(question=question)
    else:
        if can_click:
            prompt = short_answer_prompt_clickable.format(question=question)
        else:
            prompt = short_answer_prompt.format(question=question)

    if args.config_file:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {"max_turns": 25, # max number of turns
              "num_docs": 1, # number of documents to retrieve
              "max_try_time": 5, # max number of tries to generate a response
              "search_reminder_turn": 5, # number of turns to remind the agent to stop search and revise the report scripts or output the final report (only for long report)
              "final_report_reminder_turn": 15, # number of turns to remind the agent to output the final report (only for long report)
              "num_docs_to_read": 1
              } 
    
    agent = LLMAgent(config,is_long_report=is_long_report, verbose=True, can_click=can_click)

    print(f"Model: {agent.model_name}")
    if is_long_report:
        print(f"Generating long report mode...")
    else:
        print(f"Generating short answer mode...")
    
    response = agent.run_llm_loop(prompt)
    
    with open(answer_path, 'w', encoding='utf-8') as f:
        f.write(f"## Question:\n{question}\n\n")
        f.write(f"## Model: \n{agent.model_name}\n\n")
        f.write(f"## Turns: \n{agent.turn_id}\n\n")
        f.write(f"## Search Count: \n{agent.search_cnt}\n\n")
        f.write(f"## Script Count: \n{agent.search_cnt}\n\n")
        f.write(f"## Answer:\n")
        f.write(response)
    print(f"Answer saved to {answer_path}!")

