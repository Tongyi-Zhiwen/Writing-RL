import re
import random
import time
import json
import os
import threading

import openai
import dashscope

from verl.utils.reward_score.judge_prompts import *

def extract_json(response):
    """
    extract dict object from the response
    """
    if response is None or response == "":
        return None
    
    response_dict = None
    try:
        response_dict = json.loads(response.strip('json|```').strip())
    except Exception as e:
        try:
            response_dict = eval(response.strip('json|```').strip())
        except:
            pass

    if response_dict is None:
        pattern = r'(?:```json|```|json)\s*({.*?})\s*(?:```|json|```json)'
        matches = re.finditer(pattern, response, re.DOTALL)
        
        for match in matches:
            json_str = match.group(1)
            try:
                response_dict = json.loads(json_str)
                break
            except json.JSONDecodeError:
                continue
    
    if response_dict is None:
        for match in re.findall(r'\{[^{}]*\}', response):
            try:
                response_dict = json.loads(match)
            except json.JSONDecodeError:
                continue
        
    return response_dict

def extract_judge_verdict(response_text):
    """
    extract result from response
    """
    pattern = r'\[\[(A|B|C)\]\]'
    match = re.search(pattern, response_text)
    
    if match:
        return match.group(0)
    else:
        return None

def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    ### try to remove think when think ends
    if "</think>" in solution_str:
        solution_str = solution_str.split("</think>")[1].strip()

    return solution_str

#### API calling function
def dashscope_api_call(messages: list, model = "qwen-plus"): 
    """
    call dashscope api for comparative judge model, like qwen-plus
    """
    retry_cnt = 0
    while retry_cnt < 3:
        try:
            response = dashscope.Generation.call(
                model=model,
                messages=messages,
                result_format='message',
                stream=False,
                api_key=os.environ["DASHSCOPE_API_KEY"],
                headers={'X-DashScope-DataInspection': 'disable'},
                temperature=0.1,
            )
            return response['output']['choices'][0]['message']['content']
        except Exception as e:
            retry_cnt += 1

            time.sleep(5)
            print("FAILED!",e)
            continue

    return None

def openai_api_call(messages: list, model: str = "gpt-4o"):
    """
    Call OpenAI API for chat models like gpt-4o
    """
    retry_cnt = 0
    while retry_cnt < 3:
        try:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            retry_cnt += 1
            time.sleep(5)
            print("FAILED!", e)
            continue
    
    return None

def map_result(eval_response):
    if "A" in eval_response:
        return 0 ## response_A better
    elif "B" in eval_response:
        return 1 ## response_B better
    elif "C" in eval_response:
        return 2 ## tie
    else:
        return 3 ## unknown
    
def extract_info(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def llm_judge(prompt, reference, response, criteria=None, judge_template=None):
    if judge_template is None: ### when no specified judge template
        if criteria is None:
            judge_template = JUDGE_TEMPLATE_DEFAULT
        else:
            judge_template = JUDGE_TEMPLATE_CRITERIA

    judge_model = os.environ["JUDGE_MODEL"]
    if "qwen" in judge_model:
        api_function = dashscope_api_call
    elif "gpt" in judge_model:
        api_function = openai_api_call
    else:
        api_function = dashscope_api_call ## default
    
    if criteria is None: ### when no criteria provided, use default judge prompt
        prompt = judge_template.format(question=prompt, answer_a=reference, answer_b=response)
    else:  ### when criteria provided, use the provided criteria
        prompt = judge_template.format(question=prompt, answer_a=reference, answer_b=response, criteria=criteria)

    eval_response = api_function([{"role": "user", "content": prompt}], model=judge_model)
    if eval_response is not None:
        eval_result = extract_judge_verdict(eval_response)
        if eval_result is None:
            return eval_response, map_result(eval_response)
        else:
            return eval_response, map_result(eval_result)
    else:
        return None, 3

# Global lock for file access
file_lock = threading.Lock()

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for creative writing task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing prompt and target(reference)
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    prompt = ground_truth['prompt']
    reference = ground_truth['target']

    #### NOTE: when reference is a string, it is directly used as reference answer for comparison 
    #### NOTE: when reference is a list (Dynamic Curriculum Scheduling), it traces index_file for the correct index of the current reference
    reference_is_list = False
    if isinstance(reference, list): ### assume reference is a list of strings from easy to difficult
        reference_is_list = True
        data_id = ground_truth['id'] ### must include id in ground_truth
        experiment_name = os.environ["EXPERIMENT_NAME"]
        index_file = f"running_file/index_{experiment_name}.json" ### run-time log file 

        with file_lock:
            with open(index_file, "r") as f:
                index_dict = json.loads(f.read())
                f.close()
        
        chosen_index = min(index_dict[data_id], len(reference)-1)
        reference = reference[chosen_index] ### get the current reference

    criteria = None
    use_criteria = False
    if "criteria" in ground_truth and ground_truth["criteria"]: 
        criteria = ground_truth["criteria"]
        use_criteria = True

    do_print = random.randint(1, 32) == 1
    
    passage = extract_solution(solution_str)
    
    if do_print:
        print(f"--------------------------------")
        if use_criteria:
            print(f"Criteria: {criteria}")
        print(f"Prompt: {prompt}")
        print(f"Extracted passage: {passage}")
        print(f"Solution string: {solution_str}")
        print(f"--------------------------------")

    if passage is None: ### format incorrect
        if do_print:
            print(f"[WARNING] No equation found")
        return 0
    
    final_score = 0
    if os.environ["REWARD_STRATEGY"] == "position-advantage":
        ### evaluation choice 1: favor response, judge(response, reference)
        eval_response, result = llm_judge(prompt, passage, reference, criteria)
        
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            final_score = format_score
        else:   
            if result == 0:
                final_score = score + format_score
            elif result == 2:
                final_score = (score + format_score) / 2
            else:
                final_score = format_score
        
        #### Dynamic Curriculum Scheduling
        if reference_is_list and final_score >= score: ### NOTE: only update when the response is better
            data_id = ground_truth['id']
            experiment_name = os.environ["EXPERIMENT_NAME"]
            index_file = f"running_file/index_{experiment_name}.json"
            with file_lock:
                with open(index_file, "r") as f:
                    index_dict = json.loads(f.read())
                    f.close()
                index_dict[data_id] += 1 #### proceed to the next better reference

                with open(index_file, "w") as f:
                    f.write(json.dumps(index_dict, indent=2))
                    f.close()

    elif os.environ["REWARD_STRATEGY"] == "position-disadvantage":
        ### evaluation choice 2: favor reference, judge(reference, response)
        eval_response, result = llm_judge(prompt, reference, passage, criteria)
        
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            final_score += format_score
        else:   
            if result == 1:
                final_score = score + format_score
            elif result == 2:
                final_score = (score + format_score) / 2
            else:
                final_score = format_score
        
        #### Dynamic Curriculum Scheduling
        if reference_is_list and final_score >= score: ### NOTE: only update when the response is better
            data_id = ground_truth['id']
            experiment_name = os.environ["EXPERIMENT_NAME"]
            index_file = f"running_file/index_{experiment_name}.json"
            with file_lock:
                with open(index_file, "r") as f:
                    index_dict = json.loads(f.read())
                    f.close()
                index_dict[data_id] += 1 #### proceed to the next better reference

                with open(index_file, "w") as f:
                    f.write(json.dumps(index_dict, indent=2))
                    f.close()

    else:
        raise Exception("Unknown Reward Strategy")
    
    return final_score