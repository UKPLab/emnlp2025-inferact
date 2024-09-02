from .base_evaluator import BaseEvaluator
from .utils_eval import get_action_chain_webshop, get_action_chain_hotpotqa, get_action_chain_alfworld,extract_price, get_risk_level
from langchain.schema import HumanMessage
from prompts.webshop_prompt import web_k_task_inference, web_task_validator, web_task_validator_risk_sen, web_task_validator_multilabel, web_task_validator_multiclass, web_task_validator_true_false, web_task_validator_desc_term, web_task_validator_desc_term_logits
from prompts.hotpotqa_prompt import hotpot_k_task_inference, hotpot_task_validator
from prompts.alfworld_prompt import alfworld_k_task_inference, alfworld_task_validator, alfworld_task_validator_risk_sen
import re
import ipdb
from torch.nn.functional import softmax
import numpy as np
import torch
import json

class InferAct(BaseEvaluator):
    def __init__(self, task, **kwargs) -> None:
        super().__init__(task, **kwargs)
        self.prompts = {
            "webshop": {
                "verbalized_prob": web_task_validator,
                "logits_multilabel": web_task_validator_multilabel,
                "logits_multiclass": web_task_validator_multiclass,
                "logits_tf": web_task_validator_true_false,
                "verbalized_desc_term": web_task_validator_desc_term,
                "verbalized_desc_term_logits": web_task_validator_desc_term_logits
            },
            "hotpotqa": {
                "verbalized_prob": hotpot_task_validator,
                "logits": hotpot_task_validator
            },
            "alfworld": {
                "verbalized_prob": alfworld_task_validator,
                "logits": alfworld_task_validator
            }
        }
        # self.likelihood2num = {'Very Likely': 5, 'Likely': 4, 'Possible': 3, 'Unlikely': 2, 'Very Unlikely': 1}
        # self.likelihood2num = {'L5': 5, 'L4': 4, 'L3': 3, 'L2': 2, 'L1': 1}
        self.likelihood2num = {'e': 5, 'd': 4, 'c': 3, 'b': 2, 'a': 1}
        self.num2likelihood = {v: k for k, v in self.likelihood2num.items()}


    def task_inference(self, task_inference_prompt, seq_actions, num_tasks="three"):
        task_inference_prompt = task_inference_prompt.format(
            action=seq_actions, num_tasks=num_tasks
        )
        if self.task == "webshop":
            split_phrase = f"The {num_tasks} most likely user's instructions are:"
        elif self.task == "hotpotqa":
            split_phrase = f"The {num_tasks} most likely questions are:"
        elif self.task == "alfworld":
            split_phrase = f"The {num_tasks} most likely tasks are:"

        inferred_tasks = self.base_model([HumanMessage(content=task_inference_prompt)])

        try:
            splitted = inferred_tasks.split("The reason is")
            inferred_tasks = splitted[0].split(split_phrase)[1].strip().split("\n")
            # the inferred tasks: A. task1\nB. task2\nC. task3\n
            inferred_tasks_dic = {
                q.split(".", 1)[0].strip(): q.split(".", 1)[1].strip()
                for q in inferred_tasks
                if q.strip()
            }
            reason = splitted[1].strip()
        except:
            inferred_tasks_dic = {}
            reason = ""
        return inferred_tasks_dic, reason
    

    def truncate_response(self, token_list, token_logits):
        try:
            cut_index = token_list.index('Just')
            if token_list[cut_index + 1] == 'ification':
                return token_list[:cut_index], token_logits[:cut_index]
            else:
                return token_list, token_logits
        except ValueError:
            return token_list, token_logits



    def task_validator_verbalized_prob(self, validator_prompt, prob_estimation, chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks):
        
        task_validator = validator_prompt.format(
            action=chain_action, instructions=tasks_string, num=len(tasks)
        )
        task_validator_revserse = validator_prompt.format(
            action=chain_action, instructions=task_string_reverse, num=len(tasks)
        )
        try:
            # maybe order senstitive: swap the instrction1 and instruction2
            result = self.base_model([HumanMessage(content=task_validator)])
            result_reverse = self.base_model([HumanMessage(content=task_validator_revserse)])
            # tokens, token_logits = self.truncate_response(tokens, token_logits)
            # result, tokens, token_logits = self.base_model([HumanMessage(content=task_validator)], probs=True)
            # tokens, token_logits = self.truncate_response(tokens, token_logits)

            # result_reverse, tokens_reverse, token_logits_reverse = self.base_model(
            #     [HumanMessage(content=task_validator_revserse)], probs=True)
            
            # tokens_reverse, token_logits_reverse = self.truncate_response(tokens_reverse, token_logits_reverse)
            
            splitted = result.split("Justification:")
            candidates = splitted[0].strip()
            if len(splitted) == 1:
                answer_justification = "N/A"
            else:
                answer_justification = splitted[1].strip()
            # ipdb.set_trace()
            
            splitted_reverse = result_reverse.split("Justification:")
            candidates_reverse = splitted_reverse[0].strip()
            if len(splitted_reverse) == 1:
                answer_justification_reverse = "N/A"
            else:
                answer_justification_reverse = splitted_reverse[1].strip()

            # extract probablity and the reason
            if self.task != "alfworld":
                compiler = re.compile(r"G\d+: ([A-Z])\.?\s?\n?P\d+: ([\w\s]+|\d+\.\d+|L[1-5])(?=\s|$|\n)")
            else:
                compiler = re.compile(r"P\_([A-Z])\: ([\w\s]+|\d+\.\d+)")
                
          
            matches = compiler.findall(candidates)
            candidates = [(g, float(p)) if prob_estimation == 'verbalized_prob' else (g, p) for g, p in matches]

            matches_reverse = re.findall(compiler, candidates_reverse)
            candidates_reverse = [(g, float(p)) if prob_estimation == 'verbalized_prob' else (g, p) for g, p in matches_reverse]

            aggregated_probs = self.aggregate_probability(
                candidates, candidates_reverse, prob_estimation
            )
            # get logits
            # classification_type = [' Improbable', ' Unlikely', ' Possible', ' Likely', ' Certain']
            # classification_type = [' a', ' b', ' c', ' d', ' e']
            # # ipdb.set_trace()
            # gold_option_logits = self.multi_label_logits(token_logits, tokens, gold_label, classification_type)
            # gold_option_probs = softmax(torch.Tensor([gold_option_logits[k] for k in classification_type]))
            # # equal to the candidate
            # assert (gold_label, f"{self.num2likelihood[np.argmax(gold_option_probs.numpy())+1]}") in candidates

            # gold_option_logits_reverse = self.multi_label_logits(token_logits_reverse, tokens_reverse, self.reverse_mapping[gold_label], classification_type)
            # gold_option_probs_reverse = softmax(torch.Tensor([gold_option_logits_reverse[k] for k in classification_type]))
            # assert (self.reverse_mapping[gold_label], f"{self.num2likelihood[np.argmax(gold_option_probs_reverse.numpy())+1]}") in candidates_reverse

            # gold_option_prob_avg = (gold_option_probs + gold_option_probs_reverse)/2


            result = {
                "candidates": candidates,
                "candidates_reverse": candidates_reverse,
                "answer_justification": answer_justification,
                "answer_justification_reverse": answer_justification_reverse,
                "inferred_tasks_reason": inferred_tasks_reason,
                "tasks_string": tasks_string,
                "task_string_reverse": task_string_reverse,
                "gold_option": gold_label,
                "aggregated_probs": aggregated_probs,                
                # "gold_option_prob_avg": dict(zip(classification_type, gold_option_prob_avg.numpy().tolist()))
            }

        except Exception as e:
            print('error', e)
            result = {
                "candidates": result,
                "candidates_reverse": result_reverse,
                "answer_justification": "N/A",
                "answer_justification_reverse": "N/A",
                "inferred_tasks_reason": "N/A",
                "tasks_string": tasks_string,
                "task_string_reverse": task_string_reverse,
                "gold_option": gold_label,
                "aggregated_probs": "N/A",
                "gold_option_prob_avg": "N/A"}
        return result

    def multi_label_logits(self, token_logits, tokens, gold_label, classification_type):
        option_logits = {}
        cur_option = ''
        ix = 0
        for token, tok_logits in zip(tokens, token_logits):
            if token in ['A', 'B', 'C', 'D', 'E', ' A',' B',' C',' D',' E']:
                cur_option = token
            if token in classification_type and tokens[ix-1] == ':':
                option_logits[cur_option] = tok_logits
                cur_option = ''
            ix += 1
        # print('option_logits', option_logits)
        gold_option_logits = option_logits[f' {gold_label}'] if f' {gold_label}' in option_logits else option_logits[gold_label]
        return gold_option_logits

    def multi_class_logits(self, token_logits, tokens):
        all_options, all_options_logits = [], []

        for token, tok_logits in zip(tokens, token_logits):
            if token in [' A',' B',' C',' D',' E']:
                # get the logit of each option
                for tok in tok_logits:
                    if tok in [' A',' B',' C',' D',' E']:
                        all_options.append(tok)
                        all_options_logits.append(tok_logits[tok])
                break
        return all_options, all_options_logits
                

    def task_validator_logits(self, validator_prompt, prob_estimation, chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks):
        
        task_validator = validator_prompt.format(
            action=chain_action, instructions=tasks_string, gold_instruction=gold_label
        )
        task_validator_revserse = validator_prompt.format(
            action=chain_action, instructions=task_string_reverse, gold_instruction=self.reverse_mapping[gold_label]
        )

        # maybe order senstitive: swap the instrction1 and instruction2
        output, tokens, token_logits = self.base_model([HumanMessage(content=task_validator)], probs=True)

        output_reverse, tokens_reverse, token_logits_reverse = self.base_model(
            [HumanMessage(content=task_validator_revserse)], probs=True
        )

        try:
            if prob_estimation in ['logits_multilabel', 'logitis_likelihood']:
                # get the probability of all options from the token_logits and do softmax
                # A: Yes B: No ... each token has a list of logits. dict option: logits
                if prob_estimation == 'logits_multilabel':
                    classification_type = [' Yes', ' No']
                else:
                    classification_type = [' L1', ' L2', ' L3', ' L4', ' L5']
                gold_option_logits = self.multi_label_logits(token_logits, tokens, gold_label, classification_type)
                gold_option_logits_reverse = self.multi_label_logits(token_logits_reverse, tokens_reverse, self.reverse_mapping[gold_label], classification_type)
                # get softmax
                
                gold_option_probs = softmax(torch.Tensor(list(gold_option_logits.values())))

                # #reverse
                # logits_reverse = token_logits_reverse['Yes'] if 'Yes' in token_logits_reverse else token_logits_reverse['No']
                gold_option_probs_reverse = softmax(torch.Tensor(list(gold_option_logits_reverse.values())))

                aggregated_probs = (gold_option_probs + gold_option_probs_reverse)/2
                output = 'Yes' if np.argmax(aggregated_probs) == 0 else 'No'

                
            elif prob_estimation == 'logits_multiclass':
                all_options, all_options_logits = self.multi_class_logits(token_logits, tokens)
                all_options_reverse, all_options_logits_reverse = self.multi_class_logits(token_logits_reverse, tokens_reverse)
                all_options_probs = softmax(torch.Tensor(all_options_logits))
                # dict
                option2prob = dict(zip(all_options, all_options_probs))
                print('option2prob', option2prob)
                gold_option_probs = option2prob[f" {gold_label}"]

                all_options_probs_reverse = softmax(torch.Tensor(all_options_logits_reverse))
                option2prob_reverse = dict(zip(all_options_reverse, all_options_probs_reverse))
                gold_option_probs_reverse = option2prob_reverse[f" {self.reverse_mapping[gold_label]}"]
                print('option2prob_reverse', option2prob_reverse)

                aggregated_probs = (gold_option_probs + gold_option_probs_reverse)/2
                output = 'Yes' if aggregated_probs > 0.5 else 'No'

            elif prob_estimation == 'logits_tf':
                
                gold_option_probs = softmax(torch.Tensor([token_logits[0]["A"], token_logits[0]["B"]]))
                gold_option_probs_reverse = softmax(torch.Tensor([token_logits_reverse[0]["A"], token_logits_reverse[0]["B"]]))
                aggregated_probs = (gold_option_probs + gold_option_probs_reverse)/2
                output = 'Yes' if np.argmax(aggregated_probs) == 0 else 'No'



        except Exception as e:
            print('error', e)
            return {
                "inferred_tasks_reason": inferred_tasks_reason,
                "tasks_string": tasks_string,
                "task_string_reverse": task_string_reverse,
                "gold_option": gold_label,
                "output": "N/A",
                "output_reverse": "N/A",
                "prob": "N/A",
                "prob_reverse": "N/A",
                "aggregated_probs": "N/A"
            }
        
            
        ## aggerate the probability: multiply or average?
        # aggregated_probs = gold_label_prob * gold_label_prob_reverse

        # aggregated_probs = gold_label_prob.item()

        result = {
                "inferred_tasks_reason": inferred_tasks_reason,
                "tasks_string": tasks_string,
                "task_string_reverse": task_string_reverse,
                "gold_option": gold_label,
                "output": output,
                "output_reverse": output_reverse,
                "prob": gold_option_probs.numpy().tolist(),
                "prob_reverse": gold_option_probs_reverse.numpy().tolist(),
                "aggregated_probs": aggregated_probs.numpy().tolist()
            }
        # exit()  
        return result
    

    def task_validator(self, prob_estimation,  chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks):
        if 'verbalized' in prob_estimation:
            result = self.task_validator_verbalized_prob(self.prompts[self.task][prob_estimation], prob_estimation, chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks)
        
        elif 'logits' in prob_estimation:
            result = self.task_validator_logits(self.prompts[self.task][prob_estimation], prob_estimation, chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks)
        
        return result

    def aggregate_probability(self, probs, probs_reverse, prob_estimation = 'verbalized_prob'):
       

        if prob_estimation == 'verbalized_desc_term':
            option2prob = {option[0]: self.likelihood2num[option[1]] for option in probs}
            probs_reverse = [(option[0], self.likelihood2num[option[1]]) for option in probs_reverse]

        elif prob_estimation == 'verbalized_desc_term_logits':
            option2prob = {option[0]: self.likelihood2num[option[1]] for option in probs}
            probs_reverse = [(option[0], self.likelihood2num[option[1]]) for option in probs_reverse]
            
        else:    
            option2prob = {option[0]: option[1] for option in probs}

        for reversed_option in probs_reverse:
            order = self.reverse_mapping[reversed_option[0]]
            option2prob[order] = (option2prob.get(order, 0) + reversed_option[1]) / 2

        # sort the option by probability
        sorted_option = sorted(option2prob.items(), key=lambda x: x[1], reverse=True)
        return sorted_option


    def evaluate(self, message, **kwargs):

        if self.task.lower() == "webshop":
            action_chain, original_task = get_action_chain_webshop(message)
            task_inference_prompt = web_k_task_inference
            # if kwargs.get("risk_mode", False):
            #     validator_prompt = web_task_validator_risk_sen
            # else:
            #     validator_prompt = web_task_validator

        elif self.task.lower() == "hotpotqa":
            action_chain, original_task = get_action_chain_hotpotqa(message)
            task_inference_prompt = hotpot_k_task_inference
            # validator_prompt = hotpot_task_validator

        elif self.task.lower() == "alfworld":
            action_chain, original_task = get_action_chain_alfworld(message)
            task_inference_prompt = alfworld_k_task_inference
            if kwargs.get("risk_mode", False):
                validator_prompt = alfworld_task_validator_risk_sen
            else:
                validator_prompt = alfworld_task_validator

        if kwargs['risk_mode']:
            risk_level = get_risk_level(self.task, action_chain)
            if risk_level != "high_risk":
                return None

        results = {}
        for ix, chain in enumerate(action_chain):
            chain_str = "\n".join(chain)
            if "inferred_tasks" not in kwargs:
                try:
                    inferred_tasks_dic, inferred_tasks_reason = self.task_inference(
                        task_inference_prompt, chain_str)
                except:
                    ipdb.set_trace()
                    results[ix] = {
                        "candidates": "N/A",
                        "candidates_reverse": "N/A",
                        "answer_justification": "N/A",
                        "answer_justification_reverse": "N/A",
                        "inferred_tasks_reason": "N/A",
                        "tasks_string": "N/A",
                        "task_string_reverse": "N/A",
                        "gold_option": "N/A",
                        "aggregated_probs": "N/A",
                        "input_messages": chain,
                    }
                    continue

                tasks = list(inferred_tasks_dic.values())
                tasks.append(original_task)

                # add none of the above
                # if self.task != "alfworld":
                #     tasks.append("None of the above")

                tasks_string = "\n".join(
                    [
                        f"{self.label_map[ix+1]}. {q.strip()}"
                        for ix, q in enumerate(tasks)
                    ]
                )
                # order sensitive
                task_string_reverse = "\n".join(
                    [
                        f"{self.label_map[ix+1]}. {q.strip()}"
                        for ix, q in enumerate(tasks[::-1])
                    ]
                )
            else:
                tasks_string, task_string_reverse = kwargs["inferred_tasks"]
                tasks = tasks_string.split("\n")
                inferred_tasks_reason = "N/A"
            # if len(tasks) != 4:
            #     print('length of tasks is not 4')
            if "None of the above" in tasks[-1]:
                gold_label = self.label_map[len(tasks) - 1]
            else:
                gold_label = self.label_map[len(tasks)]

            results[ix] = self.task_validator(kwargs['prob_estimation'], chain_str, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks)
            results[ix]["input_messages"] = chain
        
        result_dic = {"real-time eval": results, "input_task": original_task, "env_name": kwargs.get("env_name", "")}

        if self.task == "webshop":
            result_dic["selected_product_price"] = extract_price(action_chain[0])
        return result_dic
        
    @classmethod
    def metric(self, json_objects, **kwargs):

        y_true, y_pred = [], []
        tp_tasks, predictions = [], []
        env_neg, env_pos = [], []
        fn_envs = []
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        cost = 0
        for obj in json_objects:
            y_pred.append(0)
            # update the y_pred as it might has multiple actions
            try:
                for ix in obj['real-time eval']:
                    if obj['real-time eval'][ix]["aggregated_probs"] == "N/A":
                        pred_label = "N/A"
                        break
                    if kwargs['prob_estimation'] == 'verbalized_prob':
                        aggregated_probs = {opt[0]: opt[1] for opt in obj['real-time eval'][ix]["aggregated_probs"]}
                        try:
                            gold_option_prob = aggregated_probs.get(obj['real-time eval'][ix]["gold_option"], 0)
                            pos_prob = round(1 - gold_option_prob, 3)
                        except:
                            ipdb.set_trace()
                        y_pred[-1] = pos_prob
                        if gold_option_prob > float(kwargs['threshold']):
                            pred_label = "Correct"
                        else:
                            pred_label = "Incorrect"
                            break

                    elif kwargs['prob_estimation'] in ['verbalized_desc_term']:
                        aggregated_probs = {opt[0]: opt[1] for opt in obj['real-time eval'][ix]["aggregated_probs"]}
                        gold_option_prob = aggregated_probs.get(obj['real-time eval'][ix]["gold_option"], 0)
                        
                        if round(float(gold_option_prob)) == 5:
                            pred_label = "Correct"
                        else:
                            pred_label = "Incorrect"
                            break
                    elif kwargs['prob_estimation'] == 'verbalized_desc_term_logits':
                        probs = sorted(obj['real-time eval'][ix]['gold_option_prob_avg'].items(), key=lambda x: x[1], reverse=True)
                        gold_option_prob = obj['real-time eval'][ix]['gold_option_prob_avg']
                        if ' e' in gold_option_prob:
                            y_pred[-1] = 1 - float(obj['real-time eval'][ix]['gold_option_prob_avg'][' e'])
                        else:
                            y_pred[-1] = 1
                        
                        if probs[0][0] == ' e':
                            pred_label = "Correct"
                        else:
                            pred_label = "Incorrect"
                            break

                    
                    elif kwargs['prob_estimation'] in ['logits_multilabel', 'logits_tf']:
                    
                        # ipdb.set_trace()
                        pred_label = "Yes" if np.argmax(obj['real-time eval'][ix]['aggregated_probs']) == 0 else "No"
                        y_pred[-1] = obj['real-time eval'][ix]['aggregated_probs'][1]

                        if pred_label == "Yes":
                            pred_label = "Correct"
                        else:
                            pred_label = "Incorrect"

                    elif kwargs['prob_estimation'] == 'logits_multiclass':
                        pred_label = obj['real-time eval'][ix]['output']
                        y_pred[-1] = 1-obj['real-time eval'][ix]['aggregated_probs']
                

            except:
                ipdb.set_trace()

            if obj["trace_correct"]:
                y_true.append(0)
                env_neg.append(obj["env_name"])
                if pred_label == "Correct":
                    true_negative += 1
                else:
                    false_positive += 1
                    predictions.append({"env_name": obj["env_name"], "true_label": "Correct"})
                
            else:
                y_true.append(1)
                env_pos.append(obj["env_name"])
                if pred_label == "Incorrect":
                    true_positive += 1
                    predictions.append({"env_name": obj["env_name"], "true_label": "Incorrect"})
                    try:
                        tp_tasks.append(obj["input_task"])
                    except:
                        continue
                else:
                    fn_envs.append(obj["env_name"])
                    if kwargs['task'] == 'webshop':
                        try:
                            cost += float(obj["selected_product_price"][0])
                        except:
                            pass
                    else:
                        cost += 1
                    false_negative += 1
        print('cost', cost)


        return {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "y_true": y_true,
            "y_pred": y_pred,
            "false_negative_env": fn_envs,
            "fn_cost": cost
        }, tp_tasks, predictions
