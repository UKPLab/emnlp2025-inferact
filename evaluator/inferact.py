from .base_evaluator import BaseEvaluator
from .utils_eval import get_action_chain_webshop, get_action_chain_hotpotqa, get_action_chain_alfworld,extract_price, get_risk_level
from langchain.schema import HumanMessage
from prompts.webshop_prompt import web_k_task_inference, web_task_validator, web_task_validator_risk_sen 
from prompts.hotpotqa_prompt import hotpot_k_task_inference, hotpot_task_validator
from prompts.alfworld_prompt import alfworld_k_task_inference, alfworld_task_validator, alfworld_task_validator_risk_sen
import re
import ipdb
import numpy as np
from torch.nn.functional import softmax

class InferAct(BaseEvaluator):
    def __init__(self, task, **kwargs) -> None:
        super().__init__(task, **kwargs)
    
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
    

    def task_validator_verbalized_prob(self, validator_prompt, chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks):
        
        task_validator = validator_prompt.format(
            action=chain_action, instructions=tasks_string, num=len(tasks)
        )
        task_validator_revserse = validator_prompt.format(
            action=chain_action, instructions=task_string_reverse, num=len(tasks)
        )
        try:
            # maybe order senstitive: swap the instrction1 and instruction2
            result = self.base_model([HumanMessage(content=task_validator)])

            result_reverse = self.base_model(
                [HumanMessage(content=task_validator_revserse)]
            )
            
            splitted = result.split("Justification:")
            candidates = splitted[0].strip()
            answer_justification = splitted[1].strip()

            splitted_reverse = result_reverse.split("Justification:")
            candidates_reverse = splitted_reverse[0].strip()
            answer_justification_reverse = splitted_reverse[1].strip()

            # extract probablity and the reason
            if self.task != "alfworld":
                compiler = re.compile(r"G\d+: ([A-Z])\.? P\d+: (\d+\.\d+)")
            else:
                compiler = re.compile(r"P\_([A-Z])\: (\d+\.\d+)")
            matches = compiler.findall(candidates)
            candidates = [(g, float(p)) for g, p in matches]

            matches_reverse = re.findall(compiler, candidates_reverse)

            candidates_reverse = [(g, float(p)) for g, p in matches_reverse]

            aggregated_probs = self.aggregate_probability(
                candidates, candidates_reverse
            )
            result = {
                "candidates": candidates,
                "candidates_reverse": candidates_reverse,
                "answer_justification": answer_justification,
                "answer_justification_reverse": answer_justification_reverse,
                "inferred_tasks_reason": inferred_tasks_reason,
                "tasks_string": tasks_string,
                "task_string_reverse": task_string_reverse,
                "gold_option": gold_label,
                "aggregated_probs": aggregated_probs
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
                "aggregated_probs": "N/A"}
        return result

    def task_validator_logtis(self, validator_prompt, chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks):
        
        task_validator = validator_prompt.format(
            action=chain_action, instructions=tasks_string, num=len(tasks)
        )
        task_validator_revserse = validator_prompt.format(
            action=chain_action, instructions=task_string_reverse, num=len(tasks)
        )

        # maybe order senstitive: swap the instrction1 and instruction2
        result, logits = self.base_model([HumanMessage(content=task_validator)], probs=True)

        result_reverse, logits_reverse = self.base_model(
            [HumanMessage(content=task_validator_revserse)], probs=True
        )

        # get the probability of all options from the token_logits and do softmax
        indices = [self.tokenizer.convert_tokens_to_ids(option) for option in ['Yes', 'No']]

        # get softmax
        option_probs = softmax(logits[indices])

        # get the prob of gold label
        option2ix = {option: ix for ix, option in enumerate(self.reverse_mapping.keys())}
        gold_label_prob = option_probs[option2ix[gold_label]]

        #reverse
        option_probs_reverse = softmax(logits_reverse[indices])
        gold_label_prob_reverse = option_probs_reverse[option2ix[self.reverse_mapping[gold_label]]]
            
        ## aggerate the probability: multiply or average?
        aggregated_probs = gold_label_prob * gold_label_prob_reverse

        result ={
                "inferred_tasks_reason": inferred_tasks_reason,
                "tasks_string": tasks_string,
                "task_string_reverse": task_string_reverse,
                "gold_option": gold_label,
                "aggregated_probs": aggregated_probs
            }  
        return result

    def task_validator(self, prob_estimation,  chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks):
        if prob_estimation == 'verbalized_prob':
            result = self.task_validator_verbalized_prob(self.prompts[self.task]['verbalized_prob'], chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks)
        elif prob_estimation == 'logtis':
            result = self.task_validator_logtis(self.prompts[self.task]['logtis'], chain_action, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks)
        return result

    def aggregate_probability(self, probs, probs_reverse):
        
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
            if kwargs.get("risk_mode", False):
                validator_prompt = web_task_validator_risk_sen
            else:
                validator_prompt = web_task_validator

        elif self.task.lower() == "hotpotqa":
            action_chain, original_task = get_action_chain_hotpotqa(message)
            task_inference_prompt = hotpot_k_task_inference
            validator_prompt = hotpot_task_validator

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
                if self.task != "alfworld":
                    tasks.append("None of the above")

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

            results[ix] = self.task_validator(validator_prompt, chain_str, tasks_string, task_string_reverse, gold_label, inferred_tasks_reason, tasks)
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
            for ix in obj['real-time eval']:
                if obj['real-time eval'][ix]["aggregated_probs"] == "N/A":
                    pred_label = "N/A"
                    break
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
