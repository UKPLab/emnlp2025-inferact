from .base_evaluator import BaseEvaluator
import re
from prompts.webshop_prompt import web_multi_step, web_multi_step_risk_sen, web_multi_step_inferact
from prompts.hotpotqa_prompt import hotpot_multi_step, hotpot_multi_step_inferact
from prompts.alfworld_prompt import alfworld_multi_step, alfworld_multi_step_risk_sen, alfworld_multi_step_inferact
from prompts.rjudge_prompt import rjudge_multi_step
from evaluator.utils_eval import get_trajectory_rjudge
from .utils_eval import (
    get_trajectory_webshop,
    get_trajectory_hotpotqa,
    get_trajectory_alfworld,
    extract_price,
    get_risk_level)
from langchain.schema import HumanMessage
import ipdb
import numpy as np

class MultistepEvaluator(BaseEvaluator):
    def __init__(self, task, **kwargs) -> None:
        super().__init__(task, **kwargs)
        
    def evaluate(self, message, compiler=re.compile(r"Step ([0-9]+): ([\d.]+)"), intent_compiler=re.compile(r"Step ([0-9]+)-Intent: ([\w\s]+)"), **kwargs):
        if self.task.lower() == "webshop":
            messages, original_task = get_trajectory_webshop(message)

            if kwargs.get("risk_mode", False):
                multi_step_prompt = web_multi_step_risk_sen
            else:
                multi_step_prompt = web_multi_step_inferact

        elif self.task.lower() == "hotpotqa":
            messages, original_task = get_trajectory_hotpotqa(message)
            # multi_step_prompt = hotpot_multi_step
            multi_step_prompt = hotpot_multi_step_inferact

        elif self.task.lower() == "rjudge":
            messages, original_task = get_trajectory_rjudge(message)
            multi_step_prompt = rjudge_multi_step

        elif self.task.lower() == "alfworld":
            messages, original_task = get_trajectory_alfworld(message)
            if kwargs.get("risk_mode", False):
                multi_step_prompt = alfworld_multi_step_risk_sen
            else:
                # multi_step_prompt = alfworld_multi_step
                multi_step_prompt = alfworld_multi_step_inferact
        if kwargs['risk_mode']:
            risk_level = get_risk_level(self.task, messages)
            if risk_level != "high_risk":
                return None

        items = {}
        for ix, traj in enumerate(messages):
            trajectory = "\n".join(traj)
            prompt = multi_step_prompt.format(
                instruction=original_task, trajectory=trajectory
            )
            result = self.base_model([HumanMessage(content=prompt)])
            try:
                splitted = result.split("Justification:")
                matches = compiler.findall(splitted[0])
                # get intent
                # intent_matches = intent_compiler.findall(splitted[0])
                probs = [(g, float(p)) for g, p in matches]
                # justifcation = " ".join([f"{g}.{i}" for g, i in intent_matches])
                justifcation = splitted[1] if len(splitted) > 1 else ""
            except:
                probs = []
                justifcation = result
            items[ix] = {
                "probs": probs,
                "justification": justifcation,
                "input_messages": traj,
            }
        if self.task == "WebShop":
            return {
                "real-time eval": items,
                "complete_traj": messages,
                "input_task": original_task,
                "env_name": kwargs["env_name"],
                "selected_product_price": extract_price(messages[0]),
            }
        else:
            return {
                "real-time eval": items,
                "complete_traj": messages,
                "input_task": original_task,
                "env_name": kwargs["env_name"],
            }
    @classmethod
    def metric(cls, json_objects, **kwargs):
        
        def get_prediction_probs(json_objects):
            y_pred_all, y_true = [], []
            for obj in json_objects:
                # for different steps
                y_pred = []
                # y_pred.append(0)
                # probs = [opt[1] for opt in obj['real-time eval'][ix]["probs"]]
        #             final_prob = kwargs['aggregated_func'](probs)
        #             y_pred[-1] = 1 - final_prob
                # update the y_pred as it might has multiple actions
                for ix in obj["real-time eval"]:
                    if obj['real-time eval'][ix]["probs"]:
                        probs = [opt[1] for opt in obj['real-time eval'][ix]["probs"]]
                        final_prob = kwargs['aggregated_func'](probs)
                        y_pred.append(1 - final_prob)
                    else:
                        y_pred.append(1)
                y_pred_all.append(y_pred)
                
                if obj["trace_correct"]:
                    y_true.append(0)
                else:
                    y_true.append(1)

            return y_pred_all, y_true
        
        highest_macro_f1 = -1
        y_pred, y_true = get_prediction_probs(json_objects)
        
        if kwargs["threshold_search"]:
            for v in range(1, 100):
                threshold = v / 100
                (
                    true_negative,
                    true_positive,
                    false_negative,
                    false_positive,
                    cost,
                    tp_tasks,
                    tp_predictions,
                    fn_envs,
                    y_pred_break
                ) = cls.get_confusion_matrix(y_pred, y_true, json_objects, threshold, kwargs["task"], risk_mode=False)
                
                
                precision = true_positive / (true_positive + false_positive + 1e-10)
                recall = true_positive / (true_positive + false_negative + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                recall_neg = true_negative / (true_negative + false_positive + 1e-10)
                precision_neg = true_negative / (true_negative + false_negative + 1e-10)
                f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-10)

                macro_f1 = (f1 + f1_neg) / 2
                if macro_f1 > highest_macro_f1:
                    highest_macro_f1 = macro_f1
                    highest_macro_f1_neg = f1_neg
                    highest_macro_f1_pos = f1
                    optimal_threshold = threshold
                    optimal_tp_tasks = tp_tasks
                    optimal_cost = cost
                    optimal_tp_predictions = tp_predictions
                    optimal_fn_envs = fn_envs
                    optimal_true_negative = true_negative
                    optimal_true_positive = true_positive
                    optimal_false_positive = false_positive
                    optimal_false_negative = false_negative
            
                
                # if cost < lowest_cost:
                #     lowest_cost = cost
                #     optimal_threshold = threshold
                #     optimal_tp_tasks = tp_tasks
                #     optimal_tp_predictions = tp_predictions
                #     optimal_fn_envs = fn_envs
                #     optimal_true_negative = true_negative
                #     optimal_true_positive = true_positive
                #     optimal_false_positive = false_positive
                #     optimal_false_negative = false_negative
            # ipdb.set_trace()
            return {
                "highest_macro_f1_pos": highest_macro_f1_pos,
                "highest_macro_f1": highest_macro_f1,
                "highest_macro_f1_neg": highest_macro_f1_neg,
                "balanced_acc": (optimal_true_positive/(optimal_true_positive + optimal_false_positive + 1e-10) + optimal_true_negative/(optimal_false_negative+optimal_true_negative + 1e-10)) / 2,
                "cost": optimal_cost,
                "er": (optimal_true_negative - optimal_false_negative) / (optimal_true_negative + optimal_false_negative + 1e-10),
                "er_pos": (optimal_true_positive - optimal_false_positive) / (optimal_true_positive + optimal_false_positive + 1e-10),
                "optimal_threshold": optimal_threshold,
                "true_positive": optimal_true_positive,
                "true_negative": optimal_true_negative,
                "false_positive": optimal_false_positive,
                "false_negative": optimal_false_negative,
                "y_true": y_true,
                "y_pred": 
                    y_pred_break,
                "false_negative_env": optimal_fn_envs,
            }, optimal_tp_tasks, optimal_tp_predictions
        else:
            threshold = kwargs["threshold"]

            (
                true_negative,
                true_positive,
                false_negative,
                false_positive,
                cost,
                tp_tasks,
                tp_predictions,
                fn_envs,
                y_pred_break
            ) = cls.get_confusion_matrix(y_pred, y_true, json_objects, threshold, kwargs["task"], risk_mode=False)
            
            precision = true_positive / (true_positive + false_positive + 1e-10)
            recall = true_positive / (true_positive + false_negative + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            recall_neg = true_negative / (true_negative + false_positive + 1e-10)
            precision_neg = true_negative / (true_negative + false_negative + 1e-10)
            f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-10)
            macro_f1 = (f1 + f1_neg) / 2
            highest_macro_f1_pos = f1
            highest_macro_f1 = macro_f1
            highest_macro_f1_neg = f1_neg            
            
            return {
                "highest_macro_f1_pos": highest_macro_f1_pos,
                "highest_macro_f1": highest_macro_f1,
                "highest_macro_f1_neg": highest_macro_f1_neg,
                "balanced_acc": (precision + precision_neg) / 2,
                "cost": cost,
                "er": (true_negative - false_negative) / (true_negative + false_negative + 1e-10),
                "er_pos": (true_positive - false_positive) / (true_positive + false_positive + 1e-10),
                "threshold": threshold,
                "true_positive": true_positive,
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "y_true": y_true,
                "y_pred": 
                    y_pred_break,
                "false_negative_env": fn_envs,
                
            }, tp_tasks, tp_predictions
        
        
        
        # cost = 0
        # true_negative, false_positive, false_negative, true_positive = 0, 0, 0, 0
        # fn_envs, predictions, y_true, y_pred, tp_tasks = [], [], [], [], []
        # for obj in json_objects:
        #     y_pred.append(0)
        #     for ix in obj['real-time eval']:
        #         if obj['real-time eval'][ix]["probs"]:
        #             probs = [opt[1] for opt in obj['real-time eval'][ix]["probs"]]
        #             final_prob = kwargs['aggregated_func'](probs)
        #             y_pred[-1] = 1 - final_prob
        #             if final_prob > float(kwargs['threshold']):
        #                 pred_label = "Correct"
        #             else:
        #                 pred_label = "Incorrect"
        #                 break
        #         else:
        #             pred_label = "N/A"
        #             break
        #     if obj["trace_correct"]:
        #         y_true.append(0)
        #         if pred_label == "Correct":
        #             true_negative += 1
        #         else:
        #             if pred_label == "N/A":
        #                 y_pred[-1] = 1
        #             false_positive += 1
        #             predictions.append({"env_name": obj["env_name"], "true_label": "Correct"})
        #     else:
        #         y_true.append(1)
        #         if pred_label == "Incorrect":
        #             true_positive += 1
        #             predictions.append({"env_name": obj["env_name"], "true_label": "Incorrect"})
        #             try:
        #                 tp_tasks.append(obj["input_task"])
        #             except:
        #                 continue
        #         else:
        #             if pred_label == "N/A":
        #                 y_pred[-1] = 0
        #             if kwargs['task'] == 'webshop':
        #                 try:
        #                     cost += float(obj["selected_product_price"][0])
        #                 except:
        #                     pass
        #             else:
        #                 cost += 1
        #             false_negative += 1
        #             fn_envs.append(obj["env_name"])
        # return {
        #     "true_positive": true_positive,
        #     "true_negative": true_negative,
        #     "false_positive": false_positive,
        #     "false_negative": false_negative,
        #     "false_negative_env": fn_envs,
        #     "y_true": y_true,
        #     "y_pred": y_pred,
        #     "fn_envs": fn_envs,
        #     "fn_cost": cost
        # }, tp_tasks, predictions