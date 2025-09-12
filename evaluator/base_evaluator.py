from llm import AnyLLM
from .utils_eval import extract_price
import numpy as np



class BaseEvaluator:
    def __init__(self, task, **kwargs) -> None:
        self.task = task
        assert task in ['webshop', 'hotpotqa', 'alfworld', 'rjudge'], f"your task should be one of ['webshop', 'hotpotqa', 'alfworld', 'rjudge']"
        if "gpt" in kwargs["model_name"]:
            kwargs.update({"model_kwargs": {"temperature": kwargs.get("temperature", 0.0), "max_tokens": kwargs.get("max_tokens", 8192), "logprobs": kwargs.get("logprobs", False)}})
        else:
            kwargs.update({"max_new_tokens": kwargs.get("max_tokens", 8192)})
        
        # elif "llama" in kwargs["model_name"]:

        #     self.base_model = LocalLLM(
        #         model_pth=kwargs["model_path"],
        #         temperature=kwargs.get("temperature", 0.0),
        #         tokenizer_pth=kwargs["model_path"],
        #         max_new_tokens=kwargs.get("max_tokens", 500),
        #     )
        self.base_model = AnyLLM(**kwargs)

        self.task = task
        self.label_map = {
            1: "A",
            2: "B",
            3: "C",
            4: "D",
            5: "E"
        }
        # if task == 'alfworld':
        self.reverse_mapping = self.generate_mapping(4)
        # else:
        #     self.reverse_mapping = self.generate_mapping(5)

    def generate_mapping(self, num_choices):
        # Generate a list of options
        options = list('ABCDE'[:num_choices])
        
        # Generate the mapping
        mapping = {options[i]: options[-(i+1)] for i in range(num_choices)}
        
        return mapping

    def evaluate(self, message, **kwargs):
        return NotImplementedError
    
    @staticmethod
    def get_confusion_matrix(y_pred, y_true, json_objects, threshold, task, risk_mode=False):
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        cost = 0
        tp_tasks, fn_envs, y_pred_break = [], [], []
        # env_neg, env_pos = [], []
        tp_predictions = []

        for ix in range(len(y_pred)):
            alert = False
            for y_pred_step in y_pred[ix]:
                # incorrect
                if y_pred_step >= threshold:
                    alert = True
                    if y_true[ix] == 1:
                        true_positive += 1
                        tp_predictions.append({"env_name": json_objects[ix]["env_name"], "true_label": "Incorrect"})
                    else:
                        false_positive += 1
                        if not (risk_mode and task == "webshop"):
                            cost += 1
                        tp_predictions.append({"env_name": json_objects[ix]["env_name"], "true_label": "Correct"})
                    try:
                        tp_tasks.append(json_objects[ix]["input_task"])
                    except:
                        pass
                    y_pred_break.append(y_pred_step)       
                    break

            if not alert:
                if y_true[ix] == 1:
                    false_negative += 1
                    if risk_mode and task == "webshop":
                        try:
                            if "selected_product_price" in json_objects[ix]:
                                cost += float(json_objects[ix]["selected_product_price"][0])
                            else:
                                cost += float(extract_price(json_objects[ix]["complete_traj"][0])[0])
                        except:
                            print(f"{json_objects[ix]['env_name']} is missing price")

                    else:
                        cost += 1
                    
                    fn_envs.append(json_objects[ix]["env_name"])
                else:
                    true_negative += 1
                y_pred_break.append(y_pred[ix][-1])
        

        return (
            true_negative,
            true_positive,
            false_negative,
            false_positive,
            cost,
            tp_tasks,
            tp_predictions,
            fn_envs,
            y_pred_break
        )
        
    @classmethod
    def metric(self, **kwargs):
        return NotImplementedError
    
    @classmethod
    def get_fnr(self, y_true, y_pred, risk_levels):
        # get the accuracy of trajectories under different risk levels
        risk_levels = np.array(risk_levels)
        risk_levels_labels = ["low_risk", "median_risk", "high_risk"]
        false_negative_rate_list = []
        for risk_level in risk_levels_labels:
            mask = risk_levels == risk_level
            y_true_risk = np.array(y_true)[mask]
            y_pred_risk = np.array(y_pred)[mask]
            # false negatives rate: fn/(fn+tp)
            false_negative_rate = np.sum(y_true_risk == 1 & y_pred_risk == 0)/np.sum(y_true_risk == 1)
            false_negative_rate_list.append(false_negative_rate)
            # # true positives rate: tp/(tp+fn)
            # true_positive_rate = np.sum(y_true_risk == 1 & y_pred_risk == 1)/np.sum(y_true_risk == 1)
            # true_positive_rate_list.append(true_positive_rate)
        
        return false_negative_rate_list
            
    