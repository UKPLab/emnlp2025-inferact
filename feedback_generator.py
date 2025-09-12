from llm import AnyLLM
from prompts.webshop_prompt import ws_feedback_prompt, ws_afterwards_feedback_prompt, ws_binary_feedback_prompt
from prompts.hotpotqa_prompt import hot_feedback_prompt, hot_afterwards_feedback_prompt, hotpot_binary_feedback_prompt
from prompts.alfworld_prompt import alfworld_feedback_prompt, alfworld_afterwards_feedback_prompt, alfworld_binary_feedback_prompt
from langchain.schema import HumanMessage
import argparse
import json
import os
from evaluator.utils_eval import get_trajectory_webshop, get_action_chain_webshop, get_action_chain_hotpotqa, get_action_chain_alfworld, extract_price, load_rejected_envs, load_halted_envs
from tqdm import tqdm
from actor.hotpotqa.fewshots import REFLECTIONS
from actor.hotpotqa.agents import truncate_scratchpad
import ipdb

class FeedbackGenerator:
    def __init__(self, task, feedback_type, eval_method, threshold, **kwargs):
        self.base_model = AnyLLM(
            model_name="gpt4-turbo",
            model_kwargs={"temperature": kwargs.get("temperature", 0.0), "max_tokens": kwargs.get("max_tokens", 500)}
        )
        self.task = task
        self.feedback_type = feedback_type
        self.eval_method = eval_method
        self.threshold = threshold
        self.feedback_dir = kwargs["feedback_dir"]
        self.last_rejected_path = kwargs["last_rejected_path"]
        self.traj_dir = kwargs["actor_traj_dir"]
        self.trial_num = kwargs["trial_num"]
        expert_traj_dir = kwargs.get("expert_traj_dir", None)

        if self.task == "webshop":
            self.initial_feedback_prompt = ws_feedback_prompt
            self.subseqent_feedback_prompt = ws_afterwards_feedback_prompt
            self.binary_feedback_prompt = ws_binary_feedback_prompt
            with open('./prompts/reflection_few_shot_examples.txt', 'r') as f:
                self.few_shot_examples = f.read()

        elif self.task == "hotpotqa":
            self.initial_feedback_prompt = hot_feedback_prompt
            self.subseqent_feedback_prompt = hot_afterwards_feedback_prompt
            self.binary_feedback_prompt = hotpot_binary_feedback_prompt
            self.few_shot_examples = REFLECTIONS

        elif self.task == "alfworld":
            self.initial_feedback_prompt = alfworld_feedback_prompt
            self.subseqent_feedback_prompt = alfworld_afterwards_feedback_prompt
            self.env2traj = self.clean_expert_traj_alfworld(expert_traj_dir)
            self.binary_feedback_prompt = alfworld_binary_feedback_prompt
            # few-shot examples for reflection based on binary feedback
            with open('./actor/alfworld/prompts/reflexion_few_shot_examples.txt', 'r') as f:
                self.few_shot_examples = f.read()

        if not os.path.exists(self.feedback_dir):
            os.makedirs(self.feedback_dir)

        self.get_action_chain_func = {"webshop": get_action_chain_webshop, "hotpotqa": get_action_chain_hotpotqa, "alfworld": get_action_chain_alfworld}
        self.get_trajectory_func = {"webshop": get_trajectory_webshop, "hotpotqa": get_action_chain_hotpotqa, "alfworld": get_action_chain_alfworld}
        
    def clean_expert_traj_alfworld(self, expert_traj_dir):
        # traj_dir = "/storage/ukp/work/fang/reflexion/alfworld_runs/expert_traj"
        files = os.listdir(expert_traj_dir)
        env2traj = {}
        for file in files:
            traj = []
            with open(os.path.join(expert_traj_dir, file), "r") as f:
                data = json.load(f)
            expert_traj = data["expert_traj"][1:]
            for step in expert_traj:
                traj.append("> " + step['action'])
                traj.append(step['observation'])
            env_name = data['env_name']
            env2traj[str(env_name)] = traj
        return env2traj

    def prompt_binary_feedback_gen(self, trace, prev_feedback):
        messages, original_task = self.get_trajectory_func[self.task](trace['path'])

        if self.task == 'alfworld':
            prompt = self.binary_feedback_prompt + f"\n\n{self.few_shot_examples}\n\n{trace}"
            if len(prev_feedback) > 0:
                prompt += "\n\nPlans from past attempts:\n" + "\n- ".join([f.strip() for f in prev_feedback])
        elif self.task == 'hotpotqa':
            prompt = self.binary_feedback_prompt.format(
                examples=self.few_shot_examples,
                question=original_task,
                scratchpad=truncate_scratchpad("\n".join(messages[0])),
            )

        elif self.task == 'webshop':
            scenario = f"Instruction:\n{original_task}\n\n" + "\n".join(messages[0])
            prompt = self.binary_feedback_prompt.format(
                Examples=self.few_shot_examples,
                scenario=scenario)
            if len(prev_feedback) > 0:
                prompt += "\n\nPlans from past attempts:\n" + "\n- ".join([f.strip() for f in prev_feedback])

        result = self.base_model([HumanMessage(content=prompt)])
        return result


    def prompt_nl_feedback_gen(self, env, prompt, trace, prev_feedback):
        
        messages, original_task = self.get_trajectory_func[self.task](trace['path'])
        if self.task == 'webshop':
            gold_label_actor = f"""{trace["true_item"]['asin']} [SEP] {trace["true_item"]['name']}. It has following attributes: {", ".join(trace["true_item"]['attributes'])}""" + f""" and goal option(s) are: {", ".join(trace["true_item"]["goal_options"])}""" if trace["true_item"]["goal_options" ] else ""
        
        elif self.task == 'hotpotqa':
            gold_label_actor = {trace["golden_answer"]}

        elif self.task == 'alfworld':
            gold_label_actor = "\n".join(self.env2traj[env])

        kwargs = {
                "task": original_task,
                "gold_label_actor": gold_label_actor,
                "incorrect_action_chain": "\n".join(messages[0]),
                "previous_feedback": "\n-" + "\n- ".join([f.strip() for f in prev_feedback]) if prev_feedback else ""
                }
        
        if prompt == 'init_prompt':
            prompt = self.initial_feedback_prompt.format(**kwargs)

        elif prompt == 'subsequent_prompt':
            prompt = self.subseqent_feedback_prompt.format(**kwargs)

        if 'previous_feedback' in kwargs:
            prompt += "\n\nPlans from past attempts:\n" + "\n- ".join([f.strip() for f in kwargs['previous_feedback']])
        result = self.base_model([HumanMessage(content=prompt)])
        if result.startswith('Feedback:'):
            return result.split("Feedback:")[1].strip()
        else:
            return result

    def generate_feedback(self):

        # load existing feedbacks
        existing_feedbacks = self.load_existing_feedback()

        # load rejected envs from the last trial
        rejects = load_rejected_envs(self.task, self.last_rejected_path, self.eval_method)

        # load halted envs from the last trial
        halted = load_halted_envs(self.traj_dir)

        rejects = rejects + halted
        for env in tqdm(rejects):
            file = env + ".txt"
            prev_feedback = []
            if file in existing_feedbacks:
                with open(os.path.join(self.feedback_dir, file), "r") as f:
                    prev_feedback = f.readlines()

                if len(prev_feedback) >= int(self.trial_num) + 1:
                    print(f"feedback for {env} already exists whose length is {len(prev_feedback)}")
                    continue
                else:
                    print(f"feedback for {env} exists but not enough")
                    prompt = 'subsequent_prompt'

            else:
                print(f"feedback for {env} does not exist")
                prompt = 'init_prompt'
            try:
                with open(os.path.join(self.traj_dir, env + ".json"), "r") as f:
                    trace = json.load(f)
            except:
                print(f"there is no {env}.json")
                continue
            if trace['trace_correct']:
                print(f"trace for {env} is correct")
                continue
            if self.feedback_type == "binary":
                feedback = self.prompt_binary_feedback_gen(trace, prev_feedback)
            else:
                feedback = self.prompt_nl_feedback_gen(env, prompt, trace, prev_feedback)
            with open(os.path.join(self.feedback_dir, f"{file}"), "a") as f:
                f.write(feedback + "\n")
                f.flush()
    
    def load_rejected_envs(self, eval_method):
        with open(self.last_rejected_path, "r") as f:
            predicted_pos = json.load(f)
        if self.task == "webshop":
            total_num = 136
        elif self.task == "hotpotqa":
            total_num = 120
        elif self.task == "alfworld":
            total_num = 53
        
        if eval_method == "multi_step":
            predicted_pos = predicted_pos['prod']
        tp, fp = [], []
        for rej in predicted_pos:
            if rej['true_label'] == 'Correct':
                fp.append(rej['env_name'])
            else:
                tp.append(rej['env_name'])
        rejects = tp[:max(total_num-len(fp), 0)]
        return rejects
    
   

    def load_existing_feedback(self):
        return [f for f in os.listdir(self.feedback_dir) if f.endswith('.txt')]
        


