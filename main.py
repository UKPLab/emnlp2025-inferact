from evaluator import StandardEvaluator, MultistepEvaluator, StandardEvaluatorSC, InferAct
import argparse
import os
import json
from evaluator.utils_eval import convert_json_objs
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from netcal.metrics import ECE
import numpy as np
from feedback_generator import FeedbackGenerator
from actor.alfworld.alfworld_trial import run_alfworld
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actor/webshop/'))
# for path in sys.path:
#     print(path)
from actor.webshop.webshop_trial import run_webshop
import sys
from actor.hotpotqa.agents import ReactAgent
import logging
import ipdb

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')

def run_alfworld_webshop(args, feedback_dir):

    # initialize environment configs
    env_configs = []
    for i in range(args.num_envs):
        if args.trial_num == 0:
            env_configs += [{
            'name': f'env_{i}',
            'memory': [],
            'is_skip': False
            }]
        else:
            if os.path.exists(os.path.join(feedback_dir, f"{i}.txt")):
                with open(os.path.join(feedback_dir, f"{i}.txt"), 'r') as f:
                    feedback = f.readlines()

                # this indicates that the task is solved
                if len(feedback) == args.trial_num:
                    env_configs += [{
                        'name': f'env_{i}',
                        'memory': [fed.strip() for fed in feedback],
                        'is_skip': False
                    }]

                else:
                    env_configs += [{
                        'name': f'env_{i}',
                        'memory': [],
                        'is_skip': True
                    }]

            else:
                env_configs += [{
                    'name': f'env_{i}',
                    'memory': [],
                    'is_skip': True
                }]

    # run trials
    trial_log_dir: str = os.path.join(args.traj_dir, args.task, f"retrial_{args.trial_num}", args.feedback_type if int(args.trial_num) > 0 else '')

    print(f"""
    -----
    Starting run with the following parameters:

    Number of environments: {args.num_envs}

    Sending trajectories to `{trial_log_dir}`
    -----
    """)

    if not os.path.exists(trial_log_dir):
        os.makedirs(trial_log_dir)

    # run trial
    if args.task == "alfworld":
        run_alfworld(trial_log_dir, env_configs, args.model_name)
    elif args.task == "webshop":
        run_webshop(trial_log_dir, env_configs, args.model_name)


def run_hotpotqa(args, feedback_dir, actor_flies_dir):
    with open("./outputs/actor-traj/hotpotqa/data.json", "r") as f:
        data = json.load(f)
    env2data = {row['env_name']: row for row in data}

    if not os.path.exists(actor_flies_dir):
        os.makedirs(actor_flies_dir)

    if args.trial_num == 0:
        # generate feedback for the first trial
        agents = [ReactAgent(row['question'], row['answer'], []) for row in data]
    else:
        agents = []
        # load feedback
        feedback_files = os.listdir(feedback_dir)
        # actor files
        actor_files = os.listdir(actor_flies_dir)

        for file in feedback_files:
            with open(os.path.join(feedback_dir, file), "r") as f:
                feedback = f.readlines()

            if len(feedback) == args.trial_num and not file.replace('.txt', '.json') in actor_files:
                question = env2data[file.replace('.txt', '')]['question']
                answer = env2data[file.replace('.txt', '')]['answer']
                agents.append(ReactAgent(question, answer, feedback))

    for i, agent in enumerate(tqdm([a for a in agents if not a.is_correct()])):
        agent.run()
        with open(os.path.join(actor_flies_dir, f"{i}.json"), "w") as f:
            json.dump(agent.output_traj(), f, ensure_ascii=False, indent=4)


# load existing evaluated data, load data from actor trajectories to evaluate
def run_evaluator(args, evaluator, actor_traj_dir, last_rejected_files, eval_dir, kwargs):

    # path to save eval results
    eval_result_dir = os.path.join(eval_dir, "eval_results")
    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)
    
    # load evaluated data
    existing_envs = []
    evaluated_file = os.path.join(eval_result_dir, f"{args.feedback_type}.txt" if args.trial_num > 0 else "init.txt")
    if os.path.exists(evaluated_file):
        entries = convert_json_objs(evaluated_file)
        existing_envs = [entry["env_name"] + '.json' for entry in entries]

    f_file = open(evaluated_file, "a")

    for file in tqdm(last_rejected_files):
        if not file.endswith(".json"):
            file = file + ".json"
        if file in existing_envs:
            print(f"{file} is already evaluated")
            continue
        kwargs["env_name"] = file.split(".")[0]

        if "config" in file:
            continue
        if not os.path.exists(os.path.join(actor_traj_dir, file)):
            logging.info(f"Evaluating the trajectory, but there is no {os.path.join(actor_traj_dir, file)}")
            continue
        with open(os.path.join(actor_flies_dir, file), "r") as f:
            data = json.load(f)

        is_halted = data.get("is_halted", False)
        if is_halted:
            print(f"{file} is halted")
            continue
        message = data["path"]

        output = evaluator.evaluate(message, **kwargs)
        if not output:
            continue
        # add into a json object
        output["trace_correct"] = data["trace_correct"]
        if args.task == "webshop":
            output["true_item"] = data["true_item"]

        f_file.write(json.dumps(output, indent=4) + "\n")
        f_file.flush()
    f_file.close()

def cal_metrics(eval_result_file, kwargs, evaluator):
    false_negative_env = []
    
    json_objects = convert_json_objs(eval_result_file)

    output, tp_tasks, predicted_pos = evaluator.metric(json_objects, **kwargs)

    true_positive = output["true_positive"]
    true_negative = output["true_negative"]
    false_positive = output["false_positive"]
    false_negative = output["false_negative"]
    false_negative_env = output.get('false_negative_env', [])
    y_true = output["y_true"]
    y_pred = output["y_pred"]
    fn_cost = output["fn_cost"]


    epsilon = 1e-17
    recall = true_positive / (true_positive + false_negative + epsilon)
    precision = true_positive / (true_positive + false_positive + epsilon)
    f1 = 2 * recall * precision / (recall + precision + 1e-7)
    if y_pred and any(y_pred):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
        auc_pr = auc(recalls, precisions)
        ece = ECE(10)
        print(y_pred, y_true)
        ece_score = ece.measure(np.array(y_pred), np.array(y_true))
        auc_roc = roc_auc_score(y_true, y_pred)
        precisions = precisions.tolist()
        recalls = recalls.tolist()
        thresholds = thresholds.tolist()
   
    else:
        precisions, recalls, thresholds = [], [], []
        auc_pr = -1
        ece_score = -1
        auc_roc = -1
        
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    return { "recall": recall,
            "precision": precision,
            "f1": f1,
            "auc_pr": auc_pr,
            "ece": ece_score,
            "auc_roc": auc_roc,
            'precision_list': precisions,
            'recall_list': recalls,
            "thresholds_list": thresholds,
            'false_neg_envs': false_negative_env,
            "false_neg_cost": fn_cost,
            "accuracy": accuracy,
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "threshold": kwargs.get("threshold", ""),
            "positive_count": sum(y_true),
            "negative_count": len(y_true) - sum(y_true),
            "total_count": len(y_true)}, tp_tasks, predicted_pos


def run_metrics(args, evaluator, eval_dir):
    eval_metric_dir = os.path.join(eval_dir, "eval_metrics", args.feedback_type if args.trial_num > 0 else "init")
    if not os.path.exists(eval_metric_dir):
        os.makedirs(eval_metric_dir)
    eval_result_file = os.path.join(eval_dir, "eval_results", f"{args.feedback_type}.txt" if args.trial_num > 0 else "init.txt")
    
    kwargs = {'threshold': args.threshold, 'task': args.task, 'prob_estimation': args.prob_estimation}
    if args.eval_method == "multi_step":
        result = {}
        tp_tasks = {}
        pred_pos = {}
        for func in [np.prod, np.max, np.mean, np.min]:
            kwargs["aggregated_func"] = func
            result[func.__name__], tp_tasks[func.__name__], pred_pos[func.__name__] = cal_metrics(eval_result_file, kwargs, evaluator)
    else:
        result, tp_tasks, pred_pos = cal_metrics(eval_result_file, kwargs, evaluator)
    threshold = kwargs.get("threshold", "")
    with open(os.path.join(eval_metric_dir, f"predicted_pos{threshold}.json"), "w") as f:
        json.dump(pred_pos, f, indent=4)
    
    with open(os.path.join(eval_metric_dir, f"metrics{threshold}.json"), "w") as f:
        json.dump(result, f, indent=4)

    with open(os.path.join(eval_metric_dir, f"tp_tasks{threshold}.json"), "w") as f:
        json.dump(tp_tasks, f, indent=4)

    with open(os.path.join(eval_metric_dir, f"predicted_pos{threshold}.json"), "w") as f:
        json.dump(pred_pos, f, indent=4)

if __name__ == "__main__":
    # add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt4-turbo", help="model name, gpt4-turbo, gpt35-turbo")
    parser.add_argument("--model_path", type=str, default="", help="path to the local model")
    parser.add_argument("--eval_method", type=str, default="standard", help="standard, multi_step, standard_sc, inferact")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/evaluation_results",
    )
    parser.add_argument("--feedback_type", type=str, default="none", help="results with different feedback types, nl, binary")
    parser.add_argument("--task", type=str, default="", required=True)
    parser.add_argument("--trial_num", type=int, default=0)
    parser.add_argument("--risk_mode", action="store_true", help="sensitive mode")
    parser.add_argument("--signature", type=str, default="")
    parser.add_argument("--threshold", type=str, default="")
    parser.add_argument("--traj_dir", type=str, default="./outputs/actor-traj")
    parser.add_argument("--prob_estimation", type=str, default="logits_multiclass")
    ## args for Alfworld
    parser.add_argument("--num_envs", type=int, help="The number of environments per trial in Alfworld or webshop")
    parser.add_argument("--use_memory", action='store_true', help="Allow the Agent to use memory")
    ## actions
    parser.add_argument("--run_agents", action="store_true", help="run agents")
    parser.add_argument("--do_feedback_gen", action="store_true", help="generate nl feedback for rejected envs")
    parser.add_argument("--do_eval", action="store_true", help="run evaluation")

    args = parser.parse_args()

    assert args.eval_method in ["standard", "multi_step", "standard_sc", "inferact"], "eval_method should be one of standard, multi_step, standard_sc, inferact"

    kwargs = {"model_path": args.model_path, "model_name": args.model_name, "risk_mode": args.risk_mode, "prob_estimation": args.prob_estimation}

    if args.eval_method == "self_consistency":
        kwargs["temperature"] = 0.7

    evaluators = {"standard": StandardEvaluator, "multi_step": MultistepEvaluator, "standard_sc": StandardEvaluatorSC, "inferact": InferAct}
    evaluator = evaluators[args.eval_method](args.task, **kwargs)

    save_dir = f"{args.save_dir}/{args.model_name}"


    ## All paths
    eval_dir = os.path.join(save_dir, args.task, args.eval_method, args.prob_estimation, args.signature, f"retrial_{args.trial_num}" if not args.risk_mode else f"retrial_{args.trial_num}_sensitive")
    actor_flies_dir = (
        f"{args.traj_dir}/{args.task}/retrial_{args.trial_num}/{args.feedback_type if args.trial_num > 0 else ''}"
    )
    feedback_dir = os.path.join("./outputs/feedbacks", args.task,  args.feedback_type)

    if args.run_agents:
        logging.info(f"------Running agents for {args.task}------")

        print(f"Running agents for {args.task}...")
        if args.task == "alfworld":
            run_alfworld_webshop(args, feedback_dir)
        elif args.task == "hotpotqa":
            run_hotpotqa(args, feedback_dir, actor_flies_dir)
        elif args.task == 'webshop':
            run_alfworld_webshop(args, feedback_dir)


    last_rejected_files = []
    # associated with different evaluators
    last_rejected_path = os.path.join(
            eval_dir.replace(f"retrial_{args.trial_num}", f"retrial_{max(args.trial_num - 1, 0)}"),
            "eval_metrics",
            args.feedback_type if args.trial_num > 1 else "init",
            f"predicted_pos{args.threshold}.json"
        )
    if args.trial_num > 0:    
        with open(last_rejected_path, "r") as f:
            last_rejected = json.load(f)

        if args.eval_method == "multi_step":
            last_rejected = last_rejected['prod']

        for f in last_rejected:
            if f['true_label'] == 'Incorrect':
                last_rejected_files.append((f['env_name']))
    else:
        last_rejected_files = os.listdir(actor_flies_dir)
        last_rejected_files = sorted(last_rejected_files)

    if args.do_eval:
        logging.info(f"-----Running evaluation for {args.task}------")
        run_evaluator(args, evaluator, actor_flies_dir, last_rejected_files, eval_dir, kwargs)
        # calculate F1 score, auc-pr
        run_metrics(args, evaluators[args.eval_method], eval_dir)

    if args.do_feedback_gen:
        # generate nl feedback for rejected envs
        logging.info(f"-----Generating {args.feedback_type} feedback for next trial for {args.task}------")
        kwargs = {"feedback_dir": feedback_dir, "last_rejected_path": last_rejected_path, "actor_traj_dir": actor_flies_dir, "trial_num": args.trial_num}
        
        if args.task == "alfworld":
            kwargs["expert_traj_dir"] = "./actor/alfworld/expert_traj"

        feedback_generator = FeedbackGenerator(task=args.task, feedback_type=args.feedback_type, eval_method=args.eval_method, threshold=args.threshold, **kwargs)
        feedback_generator.generate_feedback()