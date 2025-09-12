from evaluator import (
    StandardEvaluator,
    MultistepEvaluator,
    StandardEvaluatorSC,
    InferAct,
)
import argparse
import os
import json
from evaluator.utils_eval import (
    convert_json_objs,
    get_inferred_tasks,
    load_rejected_envs,
    load_halted_envs,
)
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from netcal.metrics import ECE
import numpy as np
from feedback_generator import FeedbackGenerator
from actor.alfworld.alfworld_trial import run_alfworld
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "actor/webshop/")
)
# for path in sys.path:
#     print(path)
from actor.webshop.webshop_trial import run_webshop
import sys
from actor.hotpotqa.agents import ReactAgent
import logging
import ipdb


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
)


def run_alfworld_webshop(args, feedback_dir=None):

    # initialize environment configs
    env_configs = []
    for i in range(args.num_envs):
        if args.trial_num == 0:
            env_configs += [{"name": f"env_{i}", "memory": [], "is_skip": False}]
        else:
            if os.path.exists(os.path.join(feedback_dir, f"{i}.txt")):
                with open(os.path.join(feedback_dir, f"{i}.txt"), "r") as f:
                    feedback = f.readlines()

                # this indicates that the task is solved
                if len(feedback) == args.trial_num:
                    env_configs += [
                        {
                            "name": f"env_{i}",
                            "memory": [fed.strip() for fed in feedback],
                            "is_skip": False,
                        }
                    ]

                else:
                    env_configs += [{"name": f"env_{i}", "memory": [], "is_skip": True}]

            else:
                env_configs += [{"name": f"env_{i}", "memory": [], "is_skip": True}]

    # run trials
    trial_log_dir: str = os.path.join(
        args.traj_dir,
        args.task,
        f"retrial_{args.trial_num}",
        args.feedback_type if int(args.trial_num) > 0 else "",
    )

    print(
        f"""
    -----
    Starting run with the following parameters:

    Number of environments: {args.num_envs}

    Sending trajectories to `{trial_log_dir}`
    -----
    """
    )

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
    env2data = {row["env_name"]: row for row in data}

    if not os.path.exists(actor_flies_dir):
        os.makedirs(actor_flies_dir)

    if args.trial_num == 0:
        # generate feedback for the first trial
        agents = [ReactAgent(row["question"], row["answer"], []) for row in data]
    else:
        agents = []
        # load feedback
        feedback_files = os.listdir(feedback_dir)
        # actor files
        actor_files = os.listdir(actor_flies_dir)

        for file in feedback_files:
            with open(os.path.join(feedback_dir, file), "r") as f:
                feedback = f.readlines()
            env_name = file.replace(".txt", "")
            if (
                len(feedback) == args.trial_num
                and not file.replace(".txt", ".json") in actor_files
            ):
                print("Run actor for ", file)
                question = env2data[env_name]["question"]
                answer = env2data[env_name]["answer"]
                agents.append((env_name, ReactAgent(question, answer, feedback)))
            else:
                print(
                    f"{file} already executed in trial {args.trial_num} or the number of feedback is not {len(feedback)}"
                )

    for env_name, agent in tqdm([a for a in agents if not a[1].is_correct()]):
        agent.run()
        with open(os.path.join(actor_flies_dir, f"{env_name}.json"), "w") as f:
            json.dump(agent.output_traj(), f, ensure_ascii=False, indent=4)


def load_rjudge(actor_traj_dir):
    valid_cases = []
    for root, dirs, files in os.walk(actor_traj_dir):
        for name in files:
            if os.path.splitext(name)[1] == ".json":
                with open(os.path.join(root, name), "r") as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        pass
                    else:
                        for example in data:
                            # label align with other tasks
                            valid_cases.append((example, 1 - example["label"]))

    return valid_cases


# load existing evaluated data, load data from actor trajectories to evaluate
def run_evaluator(
    args, evaluator, actor_traj_dir, last_rejected_files, eval_dir, kwargs
):

    # path to save eval results
    eval_result_dir = os.path.join(eval_dir, "eval_results")
    if not os.path.exists(eval_result_dir):
        os.makedirs(eval_result_dir)

    # load evaluated data
    existing_envs = []
    evaluated_file = os.path.join(
        eval_result_dir,
        f"{args.feedback_type}.txt" if args.trial_num > 0 else "init.txt",
    )
    if os.path.exists(evaluated_file):
        entries = convert_json_objs(evaluated_file)
        existing_envs = [entry["env_name"] + ".json" for entry in entries]

    if args.reuse_belief:
        env2inferred_tasks = get_inferred_tasks(args.reused_file)

    f_file = open(evaluated_file, "a")
    # if args.task == "rjudge":
    #     messages = load_rjudge(actor_traj_dir)
    # else:
    # last_rejected_files = [
    #     "100",
    #     "101",
    #     "102",
    #     "103",
    #     "104"]

    halted_files = []
    for file in tqdm(last_rejected_files):
        if not file.endswith(".json"):
            file = file + ".json"
        try:
            with open(os.path.join(actor_flies_dir, file), "r") as f:
                data = json.load(f)
        except:
            print("failed to load ", file)
            continue

        is_halted = data.get("is_halted", False)
        if is_halted:
            print(f"{file} is halted")
            halted_files.append(file.split(".")[0])
            continue

        if file in existing_envs:
            print(f"{file} is already evaluated")
            continue
        kwargs["env_name"] = file.split(".")[0]

        if "config" in file:
            continue
        if not os.path.exists(os.path.join(actor_traj_dir, file)):
            logging.info(
                f"Evaluating the trajectory, but there is no {os.path.join(actor_traj_dir, file)}"
            )
            continue

        # using the belief
        if args.reuse_belief:
            kwargs["inferred_tasks"] = env2inferred_tasks[kwargs["env_name"]]
        trajectory = data["path"] if args.task != "rjudge" else data["contents"][0]
        output = evaluator.evaluate(trajectory, **kwargs)
        if not output:
            continue
        # add into a json object
        output["trace_correct"] = data["trace_correct"] if args.task != "rjudge" else 1 - data["label"]
        if args.task == "rjudge":
            output["explanation"] = data["risk_description"]
            
        if args.task == "webshop":
            output["true_item"] = data["true_item"]

        f_file.write(json.dumps(output, indent=4) + "\n")
        f_file.flush()
    f_file.close()
    return halted_files


def search_best_f1(precisions, recalls, thresholds):
    best_f1 = 0
    best_threshold = 0
    for i in range(len(precisions)):
        f1 = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i] + 1e-7)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresholds[i]
    return best_f1, best_threshold


def search_lowest_cost(precisions, recalls, thresholds, positive_count, negative_count):
    lowest_cost = 1e10
    best_threshold = 0
    for ix in range(len(thresholds)):
        recall = recalls[ix]
        precision = precisions[ix]
        if recall == 0 and precision == 0:
            continue
        # try:
        #     f1 = 2 * precision * recall / (precision + recall)
        # except:
        #     ipdb.set_trace()

        true_positive = round(recall * positive_count)
        false_negative = positive_count - true_positive
        false_positive = true_positive / precision - true_positive
        true_negative = negative_count - false_positive
        cost = 2 * false_negative + false_positive
        if cost < lowest_cost:
            lowest_cost = cost
            best_threshold = thresholds[ix]

    return lowest_cost, best_threshold


def search_best_macro_f1(
    precisions, recalls, thresholds, positive_count, negative_count
):
    best_macro_f1 = 0
    best_threshold = []
    for ix in range(len(thresholds)):
        recall = recalls[ix]
        precision = precisions[ix]
        if recall == 0 and precision == 0:
            continue
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except:
            ipdb.set_trace()

        true_positive = round(recall * positive_count)
        false_negative = positive_count - true_positive
        false_positive = true_positive / precision - true_positive
        true_negative = negative_count - false_positive
        # precision for negative class
        precision_neg = true_negative / (true_negative + false_negative + 1e-10)
        # recall for negative class
        recall_neg = true_negative / (true_negative + false_positive + 1e-10)
        # # f1 for negative class
        f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg + 1e-10)
        # macro f1
        macro_f1 = (f1 + f1_neg) / 2

        if macro_f1 >= best_macro_f1:
            if macro_f1 == best_macro_f1:
                best_threshold.append(thresholds[ix])
            else:
                best_threshold = [thresholds[ix]]
            best_macro_f1 = macro_f1
            best_negative_f1 = f1_neg
            best_positive_f1 = f1
            cost_best_f1 = false_negative + false_positive

    return (
        best_macro_f1,
        best_threshold,
        best_positive_f1,
        best_negative_f1,
        cost_best_f1,
    )
    # best_true_positive, best_false_negative, best_false_positive, best_true_negative


def cal_metrics(eval_result_file, kwargs, evaluator, last_rejected_files):
    json_objects = convert_json_objs(eval_result_file)
    json_objects = [
        entry for entry in json_objects if entry["env_name"] in last_rejected_files
    ]

    output, tp_tasks, predicted_pos = evaluator.metric(json_objects, **kwargs)

    # true_positive = output["true_positive"]
    # true_negative = output["true_negative"]
    # false_positive = output["false_positive"]
    # false_negative = output["false_negative"]
    # false_negative_env = output.get('false_negative_env', [])
    y_true = output.get("y_true", [])
    y_pred = output.get("y_pred", [])
    # cost = output["cost"]

    # epsilon = 1e-17
    # recall = true_positive / (true_positive + false_negative + epsilon)
    # precision = true_positive / (true_positive + false_positive + epsilon)
    # f1 = 2 * recall * precision / (recall + precision + 1e-7)
    # neg_recall = true_negative / (true_negative + false_positive + epsilon)
    # neg_precision = true_negative / (true_negative + false_negative + epsilon)
    # neg_f1 = 2 * neg_recall * neg_precision / (neg_recall + neg_precision + epsilon)
    # macro_f1 = (f1 + neg_f1) / 2
    same = all(x == 0 for x in y_true) or all(x == 1 for x in y_true)
    if y_pred and any(y_pred) and not same:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
        # best_f1, best_threshold = search_best_f1(precisions, recalls, thresholds)
        # lowest_cost, best_threshold = search_lowest_cost(precisions, recalls, thresholds, sum(y_true), len(y_true) - sum(y_true))

        # best_macro_f1, best_threshold, best_positive_f1, best_negative_f1, cost_best_f1 = search_best_macro_f1(precisions, recalls, thresholds, sum(y_true), len(y_true) - sum(y_true))

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
    # balanced_acc = (neg_precision + precision) / 2
    # accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    res = {
        "auc_pr": auc_pr,
        "ece": ece_score,
        "auc_roc": auc_roc,
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": thresholds,
        "positive_count": sum(y_true),
        "negative_count": len(y_true) - sum(y_true),
        "total_count": len(y_true),
    }
    output.update(res)
    return output, tp_tasks, predicted_pos


def run_metrics(args, evaluator, eval_dir, last_rejected_files):
    eval_result_file = os.path.join(
        eval_dir,
        "eval_results",
        f"{args.feedback_type}.txt" if args.trial_num > 0 else "init.txt",
    )

    kwargs = {
        "threshold": args.threshold,
        "task": args.task,
        "prob_estimation": args.prob_estimation,
        "agg_method": args.agg_method,
        "threshold_search": args.threshold_search,
    }
    # ipdb.set_trace()
    if args.eval_method == "multi_step":
        result = {}
        tp_tasks = {}
        pred_pos = {}
        for func in [np.prod, np.max, np.mean, np.min]:
            kwargs["aggregated_func"] = func
            result[func.__name__], tp_tasks[func.__name__], pred_pos[func.__name__] = (
                cal_metrics(eval_result_file, kwargs, evaluator, last_rejected_files)
            )
    else:
        result, tp_tasks, pred_pos = cal_metrics(
            eval_result_file, kwargs, evaluator, last_rejected_files
        )

    if result.get("optimal_threshold", None):
        threshold = result["optimal_threshold"]
    else:
        threshold = kwargs.get("threshold", 0.5)
    with open(
        os.path.join(eval_metric_dir, f"predicted_pos{threshold}.json"), "w"
    ) as f:
        json.dump(pred_pos, f, indent=4)

    with open(os.path.join(eval_metric_dir, f"metrics{threshold}.json"), "w") as f:
        json.dump(result, f, indent=4)

    with open(os.path.join(eval_metric_dir, f"tp_tasks{threshold}.json"), "w") as f:
        json.dump(tp_tasks, f, indent=4)

    with open(
        os.path.join(eval_metric_dir, f"predicted_pos{threshold}.json"), "w"
    ) as f:
        json.dump(pred_pos, f, indent=4)


if __name__ == "__main__":
    # add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt4-turbo",
        help="model name, gpt4-turbo, gpt35-turbo",
    )
    parser.add_argument(
        "--model_path", type=str, default="", help="path to the local model"
    )
    parser.add_argument(
        "--eval_method",
        type=str,
        default="standard",
        help="standard, multi_step, standard_sc, inferact",
    )
    parser.add_argument("--reuse_belief", action="store_true", help="reuse belief")
    parser.add_argument(
        "--reused_file",
        type=str,
        default="",
        help="path to the previous file with inferred tasks",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/evaluation_results",
    )
    parser.add_argument(
        "--feedback_type",
        type=str,
        default="none",
        help="results with different feedback types, nl, binary",
    )
    parser.add_argument("--task", type=str, default="", required=True)
    parser.add_argument(
        "--trial_num", type=int, default=0, help="the number of current existing trials"
    )
    parser.add_argument("--risk_mode", action="store_true", help="sensitive mode")
    parser.add_argument("--signature", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--traj_dir", type=str, default="./outputs/actor-traj")
    parser.add_argument("--prob_estimation", type=str, default="")
    parser.add_argument("--agg_method", type=str, default="")
    parser.add_argument("--threshold_search", action="store_true", default="")

    ## args for Alfworld
    parser.add_argument(
        "--num_envs",
        type=int,
        help="The number of environments per trial in Alfworld or webshop",
    )
    parser.add_argument(
        "--use_memory", action="store_true", help="Allow the Agent to use memory"
    )
    ## actions
    parser.add_argument("--run_agents", action="store_true", help="run agents")
    parser.add_argument(
        "--do_feedback_gen",
        action="store_true",
        help="generate nl feedback for rejected envs",
    )
    parser.add_argument("--do_eval", action="store_true", help="run evaluation")
    parser.add_argument("--do_metric", action="store_true", help="calculate metrics")

    args = parser.parse_args()

    assert args.eval_method in [
        "standard",
        "multi_step",
        "standard_sc",
        "inferact",
    ], "eval_method should be one of standard, multi_step, standard_sc, inferact"

    kwargs = {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "risk_mode": args.risk_mode,
        "prob_estimation": args.prob_estimation,
        "logprobs": True if "logits" in args.prob_estimation else False,
        "agg_method": args.agg_method,
    }

    # logging.getLogger().setLevel(logging.INFO)

    if args.eval_method == "standard_sc":
        kwargs["temperature"] = 0.7
    else:
        kwargs["temperature"] = 0.0

    save_dir = f"{args.save_dir}/{args.model_name}"

    ## All paths
    eval_dir = os.path.join(
        save_dir,
        args.task,
        args.eval_method,
        args.prob_estimation,
        args.signature,
        (
            f"retrial_{args.trial_num}"
            if not args.risk_mode
            else f"retrial_{args.trial_num}_sensitive"
        ),
    )
    if args.task == "rjudge":
        actor_flies_dir = "/storage/ukp/work/fang/InferAct/outputs/actor-traj/R-judge"
    else:
        # actor_flies_dir = '/storage/ukp/work/fang/InferAct/outputs/actor-traj/webshop_1/dev'
        actor_flies_dir = f"{args.traj_dir}/{args.task}/retrial_{args.trial_num}/{args.feedback_type if args.trial_num > 0 else ''}"
        last_actor_files_dir = f"{args.traj_dir}/{args.task}/retrial_{max(args.trial_num - 1, 0)}/{args.feedback_type if args.trial_num > 1 else ''}"

    feedback_dir = os.path.join("./outputs/feedbacks", args.task, args.feedback_type)

    if args.run_agents:
        logging.info(f"------Running agents for {args.task}------")

        print(f"Running agents for {args.task}...")
        if args.task == "alfworld":
            run_alfworld_webshop(args, feedback_dir)
        elif args.task == "hotpotqa":
            run_hotpotqa(args, feedback_dir, actor_flies_dir)
        elif args.task == "webshop":
            run_alfworld_webshop(args, feedback_dir)

    last_rejected_files = []
    # associated with different evaluators
    last_metric_dir = os.path.join(
        eval_dir.replace(
            f"retrial_{args.trial_num}", f"retrial_{max(args.trial_num - 1, 0)}"
        ),
        "eval_metrics",
        args.feedback_type if args.trial_num > 1 else "init",
        args.agg_method,
    )
    last_rejected_path = os.path.join(
        last_metric_dir, f"predicted_pos{args.threshold}.json"
    )

    evaluators = {
        "standard": StandardEvaluator,
        "multi_step": MultistepEvaluator,
        "standard_sc": StandardEvaluatorSC,
        "inferact": InferAct,
    }
    if args.do_eval or args.do_metric:
        if args.trial_num > 0:
            # ipdb.set_trace()
            last_rejected_files = load_rejected_envs(
                args.task, last_rejected_path, args.eval_method
            )
            with open(os.path.join(last_metric_dir, "halted_files.json"), "r") as f:
                last_halted_files = json.load(f)
            # last_halted_files = load_halted_envs(os.path.join(last_metric_dir, "halted_files.json"))
            last_rejected_files += last_halted_files
            # if args.eval_method == "multi_step":
            #     last_rejected = last_rejected["prod"]
            # for f in last_rejected:
            #     if f["true_label"] == "Incorrect":
            #         last_rejected_files.append((f["env_name"]))
        else:
            last_rejected_files = os.listdir(actor_flies_dir)
            last_rejected_files = [f.split(".json")[0] for f in last_rejected_files]
            last_rejected_files = sorted(last_rejected_files)
        # last_rejected_files = [
        #     "289",
        #     "93",
        #     "153",
        #     "243",
        #     "220",
        #     "260",
        #     "194",
        #     "196",
        #     "157",
        #     "204",
        #     "263",
        #     "133",
        #     "226",
        #     "163",
        #     "229",
        #     "274",
        #     "170",
        #     "123",
        #     "143",
        #     "28",
        #     "201",
        #     "60",
        #     "246",
        #     "291",
        #     "209",
        #     "270",
        #     "202",
        #     "85",
        #     "284",
        #     "181",
        #     "219",
        #     "200",
        #     "293",
        #     "253",
        #     "182",
        #     "222",
        #     "70",
        #     "119",
        #     "189",
        #     "159",
        #     "256",
        #     "130",
        #     "72",
        #     "279",
        #     "92",
        #     "41",
        #     "88",
        #     "208",
        #     "83",
        #     "282",
        #     "228",
        #     "277",
        #     "179",
        #     "192",
        #     "13",
        #     "107",
        #     "106",
        #     "98",
        #     "138",
        #     "242",
        #     "99",
        #     "184",
        #     "1",
        #     "156",
        #     "198",
        #     "49",
        #     "54",
        #     "271",
        #     "221",
        #     "127",
        #     "141",
        #     "215",
        #     "231",
        #     "58",
        #     "175",
        #     "295",
        #     "111",
        #     "102",
        #     "269",
        #     "69",
        #     "203",
        #     "26",
        #     "169",
        #     "38",
        #     "158",
        #     "109",
        #     "78",
        #     "23",
        #     "258",
        #     "35",
        #     "236",
        #     "77",
        #     "95",
        #     "64",
        #     "183",
        #     "150",
        #     "188",
        #     "19",
        #     "116",
        #     "214",
        # ]
        evaluator = evaluators[args.eval_method](args.task, **kwargs)
        logging.info(f"-----Running evaluation for {args.task}------")
        # ipdb.set_trace()
        halted_files = run_evaluator(
            args, evaluator, actor_flies_dir, last_rejected_files, eval_dir, kwargs
        )
        eval_metric_dir = os.path.join(
            eval_dir,
            "eval_metrics",
            args.feedback_type if args.trial_num > 0 else "init",
            args.agg_method,
        )
        if not os.path.exists(eval_metric_dir):
            os.makedirs(eval_metric_dir)
        with open(os.path.join(eval_metric_dir, "halted_files.json"), "w") as f:
            json.dump(halted_files, f, indent=4)

    if args.do_metric:
        # calculate F1 score, auc-pr
        run_metrics(args, evaluators[args.eval_method], eval_dir, last_rejected_files)

    if args.do_feedback_gen:
        # generate nl feedback for rejected envs
        logging.info(
            f"-----Generating {args.feedback_type} feedback for next trial for {args.task}------"
        )
        kwargs = {
            "feedback_dir": feedback_dir,
            "last_rejected_path": last_rejected_path,
            "actor_traj_dir": actor_flies_dir,
            "trial_num": args.trial_num,
        }

        if args.task == "alfworld":
            kwargs["expert_traj_dir"] = "./actor/alfworld/expert_traj"

        feedback_generator = FeedbackGenerator(
            task=args.task,
            feedback_type=args.feedback_type,
            eval_method=args.eval_method,
            threshold=args.threshold,
            **kwargs,
        )
        feedback_generator.generate_feedback()
