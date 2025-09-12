# Preemptive Detection and Correction of Misaligned Actions in LLM Agents
[![Arxiv](https://img.shields.io/badge/Arxiv-2407.11843-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.11843)
[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This repository implements the preemptive evaluation approach, InferAct, for LLM agents, as described in [Preemptive Detection and Correction of Misaligned Actions in LLM Agents](https://arxiv.org/abs/2407.11843) 

> **Abstract**: Deploying LLM-based agents in real-life applications often faces a critical challenge: the misalignment between agents’ behavior and user intent. Such misalignment may lead agents to unintentionally execute some critical actions that carry negative outcomes (e.g., accidentally triggering a \textit{`buy-now'} in web shopping), resulting in undesirable or even irreversible consequences.
Although addressing these issues is crucial, the preemptive detection and correction of misaligned actions remains relatively underexplored.
To fill this gap, we introduce \texttt{InferAct}, a novel approach that leverages the belief reasoning ability of LLMs, grounded in Theory-of-Mind, to detect misaligned actions \textit{before execution}.
Once the misalignment is detected, \texttt{InferAct} alerts users for timely correction, preventing adverse outcomes and enhancing the reliability of LLM agents' decision-making processes.
Experiments on three widely used tasks demonstrate \texttt{InferAct} achieves up to 20\% improvements on Marco-F1 against baselines in misaligned action detection. An in-depth evaluation of misalignment correction further highlights \texttt{InferAct}'s effectiveness in improving agent alignment.

Contact person: [Haishuo Fang](mailto:haishuo.fang@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


![InferAct](./inferact_arch.jpg "Workflow of InferAct")


## 🚀 Setup
```sh
> python -m venv .inferact
> source ./.inferact/bin/activate
> pip install -r requirements.txt
```

### WebShop
- Install openjdk in the virtual environment.
```python
import jdk
from jdk.enums import OperatingSystem, Architecture

jdk.install('11', operating_system=OperatingSystem.LINUX)
import os
jdk_version = 'jdk-11.0.19+7' #change with your version
os.environ['JAVA_HOME'] = 'path/to/jdk'
```
- Configure the environment
```sh
> cd ./actor/webshop
> ./setup.sh -d all
```
### ALFWorld
- Download env data

Please refer to [ALFWorld](https://github.com/alfworld/alfworld)
```sh
export ALFWORLD_DATA="path/to/data"
```

## 🛠️ Usage

### Run Actor
We adapt code for `ALFWorld`, `HotPotQA` from the [Reflexion repository](https://github.com/noahshinn/reflexion)


The Actor agent is responsible for performing tasks in environments. `--run_agents` controls whether to run actor in different environments e.g. `--task webshop`.

```python
python main.py 
    --run_agents 
    --task webshop 
    --trial_num 0
    --feedback_type nl
    --num_envs 300
```

### Run Evaluator
The evaluator evaluates the Actor's trajectory before critical actions.

```python
python main.py 
    --do_eval
    --task webshop
    --eval_method inferact
    --trial_num 0
    --model_name gpt4-turbo
    --feedback_type nl
    --threshold 0.9
```

- `--eval_method` specifies different evaluation methods.<br>
- `--threshold` specifies the threshold of F1-score for `multi-step evaluation` and `inferact`.<br>
- `--do_eval` controls whether to evaluate the Actor trajectory.<br>

### Run Feedback Generation

After the off-track trajectory is detected by the Evaluator, the binary or NL feedback will be generated to prevent the critial action from executing.

```python
python main.py
    --do_feedback_gen
    --task webshop
    --eval_method inferact
    --trial_num 0
    --model_name gpt4-turbo
    --threshold 0.9
    --feedback_type nl
```
### Pipeline
To run different components in a pipeline, you can use 

```python
python main.py 
    --run_agents
    --do_eval
    --do_feedback_gen
    --task webshop
    --model_name gpt35-turbo
    --num_envs 300
    --eval_method standard
    --trial_num 0
    --threshold 0.0
    --feedback_type nl
```

## Cite

Please use the following citation:

```
@article{fang2024preemptive,
  title={Preemptive detection and correction of misaligned actions in llm agents},
  author={Fang, Haishuo and Zhu, Xiaodan and Gurevych, Iryna},
  journal={arXiv preprint arXiv:2407.11843},
  year={2024}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
