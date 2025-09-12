
hotpot_k_task_inference_original = """You have a powerful Theory-of-Mind capability, enabling you to infer and interpret intentions. A reasoning agent is searching for an answer to the user's question based on its interpretation. The agent uses the following tools to find the answer: 
(1) Search[entity], which searches the information of the entity on Wikipedia.
(2) Lookup[keyword], which returns the next sentence containing keyword in the wikipedia.
(3) Finish[answer], which returns the answer to the question and finishes the task.

Your task is to deduce the interpreted instruction by observing the agent's behaviors (e.g. actions, observations, the final answer etc).
Your response MUST use the following format: 
The question interpreted by the agent is: <your inferred question>
The reason is: <the reason you think>.

The reasoning trajectory the agent takes is {action}."""

hotpot_k_task_inference = """A reasoning agent is searching for an answer to the user's question based on its interpretation. The agent uses the following tools to find the answer: 
(1) Search[entity], which searches the information of the entity on Wikipedia.
(2) Lookup[keyword], which returns the next sentence containing keyword in the wikipedia.
(3) Finish[answer], which returns the answer to the question and finishes the task.

Your task is to infer the instruction behind the agent's actions (e.g. actions, observations, the final answer etc).
Your response MUST follow the following format: 
The inferred question is: <your inferred instruction>
The reason is: <the reason you think>.

The reasoning trajectory the agent takes is {action}."""



hotpot_task_validator = """An agent, Actor, is searching for the answer to the user's question using some tools. Your task is to evaluate whether the agent gets the correct answer to the user's question.
The reasoning trajectory performed by Actor is: {action}.
The question interpreted by the Actor is: {intended_task}.
The actual question given by the user is: {instruction}.
If the agent answers the above interpreted question, does it entail that the user's question is also answered?
A. True
B. False
The agent answering the above interpreted question implies that the user's question is also answered:<A. True/B.False>"""

hotpot_task_validator_2steps = """An agent, Actor, is searching for answers to user's questions using some tools. 
Does the agent answering the question "{intended_task}" imply that it has also answered the user's question: "{instruction}"? If not, what additional actions should the agent take to get the answer to the user's question?
A. Yes.
B. No. The agent should also <specify what else the agent should do to get the answer to the user's question>."""



hotpot_task_validator_true_false = """The trajectory performed by Actor is: {action}.
Here are some possible user's questions that the Actor might have answered:
{instructions}.
The possible question is fulfilled by the trajectory is: {gold_instruction}
Is the possible question:
A. True
B. False
The possible question is:"""


hotpot_standard_eval = """You will be given the question and the reasoning trajectory you performed to find the answer to the question. Your task is to evaluate whether the reasoning trajectory is correct or not.
Your response MUST follow the format:
The answer is: <Correct/Incorrect>
Justification: <A brief justification for your response>

The question is: {instruction}
The reasoning trajectory is {trajectory}"""

hotpot_prob = """An agent, Actor, is searching for answers to user's questions using some tools. Your task is to evaluate whether the agent finds the correct answer to the question.
The question is: {instruction}
The agent's reasoning trajectory to answer the question is: {trajectory}
Is the reasoning trajectory:
A. True
B. False
The reasoning trajectory is:"""


hotpot_multi_step = """You will be given the reasoning trajectory you performed in a question answering task for a given question. Your task is to evaluate the reasoning trajectory step by step and determine how likely each step is correct. 
Each step has three parts: Thought, Action, and Observation. You need to assign a probability (ranging from 0.0 to 1.0) to each step, indicating the likelihood that the step is correct.

Your response should follow the format:
Step 1: <A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step 1 is correct>\nStep 2:<A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step 2 is correct>\n ... \nStep i: <A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step i is correct>
Justification: <A brief justification for your response. No more than six sentences.>

The instruction is: {instruction}
The reasoning trajectory is {trajectory}"""

hotpot_multi_step_inferact = """You will be given the reasoning trajectory you performed in a question answering task for a given question. Your task is to evaluate the reasoning trajectory step by step and determine how likely each step is correct. 
Each step has three parts: Thought, Action, and Observation. You need to first reflect the intent of each step and then assign a probability (ranging from 0.0 to 1.0) to each step, indicating the likelihood that the step is correct.

Your response MUST follow the format:
Step 1-Intent: <The intent of the step 1>\nStep 1-Probability: <A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step 1 is correct>\nStep 2-Intent: <The intent of the step 2>\nStep 2-Probability: <A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step 2 is correct>\n ... \nStep i-Intent: <The intent of the step i>\nStep i-Probability: <A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step i is correct>
Justification: <A brief justification for your response. No more than six sentences.>

The instruction is: {instruction}
The reasoning trajectory is {trajectory}"""

hot_feedback_prompt = """An Actor agent is answering the user's question using some search tools. I will give you the user's question, the correct answer that the user is looking for, and the incorrect action chain performed by the Actor agent. 
You need to imagine that you are the user and provide feedback to help the Actor agent find the correct answer. Your feedback should be constructive and specific. Please do not directly tell the agent the answer to the question and provide your feedback in the following format:
Feedback: <Your feedback to help the Actor agent find the correct answer. It should be clear, concise, and no more than five sentences.>
Your (the user's) question is: {task}
The correct answer is: {gold_label_actor}
The incorrect action chain is: {incorrect_action_chain}"""

hot_afterwards_feedback_prompt = """An Actor agent is answering the user's question using search tools. You've already provided feedback to help the agent find the correct answer. However, the Actor agent still failed. I will give you the user's question, the correct answer that the user is looking for, and the incorrect action chain performed by the Actor agent. 
You need to imagine that you are the user and provide feedback to help the Actor agent find the correct answer. Your feedback should be constructive and specific. Please do not directly tell the agent the answer to the question and provide your feedback in the following format:
Feedback: <Your feedback to help the Actor agent find the correct answer. It should be clear, concise, and no more than five sentences.>
Your (the user's) question is: {task}
The correct answer is: {gold_label_actor}
The incorrect action chain is: {incorrect_action_chain}
The feedback(s) you provided before are: {previous_feedback}"""

hotpot_binary_feedback_prompt = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Question: {question}{scratchpad}

Reflection:"""