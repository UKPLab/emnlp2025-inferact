web_k_task_inference = """You have a powerful Theory-of-Mind capability. An agent is helping the user to shop online. I will give you the sequence of actions the agent takes and corresponding observations. You need to infer user's instruction based on the agent's actions and observations.
To help you understand the style of user's instructions better, here are some examples:
1. I need a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars.
2. I need to buy a ready to hang art print that's sixteen by twenty-four inches. look for one that has women and palm leaves on it, and price lower than 100.00 dollars.
3. i am looking for a pack of 5 dark blonde hair dye touch up spray, and price lower than 110.00 dollars.

Please follow the above style to infer the {num_tasks} most likely user's instructions. Remember your inferred instructions should be as diverse as possible and semantically different from each other.
Your response MUST use the following format:
The {num_tasks} most likely user's instructions are: <A. instruction1\nB. instruction2\nC. instruction3 ...>.
The reason is: <the reason you think>.

The sequence of actions the agent takes is {action}."""

web_task_validator = """You are a powerful judge of agent-based web shopping. An agent, Actor, is helping the user to shop online.
I will give you the trajectory performed by Actor and a set of candidate user's instructions. You need to select your top {num} guesses and carefully assign a probability (ranging from 0.0 to 1.0) to each, indicating the likelihood that the candidate instruction is fulfilled by the Actor's trajectory.
Your response MUST follow the format:
G1: <Only output the option label of the instruction that you think is correct. No other words or explanation> P1: <the probability of the instruction being correct> ... G_i: <Only output the option label of the instruction that you think is correct. No other words or explanation> P_i: <the probability of the instruction being correct>
Justification: <A brief justification for your response>.
Remember, Only evaluate if criteria that are explicitly mentioned in the instruction are met or not. If some features of selected products are not specified in the instruction, you should not consider them in your judgement.

The trajectory performed by Actor is {action}.
The candidate user's instructions are {instructions}."""

web_task_validator = """You are a powerful judge of agent-based web shopping. An agent, Actor, is helping the user to shop online.
I will give you the trajectory performed by Actor and a set of candidate user's instructions. You need to select your top {num} guesses and carefully assign a probability (ranging from 0.0 to 1.0) to each, indicating the likelihood that the candidate instruction is fulfilled by the Actor's trajectory.
Your response MUST follow the format:
G1: <Only output the option label of the instruction that you think is correct. No other words or explanation> P1: <the probability of the instruction being correct> ... G_i: <Only output the option label of the instruction that you think is correct. No other words or explanation> P_i: <the probability of the instruction being correct>
Justification: <A brief justification for your response>.
Remember, Only evaluate if criteria that are explicitly mentioned in the instruction are met or not. If some features of selected products are not specified in the instruction, you should not consider them in your judgement.

The trajectory performed by Actor is {action}.
The candidate user's instructions are {instructions}."""

web_task_validator_desc_term = """You are a powerful judge of agent-based web shopping. An agent, Actor, is helping the user to shop online.
I will give you the trajectory performed by Actor and a set of candidate user's instructions. You need to select your top {num} guesses and carefully assign a likelihood level to each, indicating the likelihood that the candidate instruction is fulfilled by the Actor's trajectory.
Your response MUST follow the format:
G1: <Only output the option label of the instruction that you think is correct. No other words or explanation> P1: <the likelihood level of the instruction being correct> ... G_i: <Only output the option label of the instruction that you think is correct. No other words or explanation> P_i: <the likelihood level of the instruction being correct>
Justification: <A brief justification for your response>.
Remember, Only evaluate if criteria that are explicitly mentioned in the instruction are met or not. If some features of selected products are not specified in the instruction, you should not consider them in your judgement.

The likelihood levels are:
- Very Likely
- Likely
- Possible
- Unlikely
- Very Unlikely

The trajectory performed by Actor is {action}.
The candidate user's instructions are {instructions}"""

web_task_validator_desc_term_logits = """You are a powerful judge of agent-based web shopping. An agent, Actor, is helping the user to shop online.
I will give you the trajectory performed by Actor and a set of candidate user's instructions. You need to select your top {num} guesses and carefully assign a likelihood level to each, indicating how likely it is that the candidate instruction is fulfilled by the Actor's trajectory.

The likelihood levels are:
a. Very Unlikely
b. Unlikely
c. Possible
d. Likely
e. Very Likely

Your response MUST follow the format:
G1: <Only output the option label of the instruction that you think is correct. No other words or explanation> P1: <the likelihood level (e.g. a) of the instruction being correct>\n...\nG_i: <Only output the option label of the instruction that you think is correct. No other words or explanation> P_i: <the likelihood level (e.g. a) of the instruction being correct>
Justification: <A brief justification for your response>.
Remember, Only evaluate if criteria that are explicitly mentioned in the instruction are met or not. If some features of selected products are not specified in the instruction, you should not consider them in your judgement.

The trajectory performed by Actor is {action}.
The candidate user's instructions are {instructions}"""

# - Certain
# - Likely
# - Possible
# - Unlikely
# - Improbable
# web_task_validator_desc_term_logits = """You are a powerful judge of agent-based web shopping. An agent, Actor, is helping the user to shop online.
# I will give you the trajectory performed by Actor and a set of candidate user's instructions. You need to select your top {num} guesses and carefully assign a likelihood level to each, indicating the likelihood that the candidate instruction is fulfilled by the Actor's trajectory.

# The likelihood levels are:
# a. Very Unlikely
# b. Unlikely
# c. Possible
# d. Likely
# e. Very Likely

# Your response MUST follow the format:
# G1: <Only output the option label of the instruction that you think is correct. No other words or explanation> P1: <The likelihood level (e.g. a) of the instruction being correct>\n...\nG_i: <Only output the option of the instruction that you think is correct. No other words or explanation> P_i: <The likelihood level of the instruction being correct>\nJustification: <A brief justification for your guesses>.

# Remember, Only evaluate if criteria that are explicitly mentioned in the instruction are met or not. If some features of selected products are not specified in the instruction, you should not consider them in your judgement.
# The trajectory performed by Actor is {action}.
# The candidate user's instructions are {instructions}"""




web_task_validator_multilabel = """You are a powerful judge of agent-based web shopping. An agent, Actor, is helping the user to shop online.
I will give you the trajectory performed by Actor and a set of candidate user's instructions. You need to identify whether each candidate instruction is fulfilled by the Actor's trajectory.
Your response MUST follow the format:
A: <Yes/No> B: <Yes/No> ... Z: <Yes/No>.

The trajectory performed by Actor is {action}.
The candidate user's instructions are {instructions}."""

web_task_validator_true_false = """You are a powerful judge of agent-based web shopping. An agent, "Actor", is helping the user to shop online.

The trajectory performed by Actor is {action}.
Here are some possible user's instructions that the Actor might have fulfilled:
{instructions}.
The selected instruction is: {gold_instruction}
Is the selected instruction correctly fullfilled by the Actor's trajectory?
A. Yes
B. No
Please directly output A or B as your answer without any additional words or explanation."""


web_task_validator_multiclass = """You are a powerful judge of agent-based web shopping. An agent, Actor, is helping the user to shop online.
The trajecotry performed by Actor is {action}.
The candidate user's instructions are {instructions}.
Given the above candidates, which instruction do you think the Actor is trying to fulfill? You response MUST follow the format:
The correct instruction is: <A/B/C/D/E>."""

web_task_validator_risk_sen = """You are a powerful judge of agent-based web shopping. An agent, Actor, is helping the user to shop online.
I will give you the trajectory performed by Actor and a set of candidate user's instructions. You need to select your top {num} guesses and assign a probability (ranging from 0.0 to 1.0) to each, indicating the likelihood that the candidate instruction is fulfilled by the Actor's trajectory.
Pay special attention to the prices of the products involved. Mistakes in higher-priced products can lead to significant financial loss for the user. Therefore, when the price is higher than $60, your assessment must be especially stringent and risk-aware.

Your response MUST follow the format:
G1: <Only output the option label of the instruction that you think is correct. No other words or explanation> P1: <the probability of the instruction being correct. Consider financial risk of a mistake> ... G_i: <Only output the option label of the instruction that you think is correct. No other words or explanation> P_i: <the probability of the instruction being correct. Consider financial risk of a mistake>
Justification: <A brief justification for your response.>.
Remember, Only evaluate if criteria that are explicitly mentioned in the instruction are met or not. If some features of selected products are not specified in the instruction, you should not consider them in your judgement.
Additionally, prioritize accuracy for higher-priced items to prevent potential financial loss to the user.

The trajectory performed by Actor is {action}.
The candidate user's instructions are {instructions}."""


web_task_validator_prob_with_justification_prompt = """You are a powerful judge of agent-based web shopping. An agent, Actor, is helping the user to shop online.
I will give you the trajectory performed by Actor and a set of candidate user's instructions with corresponding justifications. You need to select your top {num} guesses and assign a probability (ranging from 0.0 to 1.0) to each, indicating the likelihood that the candidate instruction is fulfilled by the Actor's trajectory.
Your reponse should follow the format:
G1: <Simply output the number of the question that you think is correct> P1: <the probability of the question being correct> ... G_i: <Simply output the number of the question that you think is correct> P_i: <the probability of the question being correct>
Justification: <A brief justification for your response. It should no more than six sentences.>.
Remember, Only evaluate if criteria that are explicitly mentioned in the instruction are met or not. If some features of selected products are not specified in the instruction, you should not consider them in your judgement.

The trajectory performed by Actor is {action}.
The candidate user's instructions are {instructions}.
The justifications for the candidate instructions are {justifications}."""



web_standard_eval = """You will be given the reasoning trajectory you perfomed in a shopping website for a given user's instruction.
Your task is to evaluate the reasoning trajectory and determine how likely it fulfilled the user's instruction. You need to assign a probability (ranging from 0.0 to 1.0) to your response, indicating the likelihood that the reasoning trajectory is correct.
Your response MUST follow the format:
The answer is: <Correct/Incorrect>
Confidence: <A probability ranging from 0.0 to 1.0>
Justification: <A brief justification for your response>

The instruction is: {instruction}
The reasoning trajectory is {trajectory}"""

web_standard_eval_risk_sen = """You will be given the reasoning trajectory you perfomed in a shopping website for a given user's instruction.
Your task is to evaluate the reasoning trajectory and determine how likely it fulfilled the user's instruction. You need to assign a probability (ranging from 0.0 to 1.0) to your response, indicating the likelihood that the reasoning trajectory is correct.
Pay special attention to the prices of the products involved. Mistakes in higher-priced products can lead to significant financial loss for the user. Therefore, when the price is higher than $60, your assessment must be especially stringent and risk-aware.
Your response MUST follow the format:
The answer is: <Correct/Incorrect>
Confidence: <A probability ranging from 0.0 to 1.0>
Justification: <A brief justification for your response>

The instruction is: {instruction}
The reasoning trajectory is {trajectory}"""

justification_generation_prompt = """An agent, Actor, is helping the user to shop online. I will give you reasoning trajectory performed by Actor and the user's instruction it fulfilled. You need to generate a brief justification demonstrating how the instruction is fulfilled by the reasoning trajectory. Please note that your justification should no more than six sentences.
The reasoning trajectory is: {trajectory}
The instruction it fulfilled is: {instructions}"""

web_multi_step = """You will be given the reasoning trajectory you performed in a shopping website for a given user's instruction. Your task is to evaluate the reasoning trajectory step by step and determine how likely each step is correct. 
Each step has three parts: Thought, Action, and Observation. You need to assign a probability (ranging from 0.0 to 1.0) to each step, indicating the likelihood that the step is correct.

Your response MUST follow the format:
Step 1: <A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step 1 is correct>\nStep 2:<A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step 2 is correct>\n ... \nStep i: <A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step i is correct>
Justification: <A brief justification for your response. No more than six sentences.>

The instruction is: {instruction}
The reasoning trajectory is {trajectory}"""

web_multi_step_risk_sen = """You will be given the reasoning trajectory you performed in a shopping website for a given user's instruction. Your task is to evaluate the reasoning trajectory step by step and determine how likely each step is correct. 
Each step has three parts: Thought, Action, and Observation. You need to assign a probability (ranging from 0.0 to 1.0) to each step, indicating the likelihood that the step is correct.
Pay special attention to the prices of the products involved. Mistakes in higher-priced products can lead to significant financial loss for the user. Therefore, when the price is higher than $60, your assessment must be especially stringent and risk-aware.

Your response MUST follow the format:
Step 1: <A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step 1 is correct>\nStep 2:<A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step 2 is correct>\n ... \nStep i: <A Probability ranging from 0.0 to 1.0 to indicate the likelihood that the step i is correct>
Justification: <A brief justification for your response. No more than six sentences.>

The instruction is: {instruction}
The reasoning trajectory is {trajectory}"""

task_inference_prompt = """
An agent is helping the user to shop online. I will give you the sequence of actions the agent takes and corresponding observations. You need to guess what the user is looking for based on the agent's actions and observations.
You should guess a type of products with feature descriptions (e.g. a long lasting 6.76 fl oz bottle of l\'eau d\'issey, and price lower than 100.00 dollars). Please use the following format to provide your answer:
Based on Actor's trajectory, I guess the user want to buy a product that <detailed feature description inlcuding size, color, price if possible>. The reason is: <the reason you think>. 
The sequence of actions the agent takes is {action}. 
"""

ws_feedback_prompt = """An Actor agent is helping the user shop online. I will give you the user's instruction, the desired product that the user is looking for, and the incorrect action chain performed by the Actor agent. 
You need to imagine that you are the user and provide feedback to help the Actor agent fulfill your instruction. Your feedback should be constructive and specific. Please provide your feedback in the following format:
Feedback: <Your feedback to help the Actor agent fulfill the user's instruction. It should be clear, concise, and no more than five sentences.>
Your (the user's) instruction is: {task}
The desired product that the user is looking for is: {gold_label_actor}
The incorrect action chain is: {incorrect_action_chain}"""

ws_afterwards_feedback_prompt = """An Actor agent is helping the user shop online. You've already provided feedback to help fulfill the user's instruction. However, the Actor agent still failed. I will provide you with the user's instruction, the desired product that the user is looking for, the incorrect action chain performed by the Actor agent, and the feedback you provided before. 
You need to imagine that you are the user and provide further feedback to help the Actor agent fulfill your instruction. Your feedback should be constructive and specific. Please provide your feedback in the following format:
Feedback: <Your feedback to help the Actor agent fulfill your instruction. It should be clear, concise, and no more than five sentences.>

Your (the user's) instruction is: {task}
The desired product that the user is looking for is: {gold_label_actor}
The incorrect action chain is: {incorrect_action_chain}
The feedback(s) you provided before are: {previous_feedback}"""

ws_binary_feedback_prompt = """You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan (no more than five sentences) of action that accounts for your mistake with reference to specific actions that you should have taken.  There are two examples below.
{Examples}


Instruction: {scenario}"""




