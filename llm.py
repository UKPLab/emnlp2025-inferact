from typing import Union, Literal
import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ipdb
import torch
from langchain.memory import ChatMessageHistory
import numpy as np
# from vllm import LLM, SamplingParams


class AnyOpenAILLM:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "gpt4-turbo")

        deployment = kwargs.get("deployment", "gpt4-turbo-128k")
        os.environ["AZURE_OPENAI_API_KEY"] = "KEY"
        os.environ["AZURE_OPENAI_ENDPOINT"] = (
            "endpoint"
        )
        
        self.model = AzureChatOpenAI(
            openai_api_version="2023-05-15", azure_deployment=deployment, **kwargs['model_kwargs'], logprobs=1
        )
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(self, chat_history: list):

        with get_openai_callback() as cb:
            output = self.model(
                chat_history
                # [
                #     HumanMessage(
                #         content=prompt,
                #     )
                # ]
            ).content
        self.prompt_tokens += cb.prompt_tokens
        self.completion_tokens += cb.completion_tokens
        return output


class LocalLLM:
    def __init__(
        self,
        model_pth,
        tokenizer_pth,
        device="cuda:0",
        max_batch_size=1,
        max_new_tokens=None,
        temperature = 0.0,
    ) -> None:
        self.model_pth = model_pth
        self.model = AutoModelForCausalLM.from_pretrained(model_pth,torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pth)
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.temperature = temperature

    def __call__(self, chat_history, probs=False):
        terminators = [self.tokenizer.eos_token_id]
        if "llama-3" in self.model_pth:
            terminators += [self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id = terminators,
            do_sample = False if self.temperature == 0.0 else True)
        
        # chat_history
        messages = []
        for ix, msg in enumerate(chat_history):
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt").to(self.device)

        with torch.inference_mode():
            generation_output = self.model.generate(
                input_ids,
                generation_config=generation_config,
                output_scores=False,
                return_dict_in_generate=True,
            )

        # transition_scores = self.model.compute_transition_scores(
        #     generation_output.sequences, generation_output.scores, normalize_logits=True
        # )
        input_length = input_ids.shape[1]
        generated_ids = generation_output.sequences[:, input_length:]

        token_probs = {}
        if probs:
            logits = generation_output.scores[:, -1, :]
            # for tok, score in zip(generated_ids[0], transition_scores[0]):
            #     token_probs[self.tokenizer.decode(tok)] = np.exp(score.item())

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if probs:
            return decoded[0].strip(), logits
        else:
            return decoded_text
        