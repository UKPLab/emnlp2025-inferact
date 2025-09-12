from typing import Union, Literal
import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from langchain.memory import ChatMessageHistory
import numpy as np
import ipdb
from langchain.memory import ChatMessageHistory
import transformers
from openai import AzureOpenAI

# from vllm import LLM, SamplingParams

class AnyLLM:
    def __init__(self, **kwargs):
        if 'gpt' in kwargs.get("model_name"):
            self.llm = OpenAILLM(**kwargs)
        else:
            self.llm = LocalLLM(**kwargs)

    def __call__(self, chat_history: list, probs=False):
        return self.llm(chat_history, probs)


class OpenAILLM:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "gpt4-turbo")

        deployment = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt4-turbo")
        
        self.model = AzureChatOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2024-02-01", 
            azure_deployment=deployment, 
            **kwargs['model_kwargs']
        )
    
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __call__(self, chat_history: list, probs=False):

        with get_openai_callback() as cb:
            response = self.model(
                chat_history
                # [
                #     HumanMessage(
                #         content=prompt,
                #     )
                # ]
            )
            output = response.content

        self.prompt_tokens += cb.prompt_tokens
        self.completion_tokens += cb.completion_tokens

        token_probs = {}
        if probs:
            for entry in response.response_metadata['logprobs']['content']:
                token_probs[entry['token']] = np.exp(entry['logprob'])
            return output, token_probs

        return output


class LocalLLM:
    def __init__(
        self,
        model_path,
        device="cuda:0",
        max_batch_size=1,
        max_new_tokens=8192,
        temperature = 0.0,
        **kwargs
    ) -> None:
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16, device_map="auto")
        # self.model = transformers.pipeline(
        #         "text-generation",
        #         model=model_path,
        #         model_kwargs={"torch_dtype": torch.bfloat16},
        #         device_map="auto",
        #     )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_batch_size = max_batch_size
        self.max_new_tokens = max_new_tokens
        print(self.max_new_tokens)
        self.device = device
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.temperature = temperature

    def __call__(self, chat_history, probs=False):
        terminators = [self.tokenizer.eos_token_id]
        if "llama-3" in self.model_path:
            terminators += [self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
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
        if "llama-3" in self.model_path:
            
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt").to(self.device)
        elif "gemma" in self.model_path:
            input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")["input_ids"]
            
        elif "Qwen" in  self.model_path or "qwen" in self.model_path:
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = self.tokenizer([text], return_tensors="pt").to(self.device)["input_ids"]

        
        output_scores = False if not probs else True
        
        with torch.inference_mode():
            generation_output = self.model.generate(
                input_ids,
                generation_config=generation_config,
                output_scores=output_scores,
                return_dict_in_generate=True,
            )
        if probs:
            transition_scores = self.model.compute_transition_scores(
                generation_output.sequences, generation_output.scores, normalize_logits=True
            )

        input_length = input_ids.shape[1]
        generated_ids = generation_output.sequences[:, input_length:]

        token_probs = {}
        if probs:
            for tok, score  in zip(generated_ids[0], transition_scores[0]):
                token_probs[self.tokenizer.decode(tok)] = np.exp(score.item())
            
            # token_logits, tokens = [], []
            # logits = generation_output.scores
            # for tok, score in zip(generated_ids[0], transition_scores[0]):
            #     # give top 20 tokens with their logit scores
            #     top20logtis = {}
            #     top20 = torch.topk(score, 20)
            #     for ix, logit in zip(top20.indices[0], top20.values[0]):
            #         top20logtis[self.tokenizer.decode(ix)] = logit

            #     token_logits.append(top20logtis)
            #     tokens.append(self.tokenizer.decode(tok))
                # token_logits[self.tokenizer.decode(tok)] = top20logtis

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded_text = decoded[0].strip()
        if "Qwen3" in self.model_path:
            decoded_text = decoded_text.split("</think>")[-1].strip()
        if probs:
            # return decoded[0].strip(), tokens, token_logits
            return decoded_text, token_probs
        else:
            return decoded_text
        