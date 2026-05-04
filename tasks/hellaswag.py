from typing import Tuple

import fishfarm
import vllm
from datasets import load_dataset
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.ai2_arc import Ai2ArcSample, Ai2ArcTask

from .base import Task, get_download_dir

CHOICES = ["A", "B", "C", "D"]


class HellaSwagTask(Task):
    def __init__(self):
        self.model_to_template = {
            "Qwen/Qwen2.5-1.5B-Instruct": None,
            "meta-llama/Meta-Llama-3-8B-Instruct": None,
            "mistralai/Mistral-7B-Instruct-v0.3": None,
        }
        self.system_msg = (
            "The following are multiple choice questions about commonsense reasoning (with answers). "
            "Think step by step and then finish your answer "
            'with "the answer is (X)" where X is the correct letter choice.'
        )
        self.target_metric_train = "acc"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = False
        self.has_training_split = False

    def get_train_data(self):
        return None, None, None

    def get_rewards(self, res):
        rewards = [1.0 if x["correct"] else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(self) -> Tuple:
        # HellaSwag validation split is used as test (test split has no labels)
        dataset = load_dataset("Rowan/hellaswag", split="validation")

        samples = []
        for i, sample in enumerate(dataset):
            endings = sample["endings"]  # list of 4 strings
            answer_idx = int(sample["label"])  # int 0-3
            answer_letter = CHOICES[answer_idx]

            # Context: activity_label + ctx
            ctx = sample["activity_label"] + ": " + sample["ctx"]
            question = "What happens next?\n" + ctx + "\n"
            question += "Options:\n"
            for j, ending in enumerate(endings):
                question += "{}. {}\n".format(CHOICES[j], ending)

            samples.append(
                Ai2ArcSample(
                    question=question,
                    answer=answer_letter,
                    options=endings,
                    question_id=f"hellaswag_val_{i}",
                )
            )

        test_eval = Ai2ArcTask(
            samples=samples,
            context_messages=[
                fishfarm.Message("system", self.system_msg),
            ],
        )
        return (None, test_eval)

    def get_prompt(self, tokenizer, samples, ix, model_id):
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": self.system_msg}
        user_msg = {"role": "user", "content": samples[ix].question}
        prompt = tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def get_vllm_model(self, model_id) -> VLLMModel:
        """Load a vLLM model."""
        model = vllm.LLM(
            model_id,
            max_model_len=1024,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            dtype="bfloat16",
            download_dir=get_download_dir(),
        )
        chat_template = self.model_to_template[model_id]
        m = model.llm_engine.model_executor.driver_worker.model_runner.model
        for _, param in m.named_parameters():
            param.requires_grad = False
        vllm_model = VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=512,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
        )
        return vllm_model
