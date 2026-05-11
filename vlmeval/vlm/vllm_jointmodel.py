import torch
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from typing import List, Dict, Union
import re

class VLLMJointModel:
    def __init__(self, perceiver_model_name_or_path: str, thinker_model_name_or_path: str, tensor_parallel_size: int = 1, **kwargs):
        self.perceiver_engine = LLM(
            model=perceiver_model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.45,
            trust_remote_code=True,
            **kwargs
        )
        self.p_processor = AutoProcessor.from_pretrained(perceiver_model_name_or_path, trust_remote_code=True)

        self.thinker_engine = LLM(
            model=thinker_model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.45,
            trust_remote_code=True,
            **kwargs
        )
        self.t_tokenizer = self.thinker_engine.get_tokenizer()

        self.prompt_added_list = ["Please select the correct answer from the options above.", '请直接回答选项字母。', 'Please select the correct answer from the options above.', ' Please answer yes or no.', 'Please try to answer the question with short words or phrases if possible.']

    @classmethod
    def from_pretrained(cls, perceiver_model_name_or_path: str, thinker_model_name_or_path: str, tensor_parallel_size: int = 1, **kwargs):
        return cls(perceiver_model_name_or_path, thinker_model_name_or_path, tensor_parallel_size, **kwargs)

    def generate(
        self,
        images: List[Union[str, Image.Image]],
        msgs: List[List[Dict[str, str]]],
        p_sampling_params: SamplingParams,
        t_sampling_params: SamplingParams
    ) -> List[Dict[str, str]]:
        assert len(images) == len(msgs)

        # p_prompt_template = 'Please describe the image in great detail, including information related to the question. Here is the question: {question}'

        p_prompt_template = "Please describe the image in great detail."

        questions = [msg[0]['content'] for msg in msgs]

        p_prompts = []
        for q in questions:
            # # for p_message, remove the added prompt
            # p_prompt_template_final = p_prompt_template.format(question=q)
            # for added_prompt in self.prompt_added_list:
            #     if added_prompt in p_prompt_template_final:
            #         p_prompt_template_final = p_prompt_template_final.replace(added_prompt, '')
            # p_prompt_template_final = p_prompt_template_final.strip()

            p_prompt_template_final = p_prompt_template

            p_message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': p_prompt_template_final}]}]
            text = self.p_processor.apply_chat_template(p_message, tokenize=False, add_generation_prompt=True)
            p_prompts.append(text)

        inputs = [
            {
                "prompt": p_prompts[i],
                "multi_modal_data": {"image": images[i]}
            } for i in range(len(p_prompts))
        ]

        p_outputs = self.perceiver_engine.generate(
            inputs,
            sampling_params=p_sampling_params,
        )

        descriptions = [output.outputs[0].text.strip() for output in p_outputs]

        t_prompt_template = '{question} This is a detailed description of the image: {description}'

        t_prompts = []
        for i in range(len(descriptions)):
            prompt = t_prompt_template.format(question=questions[i], description=descriptions[i])
            message = [{"role": "user", "content": prompt}]
            text = self.t_tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            t_prompts.append(text)

        t_outputs = self.thinker_engine.generate(
            prompts=t_prompts,
            sampling_params=t_sampling_params
        )

        final_responses = [output.outputs[0].text.strip() for output in t_outputs]

        # Clean up the responses by removing <think> tags
        cleaned_final_responses = [re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip() for response in final_responses]

        # returning both the final responses and descriptions for further processing if needed
        results_list = []
        for i in range(len(images)):
            results_list.append({
                'prediction': cleaned_final_responses[i],
                'detailed_prediction': final_responses[i],
                'description': descriptions[i]
            })
        return results_list

    def chat(
        self,
        image: Union[str, Image.Image],
        msgs: List[Dict[str, str]],
        p_sampling_params: SamplingParams,
        t_sampling_params: SamplingParams
    ) -> List[Dict[str, str]]:
        response_list = self.generate(
            images=[image],
            msgs=[msgs],
            p_sampling_params=p_sampling_params,
            t_sampling_params=t_sampling_params
        )
        return response_list[0] if response_list else {'prediction': '', 'description': ''}

    def perceiver_generate(
        self,
        images: List[Union[str, Image.Image]],
        msgs: List[List[Dict[str, str]]],
        p_sampling_params: SamplingParams
    ) -> List[str]:
        assert len(images) == len(msgs)

        p_prompt_template = 'Please describe the image in great detail, including information related to the question. Here is the question: {question}'
        questions = [msg[0]['content'] for msg in msgs]

        p_prompts = []
        for q in questions:
            p_message = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': p_prompt_template.format(question=q)}]}]
            text = self.p_processor.apply_chat_template(p_message, tokenize=False, add_generation_prompt=True)
            p_prompts.append(text)

        inputs = [
            {
                "prompt": p_prompts[i],
                "multi_modal_data": {"image": images[i]}
            } for i in range(len(p_prompts))
        ]

        p_outputs = self.perceiver_engine.generate(
            inputs,
            sampling_params=p_sampling_params,
        )

        descriptions = [output.outputs[0].text.strip() for output in p_outputs]
        return descriptions

    def perceiver_chat(
        self,
        image: Union[str, Image.Image],
        msgs: List[Dict[str, str]],
        p_sampling_params: SamplingParams
    ) -> str:
        response_list = self.perceiver_generate(
            images=[image],
            msgs=[msgs],
            p_sampling_params=p_sampling_params
        )
        return response_list[0] if response_list else ""

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise ValueError("No CUDA GPUs found. vLLM requires at least one GPU.")

    print(f"Found {world_size} GPUs. Initializing model with tensor_parallel_size={world_size}.")

    PERCEIVER_PATH = "/path/to/home/models/Qwen2.5-VL-3B-Instruct"
    THINKER_PATH = "/path/to/home/models/Qwen3-4B"
    IMAGE_PATH_1 = "sample.png"
    IMAGE_PATH_2 = "sample.png"

    model = VLLMJointModel.from_pretrained(
        perceiver_model_name_or_path=PERCEIVER_PATH,
        thinker_model_name_or_path=THINKER_PATH,
        tensor_parallel_size=world_size
    )

    print("\nModel loaded successfully.")

    # p_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
    # t_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2048)
    p_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
    t_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=32768)

    try:
        all_images = [
            Image.open(IMAGE_PATH_1).convert("RGB"),
            Image.open(IMAGE_PATH_2).convert("RGB")
        ]
    except FileNotFoundError:
        print("="*50)
        print("ERROR: Image files not found. Please update IMAGE_PATH_1 and IMAGE_PATH_2.")
        print("Using placeholder black images for demonstration purposes.")
        print("="*50)
        all_images = [Image.new('RGB', (224, 224)), Image.new('RGB', (224, 224))]

    all_msgs = [
        [{"role": "user", "content": "What is in the image?"}],
        [{"role": "user", "content": "Describe the main subject in this picture."}]
    ]

    print(f"\nProcessing {len(all_images)} samples across {world_size} GPUs...")

    final_results = model.generate(all_images, all_msgs, p_params, t_params)

    print("\n--- Batch response ---")
    for i, response in enumerate(final_results):
        print(f"Response for image {i+1}: {response}\n" + "-"*25)
