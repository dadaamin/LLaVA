import io
import argparse
import json
import math
import os

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

import shortuuid

# step 1: Setup constant
device = "cuda"
dtype = torch.float16


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def main(args):

    # step 2: Load Processor and Model
    processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
    model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True)


    ####################
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        if "binary" in line:
            if line["binary"]:
                qs += " Please start your answer with the words Yes or No."
        cur_prompt = qs


        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        inputs = processor(images=[image], text=f" USER: <s>{cur_prompt} ASSISTANT: <s>", return_tensors="pt").to(device=device, dtype=dtype)
        output = model.generate(**inputs, generation_config=generation_config)[0]
        response = processor.tokenizer.decode(output, skip_special_tokens=True)



        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": response,
                                   "answer_id": ans_id,
                                   "model_id": "chexagent",
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    ########################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    args = parser.parse_args()
    main(args)