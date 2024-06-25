import argparse
import os

import openai
from tqdm import tqdm
from transformers import Trainer, trainer_pt_utils

from prompt_utils import add_engine_argument, specify_engine, length_of_prompt
from ..utils.data_processor import load_dataset
from ..utils.openai_utils import dispatch_openai_api_requests

openai.api_key = os.getenv("")
_MAX_PROMPT_TOKENS = 25


def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argument(parser)
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')
    parser.add_argument('--num_shot', type=int, default=5)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=250)
    args = parser.parse_args()
    specify_engine(args)
    return args


def in_context_prediction(prompt_example, shots, engine, length_test_only=False):
    showcase_examples = [
        "{}\nQ: {}\nA: {}\n".format(s["context"], s["utterance"], s["label"]) for s in shots
    ]
    input_example = "{}\nQ: {}\nA:".format(prompt_example["context"], prompt_example["utterance"])

    prompt = "\n".join(showcase_examples + [input_example])

    if length_test_only:
        prompt_length = length_of_prompt(prompt, _MAX_PROMPT_TOKENS)
        print("-----------------------------------------")
        print(prompt_length)
        print(prompt)
        return prompt_length

    response = dispatch_openai_api_requests(api_model_name=engine,
                                            prompt_list=prompt,
                                            shots=shots,
                                            max_tokens=_MAX_PROMPT_TOKENS,
                                            temperature=0.0,
                                            api_batch=len(prompt))

    prediction = response["choices"][0]
    prediction["prompt"] = prompt
    prediction["text"] = prediction["text"][len(prompt):]
    return prediction


def evaluate(args):
    train_set = load_dataset("data/eRisk18T2_train.csv")
    dev_set = load_dataset("data/eRisk18T2_dev.csv")

    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = dev_set[:args.num_dev]

    train_set = [x for x in train_set]
    dev_set = [y for y in dev_set]
    predictions = []
    for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
        predictions.append(in_context_prediction(x,
                                                 train_set,
                                                 engine=args.engine,
                                                 style=args.style,
                                                 length_test_only=args.run_length_test))

    if args.run_length_test:
        print('MAX', max(predictions), 'COMP', _MAX_PROMPT_TOKENS)
        return

    metrics = Trainer.evaluate(eval_dataset=predictions)
    trainer_pt_utils.save_metrics("evaluation", metrics)
    return metrics
