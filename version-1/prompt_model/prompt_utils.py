import inspect
import os
from inspect import iscoroutinefunction, isgeneratorfunction

import openai
from transformers import GPT2TokenizerFast

_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt3')
GPT3_LENGTH_LIMIT = 2049
openai.api_key = os.getenv("")


def gpt_style_tokenize(x):
    return _TOKENIZER.tokenize(x)


def length_of_prompt(prompt, max_tokens):
    return len(_TOKENIZER.tokenize(prompt)) + max_tokens


def add_engine_argument(parser):
    parser.add_argument('--engine',
                        default='gpt-3',
                        choices=['gpt-3', 'chat-gpt', 'gpt-4'])


def specify_engine(args):
    args.engine_name = args.engine


def prompt_for_prediction(prompt_example, shots=0):
    showcase_examples = [
        "{}\nQ: {} True, False?\nA: {}\n".format(shot["premise"], shot["hypothesis"],
                                                 shot["label"]) for shot in shots
    ]
    input_example = "{}\nQ: {} True, False, or Neither?\nA:".format(prompt_example["premise"],
                                                                    prompt_example["hypothesis"])

    prompt = "\n".join(showcase_examples + [input_example])
    return prompt


def fix(args, kwargs, sig):
    ba = sig.bind(*args, **kwargs)
    ba.apply_defaults()
    return ba.args, ba.kwargs


def decorate(func, caller, extras=(), kwsyntax=False):
    sig = inspect.signature(func)
    if iscoroutinefunction(caller):
        async def fun(*args, **kw):
            if not kwsyntax:
                args, kw = fix(args, kw, sig)
            return await caller(func, *(extras + args), **kw)
    elif isgeneratorfunction(caller):
        def fun(*args, **kw):
            if not kwsyntax:
                args, kw = fix(args, kw, sig)
            for res in caller(func, *(extras + args), **kw):
                yield res
    else:
        def fun(*args, **kw):
            if not kwsyntax:
                args, kw = fix(args, kw, sig)
            return caller(func, *(extras + args), **kw)
    fun.__name__ = func.__name__
    fun.__doc__ = func.__doc__
    __wrapped__ = func  # support nested wrap
    fun.__signature__ = sig
    fun.__qualname__ = func.__qualname__
    # builtin functions like defaultdict.__setitem__ lack many attributes
    try:
        fun.__defaults__ = func.__defaults__
    except AttributeError:
        pass
    try:
        fun.__kwdefaults__ = func.__kwdefaults__
    except AttributeError:
        pass
    try:
        fun.__annotations__ = func.__annotations__
    except AttributeError:
        pass
    try:
        fun.__module__ = func.__module__
    except AttributeError:
        pass
    try:
        fun.__dict__.update(func.__dict__)
    except AttributeError:
        pass
    fun.__wrapped__ = __wrapped__  # support nested wrap
    return fun


def dsm_criteria():
    dsm_criteria = [
        "Persistent sad, anxious, or 'empty' mood",
        "Feelings of hopelessness or pessimism",
        "Irritability",
        "Feelings of guilt, worthlessness, or helplessness",
        "Loss of interest or pleasure in hobbies and activities"
    ]
    return dsm_criteria
