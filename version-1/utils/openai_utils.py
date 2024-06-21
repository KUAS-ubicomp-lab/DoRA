import asyncio
import os

import openai
import time
from tqdm import tqdm

openai.api_key = os.getenv("")


async def dispatch_openai_requests(
        messages_list,
        model: str,
        temperature: float,
        max_tokens: int,
):
    async_responses = [
        await openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=x
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def dispatch_openai_api_requests(prompt_list, shots, api_batch, temperature, max_tokens, api_model_name):
    openai_responses = []

    for i in tqdm(range(0, shots, api_batch)):
        while True:
            try:
                openai_responses += asyncio.run(
                    dispatch_openai_requests(prompt_list[i:i + api_batch], api_model_name, temperature, max_tokens)
                )
                break
            except KeyboardInterrupt:
                print(f'KeyboardInterrupt Error, retry batch {i // api_batch} at {time.ctime()}',
                      flush=True)
                time.sleep(5)
            except Exception as e:
                print(f'Error {e}, retry batch {i // api_batch} at {time.ctime()}', flush=True)
                time.sleep(5)
    return openai_responses
