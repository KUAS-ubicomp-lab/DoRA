import asyncio
import openai
import time
from tqdm import tqdm

openai.api_key = ""
openai.api_base = ""


async def dispatch_openai_requests(
        messages_list,
        model: str,
):
    async_responses = [
        await openai.ChatCompletion.create(
            model=model,
            messages=x,
            temperature=0
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def dispatch_openai_api_requests(prompt_list, batch_size, api_batch, api_model_name="gpt-3"):
    openai_responses = []

    for i in tqdm(range(0, batch_size, api_batch)):
        while True:
            try:
                openai_responses += asyncio.run(
                    dispatch_openai_requests(prompt_list[i:i + api_batch], api_model_name)
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
