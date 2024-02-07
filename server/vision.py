from PIL import Image
from moondream.moondream import Moondream
from transformers import TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer
from queue import Queue
from threading import Thread
import re
import os

def init(device, dtype):
    tokenizer = Tokenizer.from_pretrained("tokenizer")
    moondream = Moondream.from_pretrained("model").to(device=device, dtype=dtype)
    moondream.eval()

    return moondream, tokenizer

def get_embeddings(image, moondream):
    image_embeds = moondream.encode_image(image)
    return image_embeds

def process(image_embeds, moondream, tokenizer, chat_history, prompt):
    result_queue = Queue()

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    thread_args = (image_embeds, prompt, tokenizer, chat_history)
    thread_kwargs = {"streamer": streamer, "result_queue": result_queue}

    thread = Thread(
        target=moondream.answer_question,
        args=thread_args,
        kwargs=thread_kwargs,
    )
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        if not new_text.endswith("<") and not new_text.endswith("END"):
            print(buffer, end="", flush=True)
            buffer = ""
    print(re.sub("<$", "", re.sub("END$", "", buffer)))

    thread.join()

    answer = result_queue.get()
    chat_history += f"Question: {prompt}\n\nAnswer: {answer}\n\n"
    print(chat_history)
    return answer, chat_history