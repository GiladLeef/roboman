import cv2
import vision
from PIL import Image
import torch
import socket
import pickle
import struct

def process_image(prompt, moondream, image_embeds, tokenizer, chat_history, socket):
    chat_history = vision.process(image_embeds, moondream, tokenizer, chat_history, prompt, socket)
    return chat_history

def receive_data(socket):
    # Receive data type size (4 bytes for an integer)
    data_type_size_data = socket.recv(4)
    data_type_size = struct.unpack("!I", data_type_size_data)[0]

    # Receive data type
    data_type = socket.recv(data_type_size).decode('utf-8')

    # Based on the data type, either receive an image or a prompt
    if data_type == "IMAGE":
        return data_type, receive_image(socket)
    elif data_type == "PROMPT":
        return data_type, receive_prompt(socket)
    else:
        raise RuntimeError(f"Invalid data type: {data_type}")

def receive_prompt(socket):
    print("Receiving prompt size...")
    
    # Receive prompt size from client (4 bytes for an integer)
    prompt_size_data = socket.recv(4)
    prompt_size = struct.unpack("!I", prompt_size_data)[0]
    print(f"Received prompt size: {prompt_size}")

    print("Receiving prompt data...")
    
    # Receive prompt from client in chunks
    prompt_data = b''

    while len(prompt_data) < prompt_size:
        chunk = socket.recv(min(4096, prompt_size - len(prompt_data)))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        prompt_data += chunk

    # Decode the received prompt
    prompt = prompt_data.decode('utf-8')
    print(f"Received prompt: {prompt}")

    return prompt

def receive_image(socket):
    # Receive image size from client (4 bytes for an integer)
    image_size_data = socket.recv(4)
    image_size = struct.unpack("!I", image_size_data)[0]

    # Receive image from client in chunks
    image_data = b''

    while len(image_data) < image_size:
        chunk = socket.recv(min(4096, image_size - len(image_data)))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        image_data += chunk

    # Unpickle and convert to PIL Image
    pil_image = pickle.loads(image_data)
    
    return pil_image