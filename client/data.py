import socket
import pickle
import struct
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
from queue import Queue
from threading import Thread, Event

def receive_data(socket, data_available_event):
    while True:

        # Receive data size (4 bytes for an integer)
        data_size_data = b''
        while len(data_size_data) < 4:
            chunk = socket.recv(4 - len(data_size_data))
            if not chunk:
                raise RuntimeError("Socket connection broken")
            data_size_data += chunk

        data_size = struct.unpack("!I", data_size_data)[0]


        # Receive data
        data = b''
        while len(data) < data_size:
            chunk = socket.recv(min(4096, data_size - len(data)))  # Adjust buffer size as needed
            if not chunk:
                raise RuntimeError("Socket connection broken")
            data += chunk

        if data:
            data_available_event.set()  # Signal that new data is available
            return data

def send_prompt(socket, prompt):
    data_type = "PROMPT"
    socket.send(struct.pack("!I", len(data_type)))
    socket.sendall(data_type.encode('utf-8'))
    
    # Serialize the prompt and send its size
    prompt_data = prompt.encode('utf-8')
    prompt_size = struct.pack("!I", len(prompt_data))
    socket.send(prompt_size)

    # Send the prompt in chunks
    socket.sendall(prompt_data)

def send_image(socket, pil_image):
    data_type = "IMAGE"
    socket.send(struct.pack("!I", len(data_type)))
    socket.sendall(data_type.encode('utf-8'))
    # Serialize the image and send its size
    image_data = pickle.dumps(pil_image)
    image_size = struct.pack("!I", len(image_data))
    socket.send(image_size)

    # Send the image in chunks
    socket.sendall(image_data)
