import cv2
import pyttsx3
import socket
import pickle
import struct
from PIL import Image
import numpy as np
from datetime import datetime, timedelta
from queue import Queue
from threading import Thread, Event
import speech_recognition as sr

def speak(text, speed=75):
    engine = pyttsx3.init()
    
    # Set the speech rate (speed)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate * (speed / 100))
    
    engine.say(text)
    engine.runAndWait()

def webcam(webcam_id=0):
    cap = cv2.VideoCapture(webcam_id)

    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        exit()

    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame from the webcam.")
        exit()

    cap.release()

    return frame

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
            chunk = socket.recv(data_size - len(data))
            if not chunk:
                raise RuntimeError("Socket connection broken")
            data += chunk

        if data:
            data_available_event.set()  # Signal that new data is available
            return pickle.loads(data)
        else:
            # Optionally add a delay or other logic here to control the polling frequency
            pass
            
def send_data(socket, data_type, data):
    # Send data type (4 bytes for a string)
    socket.send(struct.pack("!I", len(data_type)))
    socket.sendall(data_type.encode('utf-8'))

    # Based on the data type, send either an image or a prompt
    if data_type == "IMAGE":
        send_image(socket, data)
    elif data_type == "PROMPT":
        send_prompt(socket, data)
    else:
        raise RuntimeError(f"Invalid data type: {data_type}")

def send_prompt(socket, prompt):
    # Serialize the prompt and send its size
    prompt_data = prompt.encode('utf-8')
    prompt_size = struct.pack("!I", len(prompt_data))
    socket.send(prompt_size)

    # Send the prompt in chunks
    socket.sendall(prompt_data)

def send_image(socket, pil_image):
    # Serialize the image and send its size
    image_data = pickle.dumps(pil_image)
    image_size = struct.pack("!I", len(image_data))
    socket.send(image_size)

    # Send the image in chunks
    socket.sendall(image_data)

def print_transcription(transcription):
    for line in transcription:
        print(line)
    print('', end='', flush=True)

def setup_recorder():
    recorder = sr.Recognizer()
    recorder.energy_threshold = 3000
    recorder.dynamic_energy_threshold = False
    return recorder

def setup_socket():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 5555))
    return client_socket
