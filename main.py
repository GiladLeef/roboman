import cv2
from PIL import Image
import socket
import numpy as np
from datetime import datetime, timedelta
from queue import Queue
from threading import Thread
import speech_recognition as sr
import torch
import whisper
import utils
from threading import Thread, Event

pause_listen_event = Event()

def record_callback(_, audio: sr.AudioData, data_queue: Queue) -> None:
    if not pause_listen_event.is_set():
        data_queue.put(audio.get_raw_data())
        
def process_audio(data_queue, audio_model, client_socket):
    phrase_time = None
    data_available_event = Event()
    webcam_frame = cv2.VideoCapture(0).read()[1]
    pil_image = Image.fromarray(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB))
    utils.send_data(client_socket, "IMAGE", pil_image)
    
    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=1):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                words = text.lower().split()
                print(words)

                if len(words) >= 2 and any("bot" in word for word in words[:2]):
                    processed_prompt = " ".join(words[2:]).strip()
                
                else:
                    processed_prompt = text

                utils.send_data(client_socket, "PROMPT", processed_prompt)

                server_response = utils.receive_data(client_socket, data_available_event)
                data_available_event.clear()  # Reset the event after handling data

                print(f"Server Response: {server_response}")

                pause_listen_event.set()
                utils.speak(server_response)
                pause_listen_event.clear()

        except KeyboardInterrupt:
            break

def main():
    model = "base.en"
    data_queue = Queue()

    recorder = utils.setup_recorder()
    source = sr.Microphone(sample_rate=16000)
    audio_model = whisper.load_model(model)

    record_timeout = 1

    listen_thread = recorder.listen_in_background(source, lambda _, audio: record_callback(_, audio, data_queue),
                                                   phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    client_socket = utils.setup_socket()

    try:
        audio_thread = Thread(target=process_audio, args=(data_queue, audio_model, client_socket))
        audio_thread.start()

        try:
            audio_thread.join()
        except KeyboardInterrupt:
            listen_thread.stop()
            listen_thread.join()

    finally:
        client_socket.close()

if __name__ == "__main__":
    main()
