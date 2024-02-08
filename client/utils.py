import cv2
import pyttsx3
import speech_recognition as sr
import socket

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

def setup_recorder():
    recorder = sr.Recognizer()
    recorder.energy_threshold = 3000
    recorder.dynamic_energy_threshold = False
    return recorder

def setup_socket():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 5555))
    return client_socket
