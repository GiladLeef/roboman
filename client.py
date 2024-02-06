import cv2
from PIL import Image
import socket
import pickle
import struct
import utils

def receive_all(socket, size):
    data = b''
    while len(data) < size:
        chunk = socket.recv(size - len(data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        data += chunk
    return data

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

def main():
    # Set up client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("localhost", 5555))

    try:
        # Capture image from webcam
        webcam_frame = cv2.VideoCapture(0).read()[1]
        pil_image = Image.fromarray(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB))

        # Send image to server
        send_data(client_socket, "IMAGE", pil_image)

        while True:
            # Collect prompt from user
            prompt = input("Enter a prompt: ")

            # Send prompt to server
            send_data(client_socket, "PROMPT", prompt)

            # Receive and print the answer from the server
            answer_size_data = client_socket.recv(4)
            answer_size = struct.unpack("!I", answer_size_data)[0]

            answer_data = receive_all(client_socket, answer_size)
            answer = pickle.loads(answer_data)

            print("Server's Answer:", answer)
            utils.speak(answer)

    finally:
        client_socket.close()

if __name__ == "__main__":
    main()
