import cv2
import vision
from PIL import Image
import torch
import socket
import pickle
import struct

def process_image(prompt, pil_image, moondream, image_embeds, tokenizer, device, dtype):
    answer = vision.process(image_embeds, device, dtype, moondream, tokenizer, prompt)
    return answer

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

def send_answer(socket, answer):
    # Serialize the answer and send its size
    answer_data = pickle.dumps(answer)
    answer_size = struct.pack("!I", len(answer_data))
    socket.send(answer_size)

    # Send the answer in chunks
    socket.sendall(answer_data)

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype = torch.float16

    image_embeds = None

    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 5555))
    server_socket.listen(1)

    moondream, tokenizer = vision.init(device, dtype)

    print("Server is waiting for a connection...")

    while True:
        # Accept a new client connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")

        try:
            while True:
                try:
                    data_type, data = receive_data(client_socket)

                    if data_type == "IMAGE":
                        pil_image = data
                        image_embeds = vision.get_embeddings(pil_image, moondream)
                    elif data_type == "PROMPT":
                        pil_image = None  # No new image, use the last one
                    else:
                        raise RuntimeError(f"Invalid data type: {data_type}")

                except ConnectionResetError:
                    print("Client closed the connection. Waiting for a new client...")
                    break

                except RuntimeError as e:
                    print(f"Error receiving data: {e}")
                    continue

                # Process the image and prompt
                answer = process_image(data, pil_image, moondream, image_embeds, tokenizer, device, dtype)

                # Send the answer back to the client
                send_answer(client_socket, answer)

        finally:
            client_socket.close()

if __name__ == "__main__":
    main()
