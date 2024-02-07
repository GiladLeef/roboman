import cv2
import vision
from PIL import Image
import torch
import socket
import pickle
import struct
import utils

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
    chat_history = ""
    while True:
        prompt = None
        # Accept a new client connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")

        try:
            while True:
                try:
                    data_type, data = utils.receive_data(client_socket)

                    if data_type == "IMAGE":
                        pil_image = data
                        image_embeds = vision.get_embeddings(pil_image, moondream)
                    if data_type == "PROMPT":
                        prompt = data

                except ConnectionResetError:
                    print("Client closed the connection. Waiting for a new client...")
                    break

                except RuntimeError as e:
                    print(f"Error receiving data: {e}")
                    continue

                # Check if both image and prompt are available
                if prompt is not None:
                    # Process the image and prompt
                    answer, chat_history = utils.process_image(prompt, moondream, image_embeds, tokenizer, chat_history)
                    # Send the answer back to the client
                    utils.send_answer(client_socket, answer)

        finally:
            client_socket.close()

if __name__ == "__main__":
    main()
