import socket
import json
import time
from entity_socket import SocketData
from socket_helper import SocketHelper

HOST = '127.0.0.1'
PORT = 8052

def test_send():
    print(f"Connecting to {HOST}:{PORT}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        print("Connected!")
        
        # Simulate data
        data = [1.0, 2.0, 3.0, 45.0, 90.0, 180.0]
        json_data = json.dumps(data)
        print(f"Sending: {json_data}")
        
        socket_data = SocketData()
        SocketHelper.write_payload(socket_data, json_data)
        
        sock.sendall(socket_data.get_binary())
        print("Data sent.")
        
        time.sleep(1)
        sock.close()
        print("Socket closed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_send()
