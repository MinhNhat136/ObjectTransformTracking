import socket
import struct

HOST = '127.0.0.1'
PORT = 8052

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                # Read header (8 bytes) to get payload size
                header = conn.recv(8)
                if not header:
                    break
                
                payload_size = struct.unpack('<q', header)[0]
                
                # Read payload
                data = b""
                while len(data) < payload_size:
                    packet = conn.recv(payload_size - len(data))
                    if not packet:
                        break
                    data += packet
                
                # Parse payload (assuming format from SocketHelper: 4 bytes len + string)
                # But SocketHelper writes: header(8) + (len(4) + string)
                # So the 'data' we just read contains (len(4) + string)
                
                offset = 0
                while offset < len(data):
                    if offset + 4 > len(data): break
                    str_len = struct.unpack_from('<I', data, offset)[0]
                    offset += 4
                    
                    if offset + str_len > len(data): break
                    msg_bytes = data[offset:offset+str_len]
                    msg_str = msg_bytes.decode('utf-8')
                    print(f"Received: {msg_str}")
                    offset += str_len

if __name__ == "__main__":
    start_server()
