import socket

class SocketData:
    def __init__(self):
        self._binary: bytearray = bytearray()

    def get_binary(self) -> bytearray:
        return self._binary

    def get_size_in_bytes(self) -> int:
        return len(self._binary)
