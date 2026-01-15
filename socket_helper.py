import struct
from entity_socket import SocketData

class SocketHelper:
    @staticmethod
    def write_header(socket_data: SocketData, value: int):
        binary = socket_data.get_binary()
        if len(binary) < 8:
            binary.extend(bytearray(8 - len(binary)))

        struct.pack_into('<q', binary, 0, value)

    @staticmethod
    def write_payload(socket_data: SocketData, value):
        binary = socket_data.get_binary()
        if len(binary) < 8:
            SocketHelper.write_header(socket_data, 0)

        if isinstance(value, bytes):
            data_bytes = value
        elif isinstance(value, str):
            data_bytes = value.encode('utf-8')
        else:
            raise TypeError(f"Unsupported value type: {type(value)}. Expect bytes or str.")

        string_data = struct.pack('<I', len(data_bytes)) + data_bytes
        binary.extend(string_data)

        payload_size = len(binary) - 8
        SocketHelper.write_header(socket_data, payload_size)

    @staticmethod
    def get_data_components_in_package(data: bytearray):
        results = []
        offset = 0
        while offset + 4 <= len(data):
            text_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4

            if offset + text_len > len(data):
                break

            text_bytes = data[offset:offset + text_len]
            results.append(text_bytes.decode("utf-8"))
            offset += text_len

        return results