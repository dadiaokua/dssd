"""
dssd_net.py - 轻量级 TCP 通信层（零额外依赖）
用于 UAV 端和 BS 端之间传输 Python 对象（dict/tensor 等）

协议: [4字节长度头] + [pickle序列化的数据]
"""

import io
import pickle
import socket
import struct
import torch


def serialize_message(msg: dict) -> bytes:
    """将消息（含 tensor）序列化为 bytes"""
    # 把 tensor 转为 CPU 再序列化，避免 CUDA tensor pickle 问题
    clean_msg = {}
    for k, v in msg.items():
        if isinstance(v, torch.Tensor):
            buf = io.BytesIO()
            torch.save(v.cpu(), buf)
            clean_msg[k] = ("__tensor__", buf.getvalue())
        else:
            clean_msg[k] = v
    return pickle.dumps(clean_msg, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_message(data: bytes) -> dict:
    """将 bytes 反序列化为消息（自动恢复 tensor）"""
    clean_msg = pickle.loads(data)
    msg = {}
    for k, v in clean_msg.items():
        if isinstance(v, tuple) and len(v) == 2 and v[0] == "__tensor__":
            buf = io.BytesIO(v[1])
            msg[k] = torch.load(buf, weights_only=True)
        else:
            msg[k] = v
    return msg


def send_msg(sock: socket.socket, msg: dict):
    """发送一条消息: 4字节长度头 + 序列化数据"""
    data = serialize_message(msg)
    length = struct.pack("!I", len(data))
    sock.sendall(length + data)


def recv_msg(sock: socket.socket) -> dict:
    """接收一条消息: 先读4字节长度头，再读数据"""
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        raise ConnectionError("Connection closed by remote")
    msg_len = struct.unpack("!I", raw_len)[0]
    data = _recv_exact(sock, msg_len)
    return deserialize_message(data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """精确接收 n 个字节"""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        buf.extend(chunk)
    return bytes(buf)


def _try_set_bufsize(sock: socket.socket, size: int = 4 * 1024 * 1024):
    """尝试设置 socket 缓冲区大小，macOS 有上限限制，失败则静默跳过"""
    for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, size)
        except OSError:
            pass  # macOS 等系统可能拒绝过大的缓冲区


class BSServer:
    """BS 端 TCP 服务器封装"""

    def __init__(self, host: str = "0.0.0.0", port: int = 50051):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _try_set_bufsize(self.sock)

    def start(self, handler_fn):
        """
        启动服务器，每收到一条请求就调用 handler_fn(request_dict) -> response_dict
        handler_fn 由 bs_server.py 提供
        """
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"[BS Server] Listening on {self.host}:{self.port}")

        conn, addr = self.sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[BS Server] UAV client connected from {addr}")

        try:
            while True:
                try:
                    request = recv_msg(conn)
                except ConnectionError:
                    print("[BS Server] Client disconnected.")
                    break

                response = handler_fn(request)
                send_msg(conn, response)
        finally:
            conn.close()
            self.sock.close()


class UAVClient:
    """UAV 端 TCP 客户端封装"""

    def __init__(self, bs_host: str, bs_port: int = 50051):
        self.bs_host = bs_host
        self.bs_port = bs_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _try_set_bufsize(self.sock)

    def connect(self):
        self.sock.connect((self.bs_host, self.bs_port))
        print(f"[UAV Client] Connected to BS at {self.bs_host}:{self.bs_port}")

    def call(self, request: dict) -> dict:
        """发送请求并等待响应（同步 RPC）"""
        send_msg(self.sock, request)
        return recv_msg(self.sock)

    def close(self):
        self.sock.close()
