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


def send_msg(sock: socket.socket, msg: dict) -> int:
    """发送一条消息: 4字节长度头 + 序列化数据。返回发送的总字节数。"""
    data = serialize_message(msg)
    length = struct.pack("!I", len(data))
    sock.sendall(length + data)
    return 4 + len(data)


def recv_msg(sock: socket.socket) -> tuple:
    """接收一条消息: 先读4字节长度头，再读数据。返回 (msg_dict, total_bytes)。"""
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        raise ConnectionError("Connection closed by remote")
    msg_len = struct.unpack("!I", raw_len)[0]
    data = _recv_exact(sock, msg_len)
    return deserialize_message(data), 4 + msg_len


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
    """BS 端 TCP 服务器封装（支持客户端断线重连 + handler 异常容错）"""

    def __init__(self, host: str = "0.0.0.0", port: int = 50051):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _try_set_bufsize(self.sock)

    def start(self, handler_fn):
        """
        启动服务器，循环接受客户端连接。
        每收到一条请求就调用 handler_fn(request_dict) -> response_dict。
        - handler_fn 抛异常时：向客户端返回 error 消息，继续服务
        - 客户端断开时：回到 accept() 等待新连接
        """
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"[BS Server] Listening on {self.host}:{self.port}")

        try:
            while True:
                print(f"[BS Server] Waiting for UAV client connection ...")
                conn, addr = self.sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                _try_set_bufsize(conn)
                print(f"[BS Server] UAV client connected from {addr}")

                self._serve_connection(conn, handler_fn)

                print(f"[BS Server] Connection from {addr} closed, "
                      f"ready for next client.")
        except KeyboardInterrupt:
            print("\n[BS Server] Shutting down (Ctrl+C).")
        finally:
            self.sock.close()

    def _serve_connection(self, conn: socket.socket, handler_fn):
        """持续处理单个连接上的请求，直到客户端断开"""
        req_count = 0
        try:
            while True:
                # 1. 接收请求
                try:
                    request, _ = recv_msg(conn)
                except ConnectionError:
                    print(f"[BS Server] Client disconnected after {req_count} requests.")
                    return

                # 2. 处理请求（handler 异常不会断连）
                req_count += 1
                try:
                    response = handler_fn(request)
                except Exception as e:
                    print(f"[BS Server] ❌ Handler error on request #{req_count}: {e}")
                    response = {"error": str(e)}

                # 3. 发送响应
                try:
                    send_msg(conn, response)
                except (BrokenPipeError, ConnectionError) as e:
                    print(f"[BS Server] Failed to send response: {e}")
                    return
        finally:
            conn.close()


class UAVClient:
    """UAV 端 TCP 客户端封装（带流量统计 + 自动重连）"""

    def __init__(self, bs_host: str, bs_port: int = 50051,
                 max_retries: int = 5, retry_delay: float = 2.0):
        self.bs_host = bs_host
        self.bs_port = bs_port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sock: socket.socket | None = None

        # 流量统计
        self.total_tx_bytes = 0   # 累计上行字节数 (UAV → BS)
        self.total_rx_bytes = 0   # 累计下行字节数 (BS → UAV)
        self.call_count = 0       # RPC 调用次数

    def _create_socket(self) -> socket.socket:
        """创建并配置新的 socket"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _try_set_bufsize(s)
        return s

    def connect(self):
        """连接到 BS 服务器"""
        self.sock = self._create_socket()
        self.sock.connect((self.bs_host, self.bs_port))
        print(f"[UAV Client] Connected to BS at {self.bs_host}:{self.bs_port}")

    def _reconnect(self):
        """关闭旧连接并重新建立"""
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        import time as _time
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"[UAV Client] Reconnecting to BS ... "
                      f"(attempt {attempt}/{self.max_retries})")
                self.sock = self._create_socket()
                self.sock.connect((self.bs_host, self.bs_port))
                print(f"[UAV Client] ✅ Reconnected to BS at "
                      f"{self.bs_host}:{self.bs_port}")
                return
            except (ConnectionRefusedError, OSError) as e:
                print(f"[UAV Client] Reconnect failed: {e}")
                if attempt < self.max_retries:
                    _time.sleep(self.retry_delay)
        raise ConnectionError(
            f"[UAV Client] Failed to reconnect after {self.max_retries} attempts")

    def reset_stats(self):
        """重置流量统计（每个实验开始前调用）"""
        self.total_tx_bytes = 0
        self.total_rx_bytes = 0
        self.call_count = 0

    def call(self, request: dict) -> dict:
        """发送请求并等待响应（同步 RPC），同时统计流量。
        遇到 Broken pipe / ConnectionError 时自动重连并重试一次。"""
        for attempt in range(2):  # 最多尝试 2 次（原始 + 重连后重试）
            try:
                tx = send_msg(self.sock, request)
                response, rx = recv_msg(self.sock)
                # 检查 BS 端是否返回了错误
                if "error" in response:
                    raise RuntimeError(f"BS server error: {response['error']}")
                self.total_tx_bytes += tx
                self.total_rx_bytes += rx
                self.call_count += 1
                return response
            except (BrokenPipeError, ConnectionError, OSError) as e:
                if attempt == 0:
                    print(f"[UAV Client] ⚠ Connection lost: {e}, reconnecting ...")
                    self._reconnect()
                else:
                    raise

    def get_traffic_stats(self) -> dict:
        """获取当前流量统计"""
        return {
            "net_tx_bytes": self.total_tx_bytes,
            "net_rx_bytes": self.total_rx_bytes,
            "net_total_bytes": self.total_tx_bytes + self.total_rx_bytes,
            "net_rpc_calls": self.call_count,
        }

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
