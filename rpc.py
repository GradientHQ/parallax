import asyncio
import time
import sys
from lattica import Lattica, rpc_method, rpc_stream, rpc_stream_iter, ConnectionHandler
import pickle
from starlette.concurrency import iterate_in_threadpool

class MockProtoRequest:
    def __init__(self, data=None):
        self.data = data
        self.timestamp = time.time()

    def SerializeToString(self):
        return pickle.dumps({
            'data': self.data,
            'timestamp': self.timestamp
        })

    def ParseFromString(self, data):
        parsed = pickle.loads(data)
        self.data = parsed['data']
        self.timestamp = parsed['timestamp']
        return self

class MockProtoResponse:
    def __init__(self, data=None, message=''):
        self.data = data
        self.message = message

    def SerializeToString(self):
        return pickle.dumps({
            'data': self.data,
            'message': self.message,
        })

    def ParseFromString(self, data):
        parsed = pickle.loads(data)
        self.message = parsed['message']
        self.data = parsed['data']
        return self

class RPCHandler(ConnectionHandler):
    @rpc_method
    def add(self, a: int, b: int) -> int:
        return a + b

    @rpc_method
    def simple_rpc(
            self,
            request: MockProtoRequest,
    ) -> MockProtoResponse:
        return MockProtoResponse(
            message=f"Processed data of size {len(request.data)}",
            data=None
        )

    @rpc_stream
    def stream_rpc(self, request: MockProtoRequest ) -> MockProtoResponse:
        return MockProtoResponse(
            message=f"Processed data of size {len(request.data)}",
            data=None
        )

    @rpc_stream_iter
    def stream_rpc_iter(self):
        for _ in range(100):
            text = "hello world"
            time.sleep(0.01)
            yield text

async def run_server():
    lattica_inst = Lattica.builder().with_listen_addrs(["/ip4/0.0.0.0/tcp/0"]).with_mdns(False).build()
    RPCHandler(lattica_inst)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Server shutting down...")

def parse_multiaddr(addr_str: str):
    if not addr_str.startswith('/'):
        raise ValueError(f"Invalid multiaddr format: {addr_str}")

    parts = addr_str.strip().split('/')
    if len(parts) < 6:
        raise ValueError(f"Incomplete multiaddr: {addr_str}")

    peer_id = None
    for i, part in enumerate(parts):
        if part == 'p2p' and i + 1 < len(parts):
            peer_id = parts[i + 1]
            break

    if not peer_id:
        raise ValueError(f"No peer ID found in multiaddr: {addr_str}")

    return addr_str, peer_id

async def run_client():
    bootstrap_addr = sys.argv[1]
    bootstrap_addr, server_peer_id = parse_multiaddr(bootstrap_addr)
    bootstrap_nodes = [bootstrap_addr]

    lattica_inst = Lattica.builder().with_bootstraps(bootstrap_nodes).with_mdns(False).build()
    handler = RPCHandler(lattica_inst)
    time.sleep(1)

    try:
        stub = handler.get_stub(server_peer_id)
        async def stream_generator():
            response = stub.stream_rpc_iter()
            try:
                iterator = iterate_in_threadpool(response)
                async for chunk in iterator:
                    yield chunk
            finally:
                print("finally")
                response.cancel()

        print("\n=== Testing Stream iter RPC ===")
        generators = []
        for i in range(100):
            print(f"call {i}")
            a = stream_generator()
            generators.append(a)
            count = 0
            async for text in a :
                count += 1
                print(f"{count} :recv: {text}")
            
            time.sleep(5)

        for generator in generators:
            print(f"type(generator): {type(generator)}, instance id: {id(generator)}")

    except Exception as e:
        print(f"Client error: {e}")


# node1: python rpc.py
# node2: python rpc.py /ip4/127.0.0.1/tcp/x/p2p/x
if __name__ == "__main__":
    if len(sys.argv) > 1:
        asyncio.run(run_client())
    else:
        asyncio.run(run_server())