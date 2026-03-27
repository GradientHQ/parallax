#!/bin/bash
# 在项目根目录执行，重新编译 forward.proto
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
pip install -q grpcio-tools
python -m grpc_tools.protoc -I. --python_out=. src/parallax/p2p/proto/forward.proto
echo "Done: src/parallax/p2p/proto/forward_pb2.py updated"
