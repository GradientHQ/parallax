#!/bin/bash
source scripts/check.sh

if [ $? -ne 0 ]; then
    exit 1
fi

helpFunction()
{
   echo ""
   echo "Usage: $0 [-s SCHEDULER_ADDR] [-r]"
   echo -e "\t-s SCHEDULER_ADDR (default: auto)"
   echo -e "\t-r (Optional) Use public relay servers"
   exit 1 # Exit script after printing help
}

SCHEDULER_ADDR="auto"
USE_RELAY=0

while getopts "s:r" opt
do
   case "$opt" in
      s ) SCHEDULER_ADDR="$OPTARG" ;;
      r ) USE_RELAY=1 ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$SCHEDULER_ADDR" ]
then
   SCHEDULER_ADDR="auto"
fi

echo "$SCHEDULER_ADDR"
if [ "$USE_RELAY" -eq 1 ]; then
    echo "USE_RELAY: enabled"
fi

export SGL_ENABLE_JIT_DEEPGEMM=0

CMD="python3 src/parallax/launch.py \
          --max-num-tokens-per-batch 4096 \
          --max-sequence-length 2048 \
          --max-batch-size 8 \
          --kv-block-size 1024 \
          --host 0.0.0.0 \
          --port 3000 \
          --scheduler-addr $SCHEDULER_ADDR"

if [ "$USE_RELAY" -eq 1 ]; then
    CMD="$CMD --relay-servers /dns4/relay-lattica.gradient.network/udp/18080/quic-v1/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf /dns4/relay-lattica.gradient.network/tcp/18080/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf"
    CMD="$CMD --initial-peers /dns4/bootstrap-lattica.gradient.network/udp/18080/quic-v1/p2p/12D3KooWJHXvu8TWkFn6hmSwaxdCLy4ZzFwr4u5mvF9Fe2rMmFXb /dns4/bootstrap-lattica.gradient.network/tcp/18080/p2p/12D3KooWJHXvu8TWkFn6hmSwaxdCLy4ZzFwr4u5mvF9Fe2rMmFXb"
fi

eval $CMD
