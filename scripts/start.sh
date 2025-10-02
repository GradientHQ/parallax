source scripts/check.sh

if [ $? -ne 0 ]; then
    exit 1
fi


helpFunction()
{
   echo ""
   echo "Usage: $0 [-n INIT_NODES_NUM] [-m MODEL_NAME] [-r]"
   echo -e "\t-n (Optional) Number of initial nodes"
   echo -e "\t-m (Optional) Model name"
   echo -e "\t-r (Optional) Use public relay servers"
   exit 1 # Exit script after printing help
}

USE_RELAY=0

# Parse optional arguments
while getopts "n:m:rh" opt
do
   case "$opt" in
      n ) INIT_NODES_NUM="$OPTARG" ;;
      m ) MODEL_NAME="$OPTARG" ;;
      r ) USE_RELAY=1 ;;
      h ) helpFunction ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Show what was provided (for debugging)
if [ ! -z "$INIT_NODES_NUM" ]; then
    echo "INIT_NODES_NUM: $INIT_NODES_NUM"
fi
if [ ! -z "$MODEL_NAME" ]; then
    echo "MODEL_NAME: $MODEL_NAME"
fi
if [ "$USE_RELAY" -eq 1 ]; then
    echo "USE_RELAY: enabled"
fi

# Build the python command with optional arguments
CMD="python3 src/backend/main.py --dht-port 5001 --port 3001"
if [ ! -z "$MODEL_NAME" ]; then
    CMD="$CMD --model-name $MODEL_NAME"
fi
if [ ! -z "$INIT_NODES_NUM" ]; then
    CMD="$CMD --init-nodes-num $INIT_NODES_NUM"
fi
if [ "$USE_RELAY" -eq 1 ]; then
    CMD="$CMD --relay-servers /dns4/relay-lattica.gradient.network/udp/18080/quic-v1/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf /dns4/relay-lattica.gradient.network/tcp/18080/p2p/12D3KooWDaqDAsFupYvffBDxjHHuWmEAJE4sMDCXiuZiB8aG8rjf"
fi

eval $CMD
