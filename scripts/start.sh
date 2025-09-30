source scripts/check.sh

if [ $? -ne 0 ]; then
    exit 1
fi


helpFunction()
{
   echo ""
   echo "Usage: $0 [-n INIT_NODES_NUM] [-m MODEL_NAME]"
   echo -e "\t-n (Optional) Number of initial nodes"
   echo -e "\t-m (Optional) Model name"
   exit 1 # Exit script after printing help
}

# Parse optional arguments
while getopts "n:m:h" opt
do
   case "$opt" in
      n ) INIT_NODES_NUM="$OPTARG" ;;
      m ) MODEL_NAME="$OPTARG" ;;
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

# Build the python command with optional arguments
CMD="python3 src/backend/main.py --dht-port 5001 --port 3001"
if [ ! -z "$MODEL_NAME" ]; then
    CMD="$CMD --model-name $MODEL_NAME"
fi
if [ ! -z "$INIT_NODES_NUM" ]; then
    CMD="$CMD --init-nodes-num $INIT_NODES_NUM"
fi

eval $CMD
