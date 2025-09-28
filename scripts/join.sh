#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -s SCHEDULER_ADDR -m MODEL_NAME -i LOCAL_IP"
   echo -e "\t-s Description of what is SCHEDULER_ADDR"
   echo -e "\t-m Description of what is MODEL_NAME"
   echo -e "\t-i Description of what is LOCAL_IP"
   exit 1 # Exit script after printing help
}

while getopts "s:m:i:" opt
do
   case "$opt" in
      s ) SCHEDULER_ADDR="$OPTARG" ;;
      m ) MODEL_NAME="$OPTARG" ;;
      i ) LOCAL_IP="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$SCHEDULER_ADDR" ] || [ -z "$MODEL_NAME" ] || [ -z "$LOCAL_IP" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$SCHEDULER_ADDR"
echo "$MODEL_NAME"
echo "$LOCAL_IP"

export SGL_ENABLE_JIT_DEEPGEMM=0

python3 src/parallax/launch.py \
          --model-path $MODEL_NAME \
          --max-num-tokens-per-batch 4096 \
          --max-sequence-length 2048 \
          --max-batch-size 8 \
          --kv-block-size 1024 \
          --host 0.0.0.0 \
          --port 3000 \
          --announce-http-addr $LOCAL_IP:3000 \
          --scheduler-addr $SCHEDULER_ADDR
