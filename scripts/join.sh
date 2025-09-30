#!/bin/bash
source scripts/check.sh

if [ $? -ne 0 ]; then
    exit 1
fi

helpFunction()
{
   echo ""
   echo "Usage: $0 [-s SCHEDULER_ADDR]"
   echo -e "\t-s SCHEDULER_ADDR (default: auto)"
   exit 1 # Exit script after printing help
}

SCHEDULER_ADDR="auto"

while getopts "s:" opt
do
   case "$opt" in
      s ) SCHEDULER_ADDR="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$SCHEDULER_ADDR" ]
then
   SCHEDULER_ADDR="auto"
fi

echo "$SCHEDULER_ADDR"

export SGL_ENABLE_JIT_DEEPGEMM=0

python3 src/parallax/launch.py \
          --max-num-tokens-per-batch 4096 \
          --max-sequence-length 2048 \
          --max-batch-size 8 \
          --kv-block-size 1024 \
          --host 0.0.0.0 \
          --port 3000 \
          --scheduler-addr $SCHEDULER_ADDR
