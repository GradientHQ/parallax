#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -n INIT_NODES_NUM -m MODEL_NAME"
   echo -e "\t-n Description of what is INIT_NODES_NUM"
   echo -e "\t-m Description of what is MODEL_NAME"
   exit 1 # Exit script after printing help
}

while getopts "n:m:" opt
do
   case "$opt" in
      n ) INIT_NODES_NUM="$OPTARG" ;;
      m ) MODEL_NAME="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$INIT_NODES_NUM" ] || [ -z "$MODEL_NAME" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "$INIT_NODES_NUM"
echo "$MODEL_NAME"

python3 src/backend/main.py --dht-port 5001 --port 3001 --model-name $MODEL_NAME --init-nodes-num $INIT_NODES_NUM

