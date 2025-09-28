source scripts/check.sh

if [ $? -ne 0 ]; then
    exit 1
fi


DHT_PORT=${1:-5001}
PORT=${2:-3001}
python3 src/backend/main.py --dht-port $DHT_PORT --port $PORT
