mac env:
    python: .venv/bin/python

gpu env:
    ssh H200 server, then use conda activate parallax


start a local server:
    python src/parallax/launch.py --model-path <MODEL_NAME> --log-level DEBUG

test the server:
    curl --location 'http://localhost:3000/v1/chat/completions' --header 'Content-Type: application/json' --data '{
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "hello"
            }
        ],
        "stream": true
    }'

final check:
    run when user ask to git commit
    pre-commit run --all-files
    pytest
