## Getting Started

We will walk through you the easiest way to quickly set up your own AI cluster

### With Frontend

#### Step 1: Launch scheduler

First launch our scheduler on the main node, we recommend you to use your most convenient computer for this.
- For Linux/macOS:
```sh
parallax run
```

- For Windows, start Powershell console as administrator and run:
```sh
parallax run
```

To allow the API to be accessible from other machines, add the argument `--host 0.0.0.0` when launching scheduler.
```sh
parallax run --host 0.0.0.0
```

When running `parallax run` for the first time or after an update, some basic info (like version and gpu name) might be sent to help improve the project. To disable this, use the `-u` flag:
```sh
parallax run -u
```

#### Step 2: Set cluster and model config

Open http://localhost:3001 and you should see the setup interface.

![Model select](docs/images/node_config.png)

Select your desired node and model config and click continue.

> **Note:**
When running in remote mode, Parallax will use a public relay server to help establish connections between the scheduler and nodes. The public relay server will receive the IP information of both the scheduler and the nodes in order to facilitate this connection.

#### Step 3: Connect your nodes

Copy the generated join command line to your node and run. For remote connection, you can find your scheduler-address in the scheduler logs.

```sh
# local area network env
parallax join
# public network env
parallax join -s {scheduler-address}
# example
parallax join -s 12D3KooWLX7MWuzi1Txa5LyZS4eTQ2tPaJijheH8faHggB9SxnBu
```

![Node join](docs/images/node_join.png)

You should see your nodes start to show up with their status. Wait until all nodes are successfully connected, and you will automatically be directed to the chat interface.

When running `parallax join` for the first time or after an update, some basic info (like version and gpu name) might be sent to help improve the project. To disable this, use the `-u` flag:
```sh
parallax join -u
```

#### Step 4: Chat

Done! You have your own AI cluster now.

![Chat](docs/images/chat_interface.png)

#### Accessing the chat interface from another non-scheduler computer

You can access the chat interface from any non-scheduler computer, not just those running a node server. Simply start the chat server with:

```sh
# local area network env
parallax chat
# public network env
parallax chat -s {scheduler-address}
# example
parallax chat -s 12D3KooWLX7MWuzi1Txa5LyZS4eTQ2tPaJijheH8faHggB9SxnBu
```

After launching, visit [http://localhost:3002](http://localhost:3002) in your browser to use the chat interface.

To allow the API to be accessible from other machines, add the argument `--host 0.0.0.0` when launching chat interface.
```sh
parallax chat --host 0.0.0.0
```

### Without frontend
#### Step 1: Launch scheduler
First launch our scheduler on the main node.
```sh
parallax run -m {model-name} -n {number-of-worker-nodes}
```
For example:
```sh
parallax run -m Qwen/Qwen3-0.6B -n 2
```
Please notice and record the scheduler ip4 address generated in the terminal.

#### Step 2: Connect your nodes
For each distributed nodes including the main node, open a terminal and join the server with the scheduler address.
```sh
# local area network env
parallax join
# public network env
parallax join -s {scheduler-address}
```
For example:
```sh
# first node
parallax join -s 12D3KooWLX7MWuzi1Txa5LyZS4eTQ2tPaJijheH8faHggB9SxnBu
# second node
parallax join -s 12D3KooWLX7MWuzi1Txa5LyZS4eTQ2tPaJijheH8faHggB9SxnBu
```

#### Step 3: Call chat api with Scheduler
```sh
curl --location 'http://localhost:3001/v1/chat/completions' --header 'Content-Type: application/json' --data '{
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "hello"
      }
    ],
    "stream": true
}'
```

### Skipping Scheduler
Developers can start Parallax backend engine without a scheduler. Pipeline parallel start/end layers should be set manually.
An example of serving Qwen3-0.6B with 2-nodes:
- First node:
```sh
python3 ./parallax/src/parallax/launch.py \
--model-path Qwen/Qwen3-0.6B \
--port 3000 \
--max-batch-size 8 \
--start-layer 0 \
--end-layer 14
```
- Second node:
```sh
python3 ./parallax/src/parallax/launch.py \
--model-path Qwen/Qwen3-0.6B \
--port 3000 \
--max-batch-size 8 \
--start-layer 14 \
--end-layer 28
```

Call chat API on one of the nodes:
```sh
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

```
### Uninstalling Parallax

For macOS or Linux, if you've installed Parallax via pip and want to uninstall it, you can use the following command:

```sh
pip uninstall parallax
```

For Docker installations, remove Parallax images and containers using standard Docker commands:

```sh
docker ps -a               # List running containers
docker stop <container_id> # Stop running containers
docker rm <container_id>   # Remove stopped containers
docker images              # List Docker images
docker rmi <image_id>      # Remove Parallax images
```

For Windows, simply go to Control Panel → Programs → Uninstall a program, find "Gradient" in the list, and uninstall it.
