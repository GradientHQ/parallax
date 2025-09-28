## Usage example

### Step 1: Launch scheduler
First launch our scheduler on the main node.
```sh
parallax run -m {model-name} -n {number-of-worker-nodes}
```
For example:
```sh
parallax run -m Qwen/Qwen3-0.6B -n 2
```
Please notice and record the scheduler ip4 address generated in the terminal.

### Step 2: Join each distributed nodes
For each distributed nodes including the main node, open a terminal and join the server with the scheduler address.
```sh
parallax join -s {scheduler-address}
```
For example:
```sh
# first node
parallax join -s /ip4/192.168.1.1/tcp/5001/p2p/xxxxxxxxxxxx
# second node
parallax join -s /ip4/192.168.1.1/tcp/5001/p2p/xxxxxxxxxxxx
```
