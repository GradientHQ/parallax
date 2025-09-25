## Usage example

### Step1. Launch scheduler
In the root path of parallax, first launch our scheduler on the main node.
```bash
bash scripts/launch.sh -m {model-name} -n {number-of-worker-nodes}
```
For example:
```bash
bash scripts/launch.sh -m Qwen/Qwen3-0.6B -n 2
```
Please notice and record the scheduler address generated in this step.

### Step2. Join each distributed nodes
For each distributed nodes including the main node, open a terminal and join the server with the scheduler address generated in the previous step.
```bash
bash scripts/join.sh -m {model-name} -i {ip-address-of-current-node} -s {scheduler-address}
```
For example:
```bash
# first node
bash scripts/launch.sh -m Qwen/Qwen3-0.6B -i 192.168.1.1 -s /ip4/192.168.1.1/tcp/5001/p2p/xxxxxxxxxxxx
# second node
bash scripts/launch.sh -m Qwen/Qwen3-0.6B -i 192.168.1.2 -s /ip4/192.168.1.1/tcp/5001/p2p/xxxxxxxxxxxx
```
