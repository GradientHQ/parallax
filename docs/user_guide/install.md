
## Installation

### Prerequisites
- Python>=3.11.0,<3.14.0
- Git and curl
- Ubuntu-24.04 is required when running on Blackwell GPUs (B100/B200/RTX 50 series) from source; for other GPUs and Docker, see the [Docker section](#docker) below.

Below are installation methods for different operating systems.

|  Operating System  |  Windows App  |  From Source | Docker |
|:-------------|:----------------------------:|:----------------------------:|:----------------------------:|
|Windows       | ✅️ | Not recommended | Not recommended |
|Linux | ❌️ | ✅️ | ✅️ |
|macOS | ❌️ | ✅️ | ❌️ |

### From Source

The source install script installs `uv` if needed, creates `.venv`, installs
Parallax, and builds the `vllm-rs` frontend binary into `.venv/bin`.

The bundled `vllm-rs` Rust HTTP frontend is built from vLLM source at
`v0.22.0` by default (controlled by the `VLLM_REF` env var in `install.sh`).
This is independent of the optional `vllm` Python extras (`vllm==0.14.0`),
which only apply when using the vLLM GPU backend.

```sh
git clone https://github.com/GradientHQ/parallax.git
cd parallax
./install.sh
source .venv/bin/activate
```

The script automatically installs `mac` extras on macOS and `gpu` extras on
Linux. You can also choose explicitly:

```sh
# Linux/WSL GPU
./install.sh --extras gpu

# macOS Apple silicon
./install.sh --extras mac
```

For development dependencies:
```sh
./install.sh --extras gpu,dev
# or
./install.sh --extras mac,dev
```

To use a specific supported Python version, pass `--python`, for example
`./install.sh --python 3.12`.

Next time to re-activate this virtual environment, run ```source .venv/bin/activate```.

### Windows Application
[Click here](https://github.com/GradientHQ/parallax_win_cli/releases/latest/download/Parallax_Win_Setup.exe) to get latest Windows installer.

After installing .exe, right click Windows start button and click ```Windows Terminal(Admin)``` to start a Powershell console as administrator.

❗ Make sure you open your terminal with administrator privileges.
<details>
<summary>Ways to run Windows Terminal as administrator</summary>

- Start menu: Right‑click Start and choose "Windows Terminal (Admin)", or search "Windows Terminal", right‑click the result, and select "Run as administrator".
- Run dialog: Press Win+R → type `wt` → press Ctrl+Shift+Enter.
- Task Manager: Press Ctrl+Shift+Esc → File → Run new task → enter `wt` → check "Create this task with administrator privileges".
- File Explorer: Open the target folder → hold Ctrl+Shift → right‑click in the folder → select "Open in Terminal".
</details>
<br>

Start Windows dependencies installation by simply typing this command in console:
```sh
parallax install
```

Installation process may take around 30 minutes.

To see a description of all Parallax Windows configurations you can do:
```sh
parallax --help
```

### Docker
For Linux+GPU devices, Parallax provides a docker environment for quick setup. Choose the docker image according to the device's GPU architechture.

|  GPU Architecture  |  GPU Series  | Image Pull Command |
|:-------------|:----------------------------|:----------------------------|
|Blackwell/Ampere/Hopper| RTX50 series/RTX40 series/B100/B200/A100/H100... |```docker pull gradientservice/parallax:latest```|
|DGX Spark | GB10 |```docker pull gradientservice/parallax:latest-spark```|

Run a docker container as below. Please note that generally the argument ```--gpus all``` is necessary for the docker to run on GPUs.
```sh
# For Blackwell/Ampere/Hopper
docker run -it --gpus all --network host gradientservice/parallax:latest bash
# For DGX Spark
docker run -it --gpus all --network host gradientservice/parallax:latest-spark bash
```
The container starts under parallax workspace and you should be able to run parallax directly.

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
