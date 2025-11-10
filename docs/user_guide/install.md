
## Installation

### Prerequisites
- Python>=3.11.0,<3.14.0
- Ubuntu-24.04 for Blackwell GPUs

Below are installation methods for different operating systems.

|  Operating System  |  Windows App  |  From Source | Docker |
|:-------------|:----------------------------:|:----------------------------:|:----------------------------:|
|Windows       | ✅️ | Not recommended | Not recommended |
|Linux | ❌️ | ✅️ | ✅️ |
|macOS | ❌️ | ✅️ | ❌️ |

### From Source
#### For Linux/WSL (GPU):
```sh
git clone https://github.com/GradientHQ/parallax.git
cd parallax
pip install -e '.[gpu]'
```

#### For macOS (Apple silicon):

We recommend macOS users to create an isolated Python virtual environment before installation.

```sh
git clone https://github.com/GradientHQ/parallax.git
cd parallax

# Enter Python virtual environment
python3 -m venv ./venv
source ./venv/bin/activate

pip install -e '.[mac]'
```

Next time to re-activate this virtual environment, run ```source ./venv/bin/activate```.

<details>
<summary> Note for macOS users regarding network permissions</summary>

On macOS, you need to allow your terminal or IDE (such as Terminal, iTerm2, VS Code, Cursor, etc.) access to the local network in order for Parallax to work correctly. If the application prompts you for network access the first time you run Parallax, click "Allow." If you have already denied access, follow these steps to enable it:

1. Open System Settings from the Apple menu.
2. Click on Privacy & Security in the sidebar.
3. Click on Local Network.
4. For each app listed, turn the ability to access your local network on or off using the toggle switch.

This will ensure Parallax has the proper network permissions for local communication.
</details>

#### Extra step for development:
```sh
pip install -e '.[dev]'
```

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
|Blackwell       | RTX50 series/B100/B200... |```docker pull gradientservice/parallax:latest-blackwell```|
|Ampere/Hopper | RTX30 series/RTX40 series/A100/H100... |```docker pull gradientservice/parallax:latest-hopper```|

Run a docker container as below. Please note that generally the argument ```--gpus all``` is necessary for the docker to run on GPUs.
```sh
# For Blackwell
docker run -it --gpus all --network host gradientservice/parallax:latest-blackwell bash
# For Ampere/Hopper
docker run -it --gpus all --network host gradientservice/parallax:latest-hopper bash
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
