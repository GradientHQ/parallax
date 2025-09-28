# Parallax Web UI

This is the front-end source code for Parallax, based on React and build by Vite.

## Build

Run the command to build this project (you need to prepare the front-end environment):

```bash
pnpm run build
```

The output directory is `./dist`.

## Local Debugging and Development

Prepare front-end environment (MacOS or Linux):

- `nvm` to install and manage versions of Node.js.
- `pnpm` as package manager to install dependencies.

```bash
# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"

# Download and install Node.js:
nvm install 22

# Verify the Node.js version:
node -v # Should print "v22.19.0".

# Download and install pnpm:
corepack enable pnpm

# Verify pnpm version:
pnpm -v
```

Install dependencies:

```bash
pnpm install
```

Run the hot-reload preview service:

```bash
pnpm run dev
```

Open the url `http://localhost:5173`, edit code and preview in browser.
