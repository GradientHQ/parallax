# Parallax Web UI

This is the frontend source code for Parallax, built with React and Vite.

The built assets in `dist/` are served by the Parallax scheduler and chat
servers. You only need the Node.js toolchain below when editing the web UI.

## Build

Run this command after preparing the frontend environment:

```bash
pnpm run build
```

The output directory is `./dist`.

## Local Debugging and Development

Prepare the frontend environment (macOS or Linux):

- Node.js `>=22`. `nvm` is one way to install and manage Node.js versions.
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

Open `http://localhost:5173`, edit code, and preview in your browser.
