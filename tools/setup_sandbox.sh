#!/bin/bash
# One-time setup of a Daytona H100 sandbox for tools/night.sh. Run inside the sandbox.
# Before it: ~/.ssh/id_ed25519 is a deploy key with write access to this repo, and ~/.okc_token holds
# an okc API token. After it, a person runs `claude` once to log in, then starts tools/night.sh.
set -e
cd ~
# The judge's stack: torch 2.14.1+cu130, triton 3.8.0.
uv venv -q -p 3.12 okc
. okc/bin/activate
uv pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install -q triton==3.8.0 numpy
# okc, checksum-verified by its installer.
curl -fsSL https://openkernelchallenge.dev/install.sh | OKC_NO_MODIFY_PATH=1 sh
# Claude Code.
curl -fsSL https://claude.ai/install.sh | bash
# The repo, over the deploy key.
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
git clone -q git@github.com:tech-cosmos/okc-muse-attention-decode.git repo
git -C repo config user.name "okc night agent"
git -C repo config user.email "shivam@cosmosinfinity.in"
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
OKC_TOKEN="$(cat ~/.okc_token)" ~/.okc/bin/okc whoami
echo "export PATH=\$HOME/.local/bin:\$HOME/.okc/bin:\$PATH" >> ~/.bashrc
