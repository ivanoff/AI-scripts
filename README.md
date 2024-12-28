![Image](./s17-1.png) ![Image](./s20-1.png) ![Image](./s23-1.png)

# AI scripts

This repository contains `scripts for AI models`. The scripts are written in Python and use the Hugging Face library. The models are trained on the Hugging Face platform and can be used for `music`, `sound`, and `text` generation, `image generation and recognition`.

## Table of Contents

- [AI scripts](#ai-scripts)
  - [Table of Contents](#table-of-contents)
  - [Folder Structure](#folder-structure)
  - [Installation](#installation)
    - [Pre-installation](#pre-installation)
    - [Install CUDA](#install-cuda)
    - [Install Bun and run node-llama-cpp chat](#install-bun-and-run-node-llama-cpp-chat)
    - [Install Ollama and run models](#install-ollama-and-run-models)
    - [Install Python venv and packages](#install-python-venv-and-packages)
    - [Login to Hugging Face](#login-to-hugging-face)
  - [Usefull commands](#usefull-commands)
    - [Check GPU](#check-gpu)


## Folder Structure

- [image-generation](./image-generation/) - image generation scripts
```
    ├── flux.1.py - generate image using flux v1
    ├── instruct-pix2pix.py - generate image using pix2pix
    └── stable-diffusion-3.5.py - generate image using diffusion v3.5
```

- [image-recognition](./image-recognition/) - image recognition scripts
```
    ├── .env - env file with token
    ├── aa.jpg - jpeg image
    ├── ab.jpg - jpeg image
    ├── ac.png - png image
    ├── ad.png - png image
    ├── api.py - server for image recognition
    └── moondream2.py - recognize image using moondream v2
```

- [music-generation](./music-generation/) - music generation scripts
```
    ├── files/
    │   └── 20241217154151.mp3 - mp3 file
    ├── musicgen-small.py - generate music using musicgen small model
    └── telegram-bot.py - telegram bot for music generation
```

- [sound-generation](./sound-generation/) - sound generation scripts
```
    └── stable-audio-open-1.0.py - generate sound using stable-audio-open v1
```

- [text-generation](./text-generation/) - text generation scripts
```
    ├── js/
    │   └── i.js - Python script
    ├── local-context.py - generate text using local context
    └── local.py - generate text using local model
```

- [text-train](./text-train/) - text train scripts
```
    ├── gpt/
    │   ├── train.py - train gpt2 model
    │   ├── use.py - use trained model
    │   └── your_train_data.txt
    ├── train-coffee.py - train model using coffee data
    ├── train.py - train mistralai/Mistral-7B-Instruct-v0.2 model
    ├── use-coffee.py - use coffee trained model
    ├── use.py - use trained mistralai/Mistral-7B-Instruct-v0.2 model
    └── your_train_data.txt
```

## Installation

### Pre-installation

```bash
apt -y update
apt -y upgrade
apt -y install gcc build-essential automake checkinstall git htop mc fail2ban
```

### Install CUDA

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall.
sudo apt install nvidia-cuda-toolkit
sudo reboot
```

### Install Bun and run node-llama-cpp chat

```bash
curl -fsSL https://bun.sh/install | bash
bunx -y node-llama-cpp chat
```

### Install Ollama and run models

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run mistral
ollama run llama3.3
ollama run gemma:7b
ollama run qwq
ollama run qwen2.5-coder
```

### Install Python venv and packages

```bash
sudo apt install python3.12-venv
python3 -m venv ~/venv
source ~/venv/bin/activate

pip install transformers
pip install diffusers
pip install flask pillow transformers python-dotenv
pip install huggingface_hub
pip install accelerate protobuf sentencepiece
pip install vllm
```

### Login to Hugging Face

```bash
huggingface-cli login
```

## Usefull commands

### Check GPU

```bash
sudo nvidia-smi -l
```
