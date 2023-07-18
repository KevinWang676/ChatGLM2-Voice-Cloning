# ChatGLM2+Voice Cloning+SadTalker 🎶
## Chat with any character you like immersively through ChatGLM2-6B, FreeVC and SadTalker in real time 🥳
### [简体中文](https://github.com/KevinWang676/ChatGLM2-Voice-Cloning/blob/main/README_zh.md)
## Easy to use 💡

(1) Run
```
git clone https://github.com/KevinWang676/ChatGLM2-Voice-Cloning.git
cd ChatGLM2-Voice-Cloning
pip install -r requirements.txt
```

(2) Run
```
sudo apt update && sudo apt upgrade
apt install ffmpeg
```

(3) Upload `freevc-24.pth` to the folder `./ChatGLM2-Voice-Cloning/checkpoint/`. You can download `freevc-24.pth` through this [link](https://huggingface.co/spaces/kevinwang676/FreeVC/tree/main/checkpoints).

(4) Upload `pretrained_bak_5805000.pt` to the folder `./ChatGLM2-Voice-Cloning/speaker_encoder/ckpt/`. You can download `pretrained_bak_5805000.pt` through this [link](https://huggingface.co/spaces/kevinwang676/FreeVC/tree/main/speaker_encoder/ckpt).

(5) Run `python app_en.py`.

(6) Done! Now you can open Gradio interface and chat with any character you like through ChatGLM2-6B and voice cloning. 💕

P.S. The code is based on [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B), [FreeVC](https://github.com/OlaWod/FreeVC), and [SadTalker](https://github.com/OpenTalker/SadTalker).

### Quick start: [HuggingFace Demo](https://huggingface.co/spaces/kevinwang676/ChatGLM2-SadTalker-VC-en) 🤗

### If you like the my application, please star this repository. ⭐⭐⭐

Gradio Interface:

![image](https://github.com/KevinWang676/ChatGLM2-Voice-Cloning/assets/126712357/e6eeae6b-3e75-407e-acab-cac2fb93d6a4)
