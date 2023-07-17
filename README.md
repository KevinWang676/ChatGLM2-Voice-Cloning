# ChatGLM2 Voice Cloning 🎶
### [简体中文](https://github.com/KevinWang676/ChatGLM2-Voice-Cloning/blob/main/README_zh.md)
## Chat with any character you like through ChatGLM2-6B and voice cloning in real time. 🥳
## Easy to use 💡

(1) Run
```
git clone https://huggingface.co/spaces/kevinwang676/FreeVC.git
cd FreeVC
pip install -r requirements.txt
```

(2) Run
```
sudo apt update && sudo apt upgrade
apt install ffmpeg
```

(3) Create a new folder named `checkpoint` under `./FreeVC/` and upload `freevc-24.pth` to the new folder. You can download `freevc-24.pth` through this [link](https://huggingface.co/spaces/kevinwang676/FreeVC/tree/main/checkpoints).

(4) Delete the original file named `pretrained_bak_5805000.pt` in `./FreeVC/speaker_encoder/ckpt/` and upload a new version of `pretrained_bak_5805000.pt` to the same folder `./FreeVC/speaker_encoder/ckpt/`. You can download the new version through this [link](https://huggingface.co/spaces/kevinwang676/FreeVC/tree/main/speaker_encoder/ckpt).

(5) Upload `app_new.py` in this repository to the folder `./FreeVC/` and run `python app_new.py`.

(6) Done! Now you can chat with any character you like through ChatGLM2-6B and voice cloning.

### Quick start: [HuggingFace Demo](https://huggingface.co/spaces/kevinwang676/FreeVC) 🤗

### If you like the my application, please star this repository. ⭐⭐⭐
