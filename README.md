# ChatGLM2 Voice Cloning 🎶
### [简体中文]
## How to use 💡

(1) Run
```
git clone https://huggingface.co/spaces/kevinwang676/FreeVC.git
cd FreeVC
pip install -r requirements.txt
```

(2) Create a new folder named `checkpoint` and upload `freevc-24.pth` to that folder. You can download `freevc-24.pth` through this [link](https://huggingface.co/spaces/kevinwang676/FreeVC/tree/main/checkpoints).

(3) Delete the original file named `pretrained_bak_5805000.pt` in `./FreeVC/speaker_encoder/ckpt/` and upload a new version of `pretrained_bak_5805000.pt` to the same folder `./FreeVC/speaker_encoder/ckpt/`. You can download the new version through this [link](https://huggingface.co/spaces/kevinwang676/FreeVC/tree/main/speaker_encoder/ckpt).
