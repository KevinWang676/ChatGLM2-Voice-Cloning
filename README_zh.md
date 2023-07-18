# ChatGLM2+声音克隆+视频对话 🎶
### 和喜欢的角色沉浸式对话、畅所欲言吧：ChatGLM2-6B+FreeVC+SadTalker 🌟
> 07/18/2023更新：增加SadTalker功能，开启视频对话新模态，在线程序见下方HuggingFace链接，代码正在更新中


https://github.com/KevinWang676/ChatGLM2-Voice-Cloning/assets/126712357/e33950a3-5558-4b53-8797-cf15fa9ed6ef


## 如何使用 💡

(1) 在终端中运行
```
git clone https://github.com/KevinWang676/ChatGLM2-Voice-Cloning.git
cd ChatGLM2-Voice-Cloning
pip install -r requirements.txt
```

(2) 在终端中运行
```
sudo apt update && sudo apt upgrade
apt install ffmpeg
```

(3) 将`freevc-24.pth`文件上传至`./ChatGLM2-Voice-Cloning/checkpoint/`文件夹。您可以通过[此链接](https://huggingface.co/spaces/kevinwang676/FreeVC/tree/main/checkpoints)下载`freevc-24.pth`文件。

(4) 将`pretrained_bak_5805000.pt`文件上传至`./ChatGLM2-Voice-Cloning/speaker_encoder/ckpt/`文件夹。您可以通过[此链接](https://huggingface.co/spaces/kevinwang676/FreeVC/tree/main/speaker_encoder/ckpt)下载`pretrained_bak_5805000.pt`文件。

(5) 在终端中运行`python app_new.py`。

(6) 完成！您现在就可以点击进入Gradio网页使用ChatGLM2-6B+声音克隆程序，和喜欢的角色开启沉浸式对话啦！ 💕

使用指南：
* 本项目基于[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)、[FreeVC](https://github.com/OlaWod/FreeVC)和[SadTalker](https://github.com/OpenTalker/SadTalker)，需要使用GPU。
* 如何使用此程序：输入并发送您对ChatGLM2的提问后，依次点击“开始和GLM2交流吧”、“生成对应的音频吧”、“开始AI声音克隆吧”、“开始视频聊天吧”三个按键即可。
* 如果您想让ChatGLM2进行角色扮演并与之对话，请先输入恰当的提示词，如“请你扮演成动漫角色蜡笔小新并和我进行对话”；您也可以为ChatGLM2提供自定义的角色设定。
* 当您使用声音克隆功能时，请先在此程序的对应位置上传一段您喜欢的音频，5~10秒即可；上传音频的质量会直接影响声音克隆的效果。
* 请不要生成任何会对个人或组织造成侵害的内容。


### 快速开始: [HuggingFace在线程序](https://huggingface.co/spaces/kevinwang676/ChatGLM2-SadTalker-VC) 🤗

### AutoDL部署：[AutoDL镜像](https://www.codewithgpu.com/i/KevinWang676/ChatGLM2-Voice-Cloning/ChatGLM2-Voice-Cloning)，运行环境及文件均已配置好，可一键使用 ⚡

### 如果您喜欢这个程序，欢迎给我的Github项目点赞支持！ ⭐⭐⭐

Gradio聊天界面：

![image](https://github.com/KevinWang676/ChatGLM2-Voice-Cloning/assets/126712357/2b4fe4c9-1c85-4e4c-94cb-2c96315f7abd)

