# ChatGLM2+声音克隆+视频对话 🎶
## 和喜欢的角色沉浸式视频对话吧：ChatGLM2-6B+FreeVC+SadTalker 📺💕🍻
> 07/19/2023更新：增加SadTalker功能，开启视频对话新模态，部署教程见下方[v2: ChatGLM2-6B+FreeVC+SadTalker](https://github.com/KevinWang676/ChatGLM2-Voice-Cloning/blob/main/README_zh.md#v2-chatglm2-6bfreevcsadtalker)

> 08/11/2023更新：增加AI歌手数字人功能，“想把我唱给你听”，本地部署教程见下方[AI歌手数字人](https://github.com/KevinWang676/ChatGLM2-Voice-Cloning/blob/main/README_zh.md#ai%E6%AD%8C%E6%89%8B%E6%95%B0%E5%AD%97%E4%BA%BA%E6%83%B3%E6%8A%8A%E6%88%91%E5%94%B1%E7%BB%99%E4%BD%A0%E5%90%AC-0811%E6%9B%B4%E6%96%B0)

## 如何使用 💡

### v1: ChatGLM2-6B+FreeVC

### 本地部署

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

您还可以使用AutoDL一键部署：
### AutoDL部署：[AutoDL镜像](https://www.codewithgpu.com/i/KevinWang676/ChatGLM2-Voice-Cloning/ChatGLM2-Voice-Cloning)，运行环境及文件均已配置好，可一键使用 ⚡

## 如何使用最新版SadTalker功能 📺

### v2: ChatGLM2-6B+FreeVC+SadTalker

### (1) 本地部署

### 配置环境

在终端中运行
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
sudo apt install build-essential
apt install ffmpeg
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

### 安装依赖

在终端中运行
```
git clone https://huggingface.co/spaces/kevinwang676/ChatGLM2-SadTalker.git
cd ChatGLM2-SadTalker
pip install -r requirements.txt
```

### 执行程序

在终端中运行`python app.py`；您现在可以点击进入Gradio网页使用ChatGLM2-6B+声音克隆+视频对话，和喜欢的角色开启沉浸式对话啦！ 💕

### (2) 快速开始：[Colab笔记本](https://colab.research.google.com/github/KevinWang676/ChatGLM2-Voice-Cloning/blob/main/ChatGLM2_VC_SadTalker.ipynb) ⚡ 注：需要使用Colab中的A100 GPU

### (3) HuggingFace在线程序：[ChatGLM2-6B+FreeVC+SadTalker](https://huggingface.co/spaces/kevinwang676/ChatGLM2-SadTalker-VC) 🤗

## AI歌手数字人：想把我唱给你听 (08/11更新)

### 本地部署

### 配置环境（同v2）

在终端中运行
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
sudo apt install build-essential
apt install ffmpeg
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

### 安装依赖

在终端中运行
```
git clone https://huggingface.co/spaces/kevinwang676/VoiceChanger.git
cd VoiceChanger
pip install -r requirements.txt
```

### 执行程序

在终端中运行`python app_multi.py`；您现在可以点击进入Gradio网页使用AI歌手数字人，让AI歌手唱给您听吧！ 🤵‍♀️💕

### 如果您喜欢这个程序，欢迎给我的Github项目点赞支持！ ⭐⭐⭐

#

SadTalker效果演示：

https://github.com/KevinWang676/ChatGLM2-Voice-Cloning/assets/126712357/e33950a3-5558-4b53-8797-cf15fa9ed6ef

AI歌手数字人效果演示：

Gradio聊天界面：

![image](https://github.com/KevinWang676/ChatGLM2-Voice-Cloning/assets/126712357/2b4fe4c9-1c85-4e4c-94cb-2c96315f7abd)

