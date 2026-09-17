#!/usr/bin/env python
# coding: utf-8

# In[5]:


import base64
import os

from openai import OpenAI

# DeepSeek-V4.1-Flash 的官方调用名是 deepseek-flash
# deepseek-v4.1-flash 不是 API 模型名，官方不会识别
MODEL = "deepseek-flash"

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# 文字输出
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "用中文解释AI大模型是如何工作的"}],
)

print(response.choices[0].message.content)


# In[6]:


# 图像理解：本地图片转成 data URL 后和文字一起放入 content 列表
with open("dog_and_girl.jpeg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": "帮我解释下这张照片"},
            ],
        }
    ],
)

print(response.choices[0].message.content)


# In[8]:


# 视频理解
# DeepSeek 只支持图片多模态，不能像 Gemini 那样直接上传 mp4。
# 这里从视频里均匀抽帧，再按图片理解的方式送给模型。
import cv2

video_path = "car.mp4"
frame_num = 6

cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total <= 0:
    raise ValueError("无法读取视频帧")

indexes = [int(i * (total - 1) / (frame_num - 1)) for i in range(frame_num)]
content = [{"type": "text", "text": "详细描述视频里发生了什么？如果有对话，请把关键对话提取出来。"}]

print("正在从视频抽帧...")
for idx in indexes:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"读取第 {idx} 帧失败")
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError(f"编码第 {idx} 帧失败")
    frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
        }
    )
cap.release()
print(f"抽帧完成，共 {frame_num} 帧，开始推理...")

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": content}],
)

print(response.choices[0].message.content)
