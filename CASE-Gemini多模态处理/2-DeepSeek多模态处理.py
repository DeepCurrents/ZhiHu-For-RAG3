#!/usr/bin/env python
# coding: utf-8
"""
多模态理解（二）：视觉模型只吃图片时，怎么理解视频 —— 均匀抽帧

【和上一个文件的差别】
上一个文件（Gemini）能直接把视频链接丢给模型，由服务端解析；
本文件换用只支持图片输入的视觉模型（qwen-vl-max），所以视频要自己拆：
    视频 → 【抽帧】→ 若干张图片 → 一起塞进 content 列表 → 模型综合描述

这其实就是"能力不够就用工程手段补"的典型例子：
模型侧不支持视频，就在客户端把视频降维成一串关键帧。

【抽帧策略：均匀抽帧】
按总帧数等分取 frame_num 个位置，而不是取前 N 帧。
这样能覆盖整个时间轴，避免"只看开头就下结论"。
实践中更讲究的做法是「场景切换检测」——只在画面发生明显变化时取帧，
既能保证关键情节不丢，又能减少送给模型的图片数量（=省钱、省 token）。

【学习要点】
- cv2.VideoCapture：OpenCV 读视频，CAP_PROP_FRAME_COUNT 拿总帧数
- cap.set(CAP_PROP_POS_FRAMES, idx)：跳到指定帧（按帧号定位）
- cv2.imencode(".jpg", frame)：把 numpy 图像编码成内存中的 JPEG 字节流，
  再 base64 成 data URL 发给模型——全程不落盘，不需要临时文件
- 帧数不是越多越好：模型一次能看的图片有限，且帧数多了 token 成本线性上升

【运行前提】
需要设置环境变量 AGICTO_API_KEY，并安装 opencv-python。
"""

# In[5]:


import base64
import os

from openai import OpenAI

# agicto 配置（OpenAI 兼容接口）
AGICTO_BASE_URL = "https://api.agicto.cn/v1"
# DeepSeek 官方模型不支持视觉输入，这里换用 agicto 上的视觉模型
# 可按 agicto 账户实际可用列表替换
MODEL = "qwen-vl-max"

# 当前脚本所在目录，保证从任意工作目录运行都能定位到资源文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

client = OpenAI(
    api_key=os.environ["AGICTO_API_KEY"],
    base_url=AGICTO_BASE_URL,
)

# 文字输出
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "用中文解释AI大模型是如何工作的"}],
)

print(response.choices[0].message.content)


# In[6]:


# 图像理解：本地图片转成 data URL 后和文字一起放入 content 列表
# base64 内嵌的好处：不用图床、不用上传，适合本地/内网环境
image_path = os.path.join(BASE_DIR, "dog_and_girl.jpeg")
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                # 图片元素和文字元素写在同一个列表里，顺序即"提示词的组织顺序"
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
# agicto 的视觉模型（qwen-vl-max）只接受图片输入，不能像 Gemini 那样直接传 mp4。
# 这里从视频里均匀抽帧，再按图片理解的方式送给模型。
import cv2

video_path = os.path.join(BASE_DIR, "car.mp4")
# 抽帧数量：帧数越多时间覆盖越密，但 token 成本也线性上升，6 帧是演示用的折中值
frame_num = 6

cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total <= 0:
    raise ValueError("无法读取视频帧")

# 均匀抽帧的索引计算：把 [0, total-1] 这段区间等分成 frame_num 个点
# 例：total=100、frame_num=6 → 0, 19, 39, 59, 79, 99，覆盖首尾且间隔均匀
indexes = [int(i * (total - 1) / (frame_num - 1)) for i in range(frame_num)]
# content 列表先放"任务描述"，后面每抽一帧就 append 一张图片，最终形成"一帧图 + 一句话"的混合输入
content = [{"type": "text", "text": "详细描述视频里发生了什么？如果有对话，请把关键对话提取出来。"}]

print("正在从视频抽帧...")
for idx in indexes:
    # 跳到指定帧号再读取（按帧号定位，比逐帧 read 更快）
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"读取第 {idx} 帧失败")
    # 把 numpy 图像编码为内存里的 JPEG 字节流，避免写临时文件
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError(f"编码第 {idx} 帧失败")
    # tobytes() 拿到原始字节 → base64 → data URL，与上面的图片理解完全同构
    frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
        }
    )
# 显式释放视频句柄，避免文件被占用（Windows 上尤其重要）
cap.release()
print(f"抽帧完成，共 {frame_num} 帧，开始推理...")

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": content}],
)

print(response.choices[0].message.content)
