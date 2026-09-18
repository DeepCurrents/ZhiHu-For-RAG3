#!/usr/bin/env python
# coding: utf-8
"""
多模态理解入门：用同一个接口同时理解「文字 / 图片 / 视频」

【这一步在 RAG 流水线中的位置】
    原始素材（图/视频） → 【多模态理解】→ 文字描述 → 切片 → Embedding → 向量库 → 检索 → LLM 生成
对多模态 RAG 来说，这一层是"翻译器"：把模型读不懂的图片、视频翻译成文字，
后续的切片、向量化、检索就都能复用纯文本 RAG 的成熟方案。

【核心结论】
三种模态看起来差别很大，但在 agicto（OpenAI 兼容协议）里调用方式是统一的：
    端点都是  POST /v1/chat/completions
    区别只在  messages[].content 的写法

    纯文字：content 直接是字符串
    图文混合：content 是一个列表，元素形如
        {"type": "text",      "text": "帮我解释下这张照片"}
        {"type": "image_url", "image_url": {"url": <图片地址>}}

【图片地址的两种写法】
1. 公网链接：{"url": "https://xxx.jpg"}
2. 本地文件：先 base64 编码，再拼成 data URL ——  f"data:image/jpeg;base64,{b64}"
   本文件用第 2 种，因为 dog_and_girl.jpeg 就在本地。

【视频为什么也塞在 image_url 里】
agicto 的视频理解复用了同一个字段：把视频的公网链接放进 image_url.url 即可，
由服务端自己去解析视频。注意它只接受「链接」，不支持上传本地视频文件。

【运行前提】
需要设置环境变量 AGICTO_API_KEY。
"""

# In[5]:


import base64
import os

from openai import OpenAI

# agicto 配置（OpenAI 兼容接口）
# 把 base_url、模型名抽成常量，业务代码里不散落硬编码，换供应商时只改这一处
AGICTO_BASE_URL = "https://api.agicto.cn/v1"
# 可按 agicto 账户实际可用列表替换
AGICTO_CHAT_MODEL = "gemini-2.5-pro"

# 当前脚本所在目录，保证从任意工作目录运行都能定位到资源文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# OpenAI SDK 只换 base_url 就能指向 agicto，调用方式与官方 OpenAI 完全一致
# os.environ[...] 与 os.getenv(...) 的区别：前者取不到 key 会直接抛 KeyError，属于"快速失败"
client = OpenAI(
    api_key=os.environ["AGICTO_API_KEY"],
    base_url=AGICTO_BASE_URL,
)

# 文字输出
# 最基础的用法：content 就是一个字符串
response = client.chat.completions.create(
    model=AGICTO_CHAT_MODEL,
    messages=[{"role": "user", "content": "用中文解释AI大模型是如何工作的"}],
)

# 标准返回结构：choices[0].message.content 才是模型生成的文本
print(response.choices[0].message.content)


# In[6]:


# 图像理解：本地图片转成 data URL 后和文字一起放入 content 列表
# 第 1 步：读二进制 → base64 编码 → 拼成 "data:image/jpeg;base64,xxx"
# 这样图片就不需要先上传到图床，直接内嵌在请求里发给模型
image_path = os.path.join(BASE_DIR, "dog_and_girl.jpeg")
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model=AGICTO_CHAT_MODEL,
    messages=[
        {
            "role": "user",
            # content 从字符串变成列表：文字和图片是并列的两个元素
            # 注意 MIME 类型要写对（.jpeg → image/jpeg），否则服务端可能解析失败
            "content": [
                {"type": "text", "text": "帮我解释下这张照片"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)


# In[8]:


# 视频理解
# agicto 的视频理解走同一个 /v1/chat/completions 接口，
# 视频以「链接」形式放进 image_url.url（不支持本地文件上传）。
VIDEO_URL = "https://dataset-1255932437.cos.ap-nanjing.myqcloud.com/mp4/car.mp4"  # 汽车剐蹭视频

response = client.chat.completions.create(
    model=AGICTO_CHAT_MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                # 提示词直接影响抽取质量：明确要求"提取关键对话"，模型才会把语音转成文字
                # 对 RAG 来说，描述越结构化、信息越全，后面向量检索越准
                {
                    "type": "text",
                    "text": "详细描述视频里发生了什么？如果有对话，请把关键对话提取出来。",
                },
                {"type": "image_url", "image_url": {"url": VIDEO_URL}},
            ],
        }
    ],
)

print(response.choices[0].message.content)
