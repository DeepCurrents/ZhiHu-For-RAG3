#!/usr/bin/env python
# coding: utf-8
"""
视频 Embedding：同样是"先转描述，再向量化"

【链路】
    视频（公网链接）→ 【视觉模型生成文字描述】→ 文本 → 【文本 Embedding】→ 向量

和 2-图片embedding.py 是同一套思路，唯一区别是输入从本地图片换成了视频链接。

【视频为什么必须用公网 URL】
- agicto 的视频理解只接受「链接」，不支持上传本地视频文件
- 服务端需要能主动拉取到这个视频，所以本地路径、局域网地址都不行，
  必须是对象存储（如 COS / OSS / S3）等公网可访问的地址

【为什么把"关键对话"也要求进描述里】
视频的信息有两个来源：画面 + 语音。
视觉模型能直接"看到"画面，但语音内容需要它转述出来。
所以在提示词里明确要求提取对话——这一步相当于顺带把 ASR（语音转文字）也做了，
之后用户问"视频里那个人说了什么"才能被检索到。

【运行前提】
需要设置环境变量 AGICTO_API_KEY。
"""

# In[1]:


import json
import os

from openai import OpenAI

# agicto 配置（OpenAI 兼容接口）
AGICTO_BASE_URL = "https://api.agicto.cn/v1"
# 可按 agicto 账户实际可用列表替换
AGICTO_VISION_MODEL = "gemini-2.5-pro"
AGICTO_EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(
    api_key=os.environ["AGICTO_API_KEY"],
    base_url=AGICTO_BASE_URL,
)

# 视频只支持以链接形式输入，暂不支持直接传入本地视频
# 实际使用中请将url地址替换为您的视频url地址
video = "https://dataset-1255932437.cos.ap-nanjing.myqcloud.com/mp4/car.mp4"

# 1. 视频转文字描述
# agicto 的 /v1/embeddings 只接受文本输入，没有多模态向量化接口，
# 所以先用视觉模型把视频转成文字描述，再对描述做文本向量化。
desc_resp = client.chat.completions.create(
    model=AGICTO_VISION_MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                # 描述要求里同时覆盖"视觉信息（主体/场景/事件）"和"听觉信息（关键对话）"，
                # 因为视频检索的需求既有"画面里有什么"也有"他们说了什么"
                {
                    "type": "text",
                    "text": "请详细描述这个视频，用于后续检索：包含画面主体、场景、发生的事件和关键对话。",
                },
                # 视频链接同样放在 image_url.url 字段，这是 agicto 的统一约定
                {"type": "image_url", "image_url": {"url": video}},
            ],
        }
    ],
)
description = desc_resp.choices[0].message.content
print(f"视频描述: {description}\n")

# 2. 对描述文本做向量化
# 到这里视频已经彻底"退化成一段文字"，后续检索逻辑与纯文本 RAG 完全一致
resp = client.embeddings.create(
    model=AGICTO_EMBEDDING_MODEL,
    input=description,
)

embedding = resp.data[0].embedding
result = {
    "model": resp.model,
    "usage": resp.usage.model_dump(),
    "description": description,
    "embedding_dim": len(embedding),
    "embedding": embedding,
}
print(json.dumps(result, ensure_ascii=False, indent=4))
