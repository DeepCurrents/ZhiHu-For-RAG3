#!/usr/bin/env python
# coding: utf-8
"""
图片 Embedding：两步走 —— 先"看图说话"，再"把话变向量"

【为什么不能一步到位】
理论上应该有一个多模态 Embedding 模型，能把图片和文字映射到同一个空间，
直接对图片求向量。但 agicto 的 /v1/embeddings 只接受文本输入，没有多模态向量化接口，
所以这里换了一条工程路线（也是业界常见的降级方案）：

    图片 → 【视觉模型生成文字描述】→ 文本 → 【文本 Embedding】→ 向量

【这样做的代价，必须知道】
- 会丢失信息：画面细节、风格、情绪很难被描述完全，"描述用不到的维度"就检索不到
- 描述质量决定检索质量：描述偏了，这个图就永远召不回来
- 反过来也有好处：图片和文本共享同一个向量空间，
  用户用文字问"迪士尼有什么人物合影"，也能召回图片，多模态检索天然打通

【关键设计：描述要"为检索而写"】
提示词里特意要求包含「画面主体、场景、人物、颜色、图中文字」。
这不是随便写的——这些正是用户提问时可能用到的词，让描述和查询词有交集，召回率才高。
如果图里有招牌、标语，把文字也描述进去，等于免费做了 OCR。

【运行前提】
需要设置环境变量 AGICTO_API_KEY。
"""

# In[2]:


import base64
import json
import os

from openai import OpenAI

# agicto 配置（OpenAI 兼容接口）
AGICTO_BASE_URL = "https://api.agicto.cn/v1"
# 可按 agicto 账户实际可用列表替换
# 本例用 Gemini 做视觉理解，也可以换成 qwen-vl-max 等其他视觉模型
AGICTO_VISION_MODEL = "gemini-2.5-pro"
AGICTO_EMBEDDING_MODEL = "text-embedding-3-small"

# 当前脚本所在目录，保证从任意工作目录运行都能定位到资源文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

client = OpenAI(
    api_key=os.environ["AGICTO_API_KEY"],
    base_url=AGICTO_BASE_URL,
)

# 1. 读取图片并转换为Base64
# 用 os.path.join 拼绝对路径，避免"换个目录运行就找不到文件"
image_path = os.path.join(BASE_DIR, "disney_knowledge_base", "images", "1-聚在一起说奇妙.jpg")
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
# 拼成 data URL，图片内容直接内嵌在请求体里，无需先上传到图床
image_data = f"data:image/jpeg;base64,{base64_image}"

# 2. 图片转文字描述
# agicto 的 /v1/embeddings 只接受文本输入，没有多模态向量化接口，
# 所以先用视觉模型把图片转成文字描述，再对描述做文本向量化。
desc_resp = client.chat.completions.create(
    model=AGICTO_VISION_MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                # 提示词决定了"会丢失什么信息"，所以要点明必须覆盖的维度：
                # 主体 / 场景 / 人物 / 颜色 / 图中文字 —— 这些都是后续检索可能命中的关键词
                {
                    "type": "text",
                    "text": "请详细描述这张图片，用于后续检索：包含画面主体、场景、人物、颜色和图中出现的文字。",
                },
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        }
    ],
)
description = desc_resp.choices[0].message.content
print(f"图片描述: {description}\n")

# 3. 对描述文本做向量化
# 注意：这里喂给 Embedding 的是"描述文本"而不是图片本身，
# 向量空间与 1-文本embedding.py 完全一致，图片和文字因此可以互相检索
resp = client.embeddings.create(
    model=AGICTO_EMBEDDING_MODEL,
    input=description,
)

embedding = resp.data[0].embedding
# 把 description 一并打印：出现检索异常时，第一件事就是看描述写得对不对
result = {
    "model": resp.model,
    "usage": resp.usage.model_dump(),
    "description": description,
    "embedding_dim": len(embedding),
    "embedding": embedding,
}
print(json.dumps(result, ensure_ascii=False, indent=4))
