#!/usr/bin/env python
# coding: utf-8
"""
文本 Embedding：把一句话变成一串数字（RAG 的"语义坐标"）

【这一步在 RAG 流水线中的位置】
    切片后的文本 → 【Embedding 向量化】→ 存入向量库（FAISS 等）→ 检索时算相似度 → 召回最相关的块
切片解决"切多大"，Embedding 解决"怎么比"：
它把文本映射到高维空间，语义相近的文本距离就近，于是"找答案"变成了"找最近邻"。

【关键认知】
1. 向量本身没有可读含义，它表达的是"与其他文本的相对位置关系"
2. 维度（embedding_dim）由模型决定，比如 text-embedding-3-small 是 1536 维
3. 【最容易踩的坑】建库和查询必须用同一个 Embedding 模型！
   换模型 = 换了一套坐标系，旧向量全部作废，向量库必须重建
   （这正是本案例迁移到 agicto 后需要重跑 4-disney_build_index.py 的原因）

【接口说明】
POST /v1/embeddings
- input 只接受「字符串」或「字符串数组」，一次可批量传多条
- 返回 data 是个数组，data[0].embedding 才是第一条文本的向量

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
# 注意：改这里就必须重建向量索引，否则查询向量与库内向量不在同一语义空间
AGICTO_EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(
    api_key=os.environ["AGICTO_API_KEY"],
    base_url=AGICTO_BASE_URL,
)

# 待向量化的文本：内容是迪士尼门票信息，与知识库主题一致
text = "上海迪士尼乐园门票分为一日票、两日票和特定日票三种类型。一日票可在购买时选定日期使用，价格根据季节浮动，平日成人票475元起"

# 调用模型接口
# inference 类接口都是同步阻塞的，这一行会等网络返回
resp = client.embeddings.create(
    model=AGICTO_EMBEDDING_MODEL,
    input=text,
)

# resp.data 是列表（哪怕只传一条也是列表），取 [0] 才是这条文本的向量
embedding = resp.data[0].embedding
# 把关键信息打包打印，方便观察真实返回：用了哪个模型、消耗多少 token、向量多少维
# usage.model_dump() 把 SDK 的对象转成普通字典，便于 json 序列化
result = {
    "model": resp.model,
    "usage": resp.usage.model_dump(),
    "embedding_dim": len(embedding),
    "embedding": embedding,
}
# ensure_ascii=False 才能正常显示中文；indent=4 让长向量易于阅读
print(json.dumps(result, ensure_ascii=False, indent=4))
