# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - 在线查询（Query / Retrieve / Generate）

【这一步在 RAG 流水线中的位置】
    4-disney_build_index.py 负责「离线建库」：把知识切成小块 → 向量化 → 存进 FAISS + 元数据 json。
    本文件负责「在线查询」，也就是 RAG 的后半条链路，四步走：
        1) Query 向量化 —— 把用户问题送进同一个 Embedding 模型，得到查询向量
        2) Retrieve     —— 用查询向量在 FAISS 里找最相近的 N 条知识
        3) 组装 Context —— 把命中的原文拼进 prompt（这就是 RAG 的 "Augmented"）
        4) Generate     —— 让 LLM 只依据 Context 作答，减少胡编（幻觉）

【对比离线阶段，理解"为什么"】
    - 离线阶段做的是"重活"：解析 PDF/DOCX、调用视觉模型描述图片和视频、逐条 embedding。
      这些都是一次性的，建好之后就落盘了，线上不用重复做。
    - 在线阶段做的是"轻活"：整个流程只有 2 次外部调用 —— 1 次 embedding + 1 次 chat。
      所以 RAG 的响应速度主要取决于这两次 API 往返，而不是知识库多大。

【学习要点】
    1. 向量一致性铁律：本文件的 AGICTO_EMBEDDING_MODEL 必须和 4-disney_build_index.py 完全一致。
       换模型 = 换了坐标系，旧向量与新查询向量不在同一个空间里，检索结果会彻底失效，必须重建索引。
    2. L2 距离 → 相似度的换算：FAISS 返回的是距离（越小越近），不是相似度（越大越像）。
       这里用 1/(1+d) 把"距离"映射到 (0, 1] 区间，纯粹为了打印/阈值更直观。
    3. 多模态检索的"降维"本质：图片和视频早在离线阶段就被视觉模型转成了文字描述，
       再走同一个文本 embedding。所以在线阶段完全不需要视觉模型，一条 query 就能同时命中
       文本、图片、视频三种类型——它们本就躺在同一个向量空间里。
    4. 媒体意图路由：先用关键词判断用户是否"想看图片/视频"，再在对应类型的结果里挑距离最近的，
       并用距离阈值兜底，避免"用户随口一提视频"就硬塞一个不相关的视频进答案。

【运行前提】
    - 需要先跑通 4-disney_build_index.py，生成 disney_index.faiss 与 disney_metadata.json
    - 需设置环境变量 AGICTO_API_KEY
"""
import os
import json
import numpy as np
import faiss
from openai import OpenAI

# agicto 配置（OpenAI 兼容接口）
AGICTO_BASE_URL = "https://api.agicto.cn/v1"
# 可按 agicto 账户实际可用列表替换
# 注意：向量模型必须与 4-disney_build_index.py 保持一致，否则索引需重建
AGICTO_EMBEDDING_MODEL = "text-embedding-3-small"
AGICTO_CHAT_MODEL = "qwen-plus"

# 当前脚本所在目录，保证从任意工作目录运行都能定位到资源文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

AGICTO_API_KEY = os.getenv("AGICTO_API_KEY")
if not AGICTO_API_KEY:
    raise ValueError("错误：请设置 'AGICTO_API_KEY' 环境变量。")

client = OpenAI(
    api_key=AGICTO_API_KEY,
    base_url=AGICTO_BASE_URL,
)

# 与 4-disney_build_index.py 落盘时的文件名严格对应，两边必须一致
INDEX_FILE = os.path.join(BASE_DIR, "disney_index.faiss")
METADATA_FILE = os.path.join(BASE_DIR, "disney_metadata.json")

# 关键词配置
# 说明：这是最朴素的"意图识别"，够用是因为场景窄（迪士尼客服）。
# 它的作用是决定"要不要额外附上图片/视频"，而不是决定能否检索到——
# 三类数据本来就在同一个索引里，是否命中由向量相似度说了算。
IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放"]
MEDIA_DISTANCE_THRESHOLD = 3.0  # 图片/视频匹配的距离阈值


def load_index():
    """加载索引和元数据。

    索引（.faiss）只存向量，元数据（.json）存原文/来源/类型，
    两者靠"下标"对齐：index 里第 i 个向量对应 metadata[i]。
    所以这里必须成对加载，只加载索引是没法还原出任何内容的。

    返回:
        (index, metadata) 二元组：FAISS 索引对象 + 与向量下标一一对应的元数据列表
    """
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"已加载索引: {index.ntotal} 条记录")
    return index, metadata


def get_text_embedding(text):
    """文本embedding。

    注意这里没有任何"图片/视频"分支——多模态的差异已经在离线阶段消化掉了，
    在线只负责把 query 变成向量，与库里的向量做同尺度比较。

    参数:
        text: 用户问题
    返回:
        向量列表（长度 = 建库时同一模型的维度）
    """
    resp = client.embeddings.create(
        model=AGICTO_EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def distance_to_similarity(distance):
    """L2距离转相似度 (0-1之间，越大越相似)。

    FAISS 返回的 L2 距离范围是 [0, +∞)，越小越相似，直接看数字不直观。
    用 1/(1+d) 做单调递减映射：距离 0 → 相似度 1，距离越大 → 相似度趋近 0。
    它不改变排序（单调映射保序），只是让展示和阈值判断更好读。

    参数:
        distance: 非负的 L2 距离
    返回:
        (0, 1] 区间内的相似度
    """
    return 1 / (1 + distance)


def detect_media_intent(query):
    """检测query中是否包含图片/视频意图。

    参数:
        query: 用户原始问题
    返回:
        (want_image, want_video) 两个布尔值
    """
    query_lower = query.lower()
    # any(...) 等价于"命中任意一个关键词就算有意向"
    want_image = any(kw in query_lower for kw in IMAGE_KEYWORDS)
    want_video = any(kw in query_lower for kw in VIDEO_KEYWORDS)
    return want_image, want_video


def search_with_details(query, index, metadata):
    """检索并打印相似度详情。

    参数:
        query:    用户问题
        index:    FAISS 索引
        metadata: 与向量下标对齐的元数据列表
    返回:
        results 列表，每项含 idx / distance / similarity / metadata，已按距离升序
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    # 关键点：查询向量要包一层 [] 再转 float32。
    #   - 外层 [] 是因为 FAISS 的 search 要求入参是二维矩阵 (n_queries, dim)；
    #   - float32 是 FAISS 的硬性类型要求，与离线建库时保持一致。
    query_vec = np.array([get_text_embedding(query)]).astype('float32')

    # 检索所有
    # 这里 k 传 index.ntotal（即"全库召回"），是为了把完整排名打印出来便于教学观察。
    # 真实生产环境不会这么干：库一大就是全表扫描，应该只取 top-k（比如 5~20 条）。
    distances, indices = index.search(query_vec, index.ntotal)

    print(f"\n相似度排名 (越大越相似):")
    print("-" * 80)
    print(f"{'排名':4s} {'ID':4s} {'类型':6s} {'相似度':8s} {'距离':8s} 内容")
    print("-" * 80)

    results = []
    # search 返回值都是二维数组（形状 (n_queries, k)），所以取 [0] 拿本次查询那一行
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        # FAISS 在可用结果不足 k 个时，空位会填 -1，需要跳过
        if idx == -1:
            continue
        # 靠下标回查元数据：这就是"向量与 metadata 一一对应"的兑现处
        m = metadata[idx]
        sim = distance_to_similarity(dist)
        content_preview = m['content'][:45].replace('\n', ' ')
        type_tag = m['type']

        marker = ""
        if type_tag == "image":
            marker = " <-- 图片"
        elif type_tag == "video":
            marker = " <-- 视频"

        print(f"{rank+1:4d} {idx:4d} [{type_tag:5s}] {sim:6.4f}  {dist:8.4f}  {content_preview}...{marker}")
        results.append({"idx": idx, "distance": dist, "similarity": sim, "metadata": m})

    return results


def rag_ask(query, index, metadata, k=3):
    """RAG问答，支持图片/视频关键词检测。

    完整串起在线链路：检索 → 意图判断 → 挑上下文 → 拼 prompt → 生成 → 附加媒体。

    参数:
        query:    用户问题
        index:    FAISS 索引
        metadata: 元数据列表
        k:        送入 prompt 的文本知识条数（不是召回数，召回在本函数里是全库）
    返回:
        LLM 生成的答案字符串
    """
    results = search_with_details(query, index, metadata)

    # 检测媒体意图
    want_image, want_video = detect_media_intent(query)
    print(f"\n意图检测: 需要图片={want_image}, 需要视频={want_video}")

    # 取top-k文本结果
    # 只把 type == "text" 的送进 prompt：图片/视频的 content 是"视觉模型生成的描述"，
    # 描述本身是为检索而写的，细节和语气都不适合直接当客服话术，所以让它们只作为附件返回。
    top_results = [r for r in results if r["metadata"]["type"] == "text"][:k]

    # 如果需要图片，找距离<3的图片中距离最小的Top1
    # 阈值 MEDIA_DISTANCE_THRESHOLD 是防"误挂"的闸门：
    # 用户提到"海报"但库里没有相关图时，最近的图片可能距离很大（很不想关），
    # 此时宁可不给，也不要塞一张不相干的图片误导用户。
    matched_image = None
    if want_image:
        image_results = [r for r in results if r["metadata"]["type"] == "image" and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        if image_results:
            # 按距离排序，取距离最小的Top1
            image_results.sort(key=lambda x: x["distance"])
            matched_image = image_results[0]
            print(f"  -> 匹配到图片: {matched_image['metadata']['path']} (距离: {matched_image['distance']:.4f}, 相似度: {matched_image['similarity']:.4f})")

    # 如果需要视频，找距离<3的视频中距离最小的Top1
    matched_video = None
    if want_video:
        video_results = [r for r in results if r["metadata"]["type"] == "video" and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        if video_results:
            # 按距离排序，取距离最小的Top1
            video_results.sort(key=lambda x: x["distance"])
            matched_video = video_results[0]
            print(f"  -> 匹配到视频: {matched_video['metadata']['url']} (距离: {matched_video['distance']:.4f}, 相似度: {matched_video['similarity']:.4f})")

    print(f"\n选取Top-{k}文本构建Prompt:")
    for r in top_results:
        print(f"  - {r['metadata']['content'][:50]}... (相似度: {r['similarity']:.4f})")

    # 构建context
    # 每条知识都带上"来源 + 相似度"：来源便于人工溯源核查，相似度让模型对可信度有感知。
    # 注意这里是把检索到的原文"塞"进 prompt —— 这就是 RAG 中 Augmented（增强）的字面含义，
    # 模型本身没被训练过迪士尼知识，全靠这段上下文现学现答。
    context_str = ""
    for i, r in enumerate(top_results):
        m = r["metadata"]
        context_str += f"背景知识 {i+1} (来源: {m['source']}, 相似度: {r['similarity']:.4f}):\n{m['content']}\n\n"

    prompt = f"""你是一个迪士尼客服助手。请根据以下背景知识回答用户问题。

[背景知识]
{context_str}
[用户问题]
{query}
"""

    # 调用LLM
    # system 定角色，user 交任务，是对话模型最标准的用法。
    # 生成阶段是"幻觉"唯一可能引入的地方：上下文没覆盖到的内容，模型可能自行补全。
    # 生产环境通常会在 system 里加一句"若背景知识中没有答案，请回答不知道"来压制幻觉。
    print("\n调用LLM生成答案...")
    completion = client.chat.completions.create(
        model=AGICTO_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个迪士尼客服助手。"},
            {"role": "user", "content": prompt}
        ]
    )
    answer = completion.choices[0].message.content

    # 附加匹配到的媒体
    # 媒体不进 prompt、只在答案末尾追加路径/URL，这样做的好处：
    # 一是避免把图片描述当正文污染回答风格；二是前端拿到路径就能直接渲染出图/播视频。
    if matched_image:
        answer += f"\n\n[相关图片]: {matched_image['metadata']['path']}"
    if matched_video:
        answer += f"\n\n[相关视频]: {matched_video['metadata']['url']}"

    print(f"\n最终答案:\n{answer}")
    return answer


if __name__ == "__main__":
    index, metadata = load_index()

    # 四个测试用例分别覆盖四种典型路径，便于观察在线链路的差异：
    #   测试1 → 纯文本：没有媒体意图，只走检索 + 生成
    #   测试2 → 图片意图 + 实际命中图片（会看到 "-> 匹配到图片"）
    #   测试3 → 视频意图 + 实际命中视频
    #   测试4 → 图片意图（"海报"既是文本语料也是图片语料，可观察两类结果的相似度差异）
    print("\n" + "="*60)
    print("测试1: 文本查询")
    rag_ask("我想了解一下迪士尼门票的退款流程", index, metadata, k=3)

    print("\n" + "="*60)
    print("测试2: 图片查询 - 万圣节海报")
    rag_ask("最近万圣节的活动海报是什么", index, metadata, k=3)

    print("\n" + "="*60)
    print("测试3: 视频查询 - 汽车剐蹭")
    rag_ask("我的汽车被剐蹭了，你能看到视频么？", index, metadata, k=3)

    print("\n" + "="*60)
    print("测试4: 图片查询 - 聚在一起")
    rag_ask("聚在一起说奇妙的海报", index, metadata, k=3)
