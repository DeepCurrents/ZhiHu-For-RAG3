#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片策略对比测试脚本 —— 一次跑完 6 种切片策略，横向对比效果

【为什么要做对比】
切片是 RAG 链路的第一步，也是"改一行参数就能显著影响召回质量"的一步。
单看某一种策略很难判断好坏，把同一段文本喂给 6 种策略、用同一套指标量化，
才能直观看出差异。本文件相当于把前 6 个脚本的函数集中到一起做 A/B 测试。

【6 种策略一览】
1. 固定长度切片        —— 按字数切，切点回退到句子边界
2. 句子边界切片        —— 先分句再按长度合并
3. LLM 语义切片        —— 交给大模型按语义切（成本最高、效果通常最好）
4. 滑动窗口切片        —— 固定窗口 + 固定步长，天然重叠
5. 自适应切片          —— 按段落合并，带长度容差
6. 智能自适应切片      —— 在 5 的基础上，再加"下一句太长就先收尾"等启发式规则

【对比指标怎么读】
- 切片数：同样文本切出的块数越多，平均每块信息越少
- 平均长度：太短 → 上下文不足；太长 → 噪声多、稀释相关性
- 长度方差：越小越均匀，检索质量越稳定
- 推荐度：脚本里用几条经验规则打的星级，仅作示意，真实项目应改用检索命中率来评估

【运行前提】
策略 3（LLM 切片）需要设置环境变量 AGICTO_API_KEY；
未设置时会自动降级为"句子边界切片"，其余 5 种策略纯本地计算、无需联网。
"""

import json
import os
import re

from openai import OpenAI

# agicto 配置（OpenAI 兼容接口）
AGICTO_BASE_URL = "https://api.agicto.cn/v1"
# 可按 agicto 账户实际可用列表替换
AGICTO_CHAT_MODEL = "qwen-plus"


# 1. 固定长度切片
def improved_fixed_length_chunking(text, chunk_size=512, overlap=50):
    """固定长度切片 - 在句子边界切分

    先按 chunk_size 圈窗口，再往前回溯最多 100 字符找句末标点，把切点对齐到句子结束处，
    最后用 overlap 让相邻块重叠，避免关键信息正好落在切点上被丢失。
    """
    chunks = []
    start = 0

    # while 而非 for：end 会被回退到句子边界，步长不固定
    while start < len(text):
        end = start + chunk_size

        # 尝试在句子边界切分
        if end < len(text):
            # 寻找最近的句子结束符（最多回溯 100 字符，找不到就硬切）
            for i in range(end, max(start, end - 100), -1):
                if text[i] in '.!?。！？':
                    end = i + 1
                    break

        chunk = text[start:end]

        # 跳过空白块，避免产生无意义的向量
        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())

        # 下一块从「本节结束位置 - overlap」开始，实现重叠
        start = end - overlap

    return chunks


# 2. 句子边界切片
def semantic_chunking(text, max_chunk_size=512):
    """基于句子边界的切片 - 按句子分割

    先用标点把文本拆成句子，再贪心地把句子攒成不超过 max_chunk_size 的块。
    块内一定是完整句子，但单句超长时无法再拆。
    """
    # 使用正则表达式分割句子（\n 也当分隔符，避免整段变成一个超长"句子"）
    sentences = re.split(r'[.!?。！？\n]+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 如果当前句子加入后超过最大长度，保存当前块
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # 添加最后一个块（循环结束后残留内容别漏）
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# 3. LLM语义切片（LLM）
def advanced_semantic_chunking_with_llm(text, max_chunk_size=512):
    """使用LLM进行语义切片

    唯一"理解内容"的策略：把文本交给大模型，让它按语义完整性划分。
    代价是调用成本与网络延迟；因为模型返回的是自然语言，解析必须做容错。
    """
    # 检查环境变量：没有 Key 就降级到纯本地的句子边界切片
    api_key = os.getenv("AGICTO_API_KEY")
    if not api_key:
        print("警告: 未设置 AGICTO_API_KEY 环境变量，将使用基础语义切片")
        return semantic_chunking(text, max_chunk_size)

    # agicto 兼容 OpenAI 协议，指定 base_url 即可
    client = OpenAI(
        api_key=api_key,
        base_url=AGICTO_BASE_URL
    )

    # Prompt 三要素：任务 + 约束 + 输出格式；强制 JSON 是为了能被程序稳定解析
    prompt = f"""
请将以下文本按照语义完整性进行切片，每个切片不超过{max_chunk_size}字符。
要求：
1. 保持语义完整性
2. 在自然的分割点切分
3. 返回JSON格式的切片列表，格式如下：
{{
  "chunks": [
    "第一个切片内容",
    "第二个切片内容",
    ...
  ]
}}

文本内容：
{text}

请返回JSON格式的切片列表：
"""

    try:
        response = client.chat.completions.create(
            model=AGICTO_CHAT_MODEL,
            messages=[
                # system 定规矩、user 给任务，双重强调"只返回 JSON"
                {"role": "system", "content": "你是一个专业的文本切片助手。请严格按照JSON格式返回结果，不要添加任何额外的标记。"},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content

        # 清理结果，移除可能的Markdown代码块标记
        # 模型经常把 JSON 包成 ```json ... ```，直接 json.loads 会报错，所以先剥壳
        cleaned_result = result.strip()
        if cleaned_result.startswith('```'):
            cleaned_result = re.sub(r'^```(?:json)?\s*', '', cleaned_result)
        if cleaned_result.endswith('```'):
            cleaned_result = re.sub(r'\s*```$', '', cleaned_result)

        # 解析JSON结果
        chunks_data = json.loads(cleaned_result)

        # 处理不同的返回格式（不同模型/提示词下结构可能不一样，做兼容）
        if "chunks" in chunks_data:
            return chunks_data["chunks"]
        elif "slice" in chunks_data:
            if isinstance(chunks_data, list):
                return [item.get("slice", "") for item in chunks_data if item.get("slice")]
            else:
                return [chunks_data["slice"]]
        else:
            if isinstance(chunks_data, list):
                return chunks_data
            else:
                return []

    except Exception as e:
        # 兜底：网络异常、鉴权失败、JSON 解析失败等都走降级，保证对比测试能跑完
        print(f"LLM切片失败: {e}")
        return semantic_chunking(text, max_chunk_size)


# 4. 滑动窗口切片
def sliding_window_chunking(text, window_size=512, step_size=256):
    """滑动窗口切片

    固定窗口宽度、固定前进步长；step_size < window_size 时相邻块天然重叠。
    实现最简单、速度最快，但切点可能落在句子中间，且块间冗余度高。
    """
    chunks = []

    # 起点按 step_size 递增，每次截取 [i, i + window_size)
    for i in range(0, len(text), step_size):
        chunk = text[i:i + window_size]

        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())

    return chunks


# 5. 自适应切片
def adaptive_chunking(text, target_size=512, tolerance=0.2):
    """自适应切片 - 根据内容自适应调整

    按自然段落合并；实际允许的最大长度是 target_size × (1 + tolerance)，
    多容忍一点长度，换取更完整的语义、更均匀的切片。
    """
    chunks = []

    # 按段落分割
    paragraphs = text.split('\n\n')

    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # 如果当前段落加入后超过目标大小（含容差）
        if len(current_chunk) + len(paragraph) > target_size * (1 + tolerance):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += " " + paragraph if current_chunk else paragraph

    # 处理最后一个块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# 6. 智能自适应切片
def smart_adaptive_chunking(text, target_size=512, min_size=100, max_size=1000):
    """智能自适应切片 - 考虑语义和长度

    比策略 5 多了两道"闸门"：
    - min_size：块还没攒够最小长度时，即使超过 target_size 也先不切，避免碎块
    - max_size：无论下一句多短，超过硬上限都必须切
    - 若下一句本身很长（> target_size 的 30%），提前收尾，避免"长句把块撑爆"
    """
    chunks = []

    # 按句子分割（比按段落分粒度更细，便于精细控制）
    sentences = re.split(r'[.!?。！？\n]+', text)

    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 检查是否需要开始新块
        # 闸门1：超过硬上限，且已达到最小长度，必须切
        if (len(current_chunk) + len(sentence) > max_size and
            len(current_chunk) >= min_size):
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        elif len(current_chunk) + len(sentence) > target_size and len(current_chunk) >= min_size:
            # 接近目标长度，考虑是否结束当前块
            if len(sentence) > target_size * 0.3:  # 如果下一句很长，结束当前块
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        else:
            # 闸门2：还没攒够 min_size，继续累加
            current_chunk += " " + sentence if current_chunk else sentence

    # 处理最后一个块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def print_chunk_analysis(chunks, method_name):
    """打印单个策略的切片统计与内容

    评价切片质量的两个常用指标：
    - 平均长度：过短则上下文不足，过长则引入噪声
    - 长度方差（最长 - 最短）：方差越小说明切片越均匀，检索质量越稳定
    """
    print(f"\n{'='*60}")
    print(f"📋 {method_name}")
    print(f"{'='*60}")

    if not chunks:
        print("❌ 未生成任何切片")
        return

    total_length = sum(len(chunk) for chunk in chunks)
    avg_length = total_length / len(chunks)
    min_length = min(len(chunk) for chunk in chunks)
    max_length = max(len(chunk) for chunk in chunks)

    print(f"📊 统计信息:")
    print(f"   - 切片数量: {len(chunks)}")
    print(f"   - 平均长度: {avg_length:.1f} 字符")
    print(f"   - 最短长度: {min_length} 字符")
    print(f"   - 最长长度: {max_length} 字符")
    print(f"   - 长度方差: {max_length - min_length} 字符")

    print(f"\n📝 切片内容:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   块 {i} ({len(chunk)} 字符):")
        print(f"   {chunk}")
        print()


def main():
    """主测试函数：同一段文本跑 6 种策略，最后输出汇总对比表"""
    # 测试文本（所有策略共用同一份文本，保证对比公平）
    text = """
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

    print("🎯 切片策略对比测试")
    print(f"📄 测试文本长度: {len(text)} 字符")

    # 测试参数：所有策略都尽量用同一个 300 字作为目标长度，保证可比性
    target_size = 300

    # 1. 固定长度切片
    chunks1 = improved_fixed_length_chunking(text, chunk_size=target_size, overlap=50)
    print_chunk_analysis(chunks1, "1. 固定长度切片")

    # 2. 句子边界切片
    chunks2 = semantic_chunking(text, max_chunk_size=target_size)
    print_chunk_analysis(chunks2, "2. 句子边界切片")

    # 3. LLM语义切片（LLM）—— 唯一需要联网的步骤
    print("\n🤖 正在调用LLM进行语义切片...")
    chunks3 = advanced_semantic_chunking_with_llm(text, max_chunk_size=target_size)
    print_chunk_analysis(chunks3, "3. LLM语义切片（LLM）")

    # 4. 滑动窗口切片：步长取窗口的一半，即固定重叠 50%
    chunks4 = sliding_window_chunking(text, window_size=target_size, step_size=target_size//2)
    print_chunk_analysis(chunks4, "4. 滑动窗口切片")

    # 5. 自适应切片
    chunks5 = adaptive_chunking(text, target_size=target_size, tolerance=0.3)
    print_chunk_analysis(chunks5, "5. 自适应切片")

    # 6. 智能自适应切片
    chunks6 = smart_adaptive_chunking(text, target_size=target_size, min_size=100, max_size=500)
    print_chunk_analysis(chunks6, "6. 智能自适应切片")

    # 总结对比
    print(f"\n{'='*80}")
    print("📈 策略对比总结")
    print(f"{'='*80}")

    methods = [
        ("固定长度", chunks1),
        ("句子边界切片", chunks2),
        ("LLM语义切片", chunks3),
        ("滑动窗口", chunks4),
        ("自适应切片", chunks5),
        ("智能自适应", chunks6)
    ]

    print(f"{'策略':<12} {'切片数':<6} {'平均长度':<8} {'长度方差':<8} {'推荐度':<8}")
    print("-" * 50)

    for name, chunks in methods:
        if chunks:
            avg_len = sum(len(c) for c in chunks) / len(chunks)
            min_len = min(len(c) for c in chunks)
            max_len = max(len(c) for c in chunks)
            variance = max_len - min_len

            # 简单的推荐度评估（仅按长度分布打星，不代表真实检索效果）
            # 真实评估应该用「召回率 / 命中率」这类端到端指标，这里只是便于直观比较
            if len(chunks) >= 2 and variance < 100 and avg_len > 150:
                recommendation = "⭐⭐⭐⭐⭐"
            elif len(chunks) >= 2 and variance < 150:
                recommendation = "⭐⭐⭐⭐"
            elif len(chunks) >= 1:
                recommendation = "⭐⭐⭐"
            else:
                recommendation = "⭐⭐"

            print(f"{name:<12} {len(chunks):<6} {avg_len:<8.1f} {variance:<8.1f} {recommendation:<8}")
        else:
            # chunks 为 None（LLM 调用完全失败）或空列表时兜底
            print(f"{name:<12} {'0':<6} {'N/A':<8} {'N/A':<8} {'⭐':<8}")


if __name__ == "__main__":
    main()
