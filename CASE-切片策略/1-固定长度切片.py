#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片策略一：固定长度切片（并在句子边界处"回退"对齐）

【这一步在 RAG 流水线中的位置】
    文档 → 【切片 Chunking】→ 向量化 Embedding → 存入向量库 → 检索召回 → 交给 LLM 生成答案
切片是整条链路的第一步。切得好不好，直接决定后面"召回的内容是否完整、是否夹带噪声"。

【要解决的问题】
最朴素的固定长度切片是"每 N 个字符切一刀"，很容易把一句话从中间劈开，例如：
    原文：一日票可在购买时选定日期使用
    误切：一日票可在购买时   |   选定日期使用
两半都读不通，向量化后的语义也会被稀释，检索时可能两边都召回不到。

【本策略的做法】
先按 chunk_size 圈出一个窗口，再从这个窗口末尾往前找最近的句号/问号/感叹号，
把切点挪到句子结束处，保证每个切片都以完整句子收尾；
同时用 overlap 让相邻切片共享一小段内容，避免关键信息正好落在切点上被丢掉。

【学习要点】
- chunk_size：切片长度。越大上下文越完整，但噪声越多、检索越"粗"；越小越精准，但容易断章取义
- overlap：重叠长度。用来缓解"切点切断关键信息"的问题，代价是存储量与向量条数成倍增加
"""


def improved_fixed_length_chunking(text, chunk_size=512, overlap=50):
    """按固定长度切片，但切点会对齐到最近的句子结束符（句子边界优先）。

    参数:
        text:       待切分的原始文本
        chunk_size: 目标切片长度（字符数）
        overlap:    相邻切片的重叠字符数
    返回:
        切片列表（每块已去掉首尾空白）
    """
    chunks = []
    start = 0

    # 用 while 而不是 for：因为 end 会被"向前回退"到句子边界，实际步长并不固定
    while start < len(text):
        end = start + chunk_size

        # 只有窗口还没到文本末尾时，才需要做句子边界对齐
        if end < len(text):
            # 从 end 往前最多回溯 100 个字符，遇到句子结束符就把切点挪到它后面
            # 若 100 个字符内都没有结束符，说明这句特别长，放弃对齐、直接按 end 硬切
            for i in range(end, max(start, end - 100), -1):
                if text[i] in '.!?。！？':
                    end = i + 1
                    break

        chunk = text[start:end]

        # 过滤掉纯空白切片，避免把空块也写进向量库（空块会产生无意义的向量）
        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())

        # 下一次从「本次结束位置 - overlap」开始，从而实现相邻切片的内容重叠
        start = end - overlap

    return chunks


def print_chunk_analysis(chunks, method_name):
    """打印切片结果的统计信息，便于横向比较不同切片策略的优劣。

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


# 测试文本：一段关于迪士尼门票的非结构化说明，用来观察切片效果
text = """
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

if __name__ == "__main__":
    print("🎯 固定长度切片策略测试")
    print(f"📄 测试文本长度: {len(text)} 字符")

    # chunk_size=300 是为了在短文本上也能切出多块，方便观察"句子边界对齐"的效果
    chunks = improved_fixed_length_chunking(text, chunk_size=300, overlap=50)
    print_chunk_analysis(chunks, "固定长度切片")
