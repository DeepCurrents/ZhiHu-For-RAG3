#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片策略二：句子边界切片（先分句，再按长度合并）

【思路】
思路正好和"固定长度切片"反过来：
    固定长度切片 = 先按字数切，再回头对齐句子边界
    句子边界切片 = 先用标点把文本拆成句子，再把句子逐个"攒"成不超过上限的块

这样做的好处是每个切片天然由若干完整句子组成，绝不会出现半句话。

【和策略一的区别】
- 策略一是"边切边对齐"，本策略是"先分句再合并"，实现更简单直观
- 本策略没有 overlap（相邻块不重叠），信息丢失的风险略高
- 若单句长度就超过 max_chunk_size，本策略无法拆分它，会产出一个"超长块"（见下方注释）

【学习要点】
正则 r'[.!?。！？\\n]+' 中的 \\n 很关键：中文文本里段落之间常常只有换行而没有句号，
把换行也当作切分符，才能避免整段被当成一个超长句子。
"""

import re


def semantic_chunking(text, max_chunk_size=512):
    """基于句子边界的切片：先按标点分句，再贪心地合并句子直到接近长度上限。

    参数:
        text:           待切分的原始文本
        max_chunk_size: 每个切片的长度上限（字符数）
    返回:
        切片列表
    """
    # 使用正则表达式分割句子：中英文句号、问号、感叹号、换行都视为句子边界
    # re.split 会把分隔符本身丢掉，所以切片里不会再保留句末标点
    sentences = re.split(r'[.!?。！？\n]+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 核心逻辑：如果"当前块 + 这句话"会超出上限，就把当前块结算掉，从这句话重新开块
        # 注意这里没有对单句长度做限制：若某句话本身就超过上限，它仍会整句进入切片
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # 循环结束后，最后攒在 current_chunk 里的内容还没结算，别漏掉
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

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
    print("🎯 句子边界切片策略测试")
    print(f"📄 测试文本长度: {len(text)} 字符")

    # max_chunk_size=300 同样是为了在短文本上切出多块，方便对比
    chunks = semantic_chunking(text, max_chunk_size=300)
    print_chunk_analysis(chunks, "句子边界切片")
