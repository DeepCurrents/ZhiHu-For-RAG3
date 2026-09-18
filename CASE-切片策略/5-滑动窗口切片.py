#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片策略五：滑动窗口切片（最简单、最粗暴，也最快）

【原理】
把文本想象成一条带子，用一个固定宽度的"窗口"从头往右滑，每滑一次就截一段：
    窗口宽度 = window_size
    每次滑动距离 = step_size
当 step_size < window_size 时，相邻切片就会天然重叠：
    重叠长度 = window_size - step_size

【和前几种策略的关系】
- 固定长度切片：切点会回退到句子边界 → 切片更"可读"
- 滑动窗口切片：不做任何对齐，纯按位置切 → 实现最简单、速度最快
代价就是切点可能落在句子中间，单个切片读起来不完整。

【优点 / 缺点】
+ 实现极简，不依赖标点、不需要调模型，任何语言都适用（中文、代码、日志都可以）
+ 通过重叠保证"跨切点的重要信息"至少在一个切片里完整出现一次
- 切片之间高度重复，向量库里会存很多近似内容，检索时容易召回一堆相似块
- 完全不理解语义

【学习要点】
overlap 越大越不容易漏信息，但冗余也越大。经验上 window_size 的 10%~25% 比较常见，
本文件测试用的 150/300 = 50% 是刻意放大的，方便观察重叠效果。
"""


def sliding_window_chunking(text, window_size=512, step_size=256):
    """滑动窗口切片。

    参数:
        text:        待切分的原始文本
        window_size: 窗口宽度，即每个切片的字符数上限
        step_size:   每次滑动的步长；step_size < window_size 时相邻切片重叠
    返回:
        切片列表
    """
    chunks = []

    # range(起点, 终点, 步长)：起点按 step_size 递增，每次截取 [i, i + window_size)
    for i in range(0, len(text), step_size):
        # Python 切片超出长度不会报错，末尾不足一个窗口时会自动截短
        chunk = text[i:i + window_size]

        # 过滤纯空白切片
        if len(chunk.strip()) > 0:
            chunks.append(chunk.strip())

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
    print("🎯 滑动窗口切片策略测试")
    print(f"📄 测试文本长度: {len(text)} 字符")

    # 使用滑动窗口切片：window_size=300 且 step_size=150，
    # 也就是每块 300 字符、每次前进 150 字符，相邻块必然重叠 150 字符
    chunks = sliding_window_chunking(text, window_size=300, step_size=150)
    print_chunk_analysis(chunks, "滑动窗口切片")
