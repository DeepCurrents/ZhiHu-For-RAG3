#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片策略六：自适应切片（段落优先 + 长度容差）

【核心思想】
"自然段落本身就是作者划好的语义边界"，所以优先按段落来切，
只在段落累积到接近上限时才收尾。关键在于那个"容差"：

    允许的最大长度 = target_size × (1 + tolerance)

也就是说，目标写 200 字，容差 0.3，实际允许放到 260 字。
这样做的目的是：不要因为"再加这一段就超了 3 个字"而硬生生多切一刀——
多容忍一点长度，换来更完整的语义，这是典型的工程折中。

【为什么需要容差】
如果严格按 target_size 卡死，最后一个段落很容易被推到下一块，
结果是一堆长度参差不齐的碎块，检索质量反而下降。

【与前几种策略的对比】
- 固定长度/滑动窗口：只看位置，不理解结构
- 句子边界切片：按标点切，块内是完整句子
- 自适应切片：按段落切，块内是完整段落——粒度更粗、语义更完整

【局限】
若某个段落本身就比 target_size × (1 + tolerance) 还长，本策略不会拆它，
会产出一个超长块（真实项目中需要再加一层"超长块递归切分"来兜底）。
"""


def adaptive_chunking(text, target_size=512, tolerance=0.2):
    """自适应切片：按段落合并，允许在目标长度上下浮动。

    参数:
        text:        待切分的原始文本
        target_size: 目标切片长度（字符数）
        tolerance:   长度容差，实际上限 = target_size × (1 + tolerance)
    返回:
        切片列表
    """
    chunks = []

    # 按空行切分段落（\n\n 是 Markdown / 纯文本里最常见的段落分隔符）
    paragraphs = text.split('\n\n')

    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # 如果"当前块 + 这个段落"超过了「目标长度 × (1 + 容差)」，就收尾重开一块
        # 注意判断的是 target_size * (1 + tolerance)，而不是死板的 target_size
        if len(current_chunk) + len(paragraph) > target_size * (1 + tolerance):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            # 还在容差范围内，继续往当前块里追加（非首段时用空格连接）
            current_chunk += " " + paragraph if current_chunk else paragraph

    # 处理最后一个块：循环结束后残留内容要结算，否则最后一段会丢
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
    print("🎯 自适应切片策略测试")
    print(f"📄 测试文本长度: {len(text)} 字符")

    # 使用自适应切片：目标 200 字、容差 0.3，即实际允许到 260 字
    # 把 tolerance 调成 0 再跑一次，可以直观看到"容差"对切片均匀度的影响
    chunks = adaptive_chunking(text, target_size=200, tolerance=0.3)
    print_chunk_analysis(chunks, "自适应切片")
