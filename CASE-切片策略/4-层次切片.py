#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片策略四：层次切片（按文档结构切）

【适用场景】
结构化文档：Markdown、Word 说明书、API 文档、法律法规等。
这类文档本身就用"标题层级"表达了语义边界，直接顺着结构切，比按字数切合理得多。

【核心思想】
把"标题级别"当作强切分信号：
- 遇到一级/二级标题 → 无条件开新块（保证一个章节不会被拆散）
- 块长超过 target_size → 强制开新块（保证不出现超长块）
- 走到段落且长度已达目标的 80% → 提前收尾，避免再塞进一个长段落导致超标

也就是说：结构优先，长度兜底。

【学习要点】
这是"从文档源码里提取结构"的典型做法。工程上更常见的形态是先把 PDF/Word 转成
Markdown，再按 `#` 层级递归切分，思路与本文件完全一致。

【局限】
- 只认本文件 hierarchy_markers 里列举的几种标记，换个模板就可能失效
- 超长章节会被"长度兜底"规则强行切开，此时切点不一定落在语义边界上
"""


def hierarchical_chunking(text, target_size=512, preserve_hierarchy=True):
    """层次切片：优先按标题层级切分，长度超限时再强制切分。

    参数:
        text:               待切分文本（需自带 #、一、二 之类的层次标记）
        target_size:        期望的切片长度
        preserve_hierarchy: 是否尊重标题层级（True 时遇到一/二级标题必开新块）
    返回:
        切片列表
    """
    chunks = []

    # 定义层次标记：把不同文档风格（Markdown / 中文序号 / 数字序号）统一映射成级别
    hierarchy_markers = {
        'title1': ['# ', '标题1：', '一、', '1. '],
        'title2': ['## ', '标题2：', '二、', '2. '],
        'title3': ['### ', '标题3：', '三、', '3. '],
        'paragraph': ['\n\n', '\n']
    }

    # 分割文本为行：按行处理才能逐行判断"这一行是不是标题"
    lines = text.split('\n')
    current_chunk = ""
    # 记录当前切片所属的层次路径（真实项目中可写进 metadata，便于按章节过滤检索）
    current_hierarchy = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 检测当前行的层次级别：从高级别往低级别依次试，命中即止
        line_level = None
        for level, markers in hierarchy_markers.items():
            for marker in markers:
                if line.startswith(marker):
                    line_level = level
                    break
            if line_level:
                break

        # 如果没有检测到层次标记，默认为段落
        if not line_level:
            line_level = 'paragraph'

        # 判断是否需要开始新的切片
        should_start_new_chunk = False

        # 规则1：遇到一级/二级标题就开新块——这是"层次切片"的灵魂，保证章节不被拆分
        if preserve_hierarchy and line_level in ['title1', 'title2']:
            should_start_new_chunk = True

        # 规则2：当前块加上这一行会超过目标长度，说明必须收尾了
        if len(current_chunk) + len(line) > target_size and current_chunk.strip():
            should_start_new_chunk = True

        # 规则3：已经写到目标的 80%，且接下来是普通段落，就提前收尾（避免超标）
        if line_level == 'paragraph' and len(current_chunk) > target_size * 0.8:
            should_start_new_chunk = True

        # 开始新切片：先把上一块存起来，再清空状态
        if should_start_new_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_hierarchy = []

        # 添加当前行到切片（第一行直接赋值，后续行用 \n 拼接以保留段落结构）
        if current_chunk:
            current_chunk += "\n" + line
        else:
            current_chunk = line

        # 更新层次信息
        if line_level != 'paragraph':
            current_hierarchy.append(line_level)

    # 处理最后一个切片：循环结束时残留的内容同样要结算
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


# 测试文本 - 包含层次结构（刻意写成带 # 标题的 Markdown，便于观察"结构优先"的效果）
text = """
# 迪士尼乐园门票指南

## 一、门票类型介绍

### 1. 基础门票类型
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

### 2. 特殊门票类型
年票适合经常游玩的游客，提供更多优惠和特权。VIP门票包含快速通道服务，可减少排队时间。团体票适用于10人以上团队，享受团体折扣。

## 二、购票渠道与流程

### 1. 官方购票渠道
购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。这些渠道提供最可靠的服务和最新的票务信息。

### 2. 第三方平台
第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。建议优先选择官方渠道以确保购票安全。

### 3. 证件要求
所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

## 三、入园须知

### 1. 入园时间
乐园通常在上午8:00开园，晚上8:00闭园，具体时间可能因季节和特殊活动调整。建议提前30分钟到达园区。

### 2. 安全检查
入园前需要进行安全检查，禁止携带危险物品、玻璃制品等。建议轻装简行，提高入园效率。

### 3. 园区服务
园区内提供寄存服务、轮椅租赁、婴儿车租赁等服务，可在游客服务中心咨询详情。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

if __name__ == "__main__":
    print("🎯 层次切片策略测试")
    print(f"📄 测试文本长度: {len(text)} 字符")

    # preserve_hierarchy=True：开启"标题必开新块"，可以对比改成 False 后的差异
    chunks = hierarchical_chunking(text, target_size=300, preserve_hierarchy=True)
    print_chunk_analysis(chunks, "层次切片")
