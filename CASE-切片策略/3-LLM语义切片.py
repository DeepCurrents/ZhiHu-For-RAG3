#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
切片策略三：LLM 语义切片

【为什么需要它】
前两种策略都只看"字符数和标点"，不理解内容在讲什么。
例如下面这段文本里既有"门票类型"又有"购票渠道"，机器切片可能把两个主题混在同一块里。
LLM 语义切片把整段文本交给大模型，让它按"语义完整性"来划分——
效果通常最好，但也是成本最高的一种。

【代价与工程要点】
- 有 token 成本和网络延迟，不适合超长文档一次性处理（实践中会先粗切再让模型细切）
- 模型返回的是自然语言，可能把 JSON 包在 ```json ``` 里，也可能多说几句客套话，
  所以必须"约束输出格式 + 容错解析"——本文件后半部分全在做这件事

【接口说明】
统一走 agicto 的 OpenAI 兼容接口：POST /v1/chat/completions
模型名等配置集中在文件顶部常量里，业务代码不硬编码，便于整体替换供应商。
"""
from openai import OpenAI
import json
import os
import re

# agicto 配置（OpenAI 兼容接口）
AGICTO_BASE_URL = "https://api.agicto.cn/v1"
# 可按 agicto 账户实际可用列表替换
AGICTO_CHAT_MODEL = "qwen-plus"


def fallback_semantic_chunking(text, max_chunk_size=512):
    """降级方案：不依赖大模型的句子边界切片。

    当没有配置 API Key、或大模型调用/解析失败时启用，
    保证程序在无网络、无额度的情况下依然能跑通，方便学习和调试。
    """
    sentences = re.split(r'[.!?。！？\n]+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # 累加后会超上限，就先结算当前块，再从这句话重新开块
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def advanced_semantic_chunking_with_llm(text, max_chunk_size=512):
    """调用大模型做语义切片。

    参数:
        text:           待切分文本
        max_chunk_size: 每个切片的长度上限（写进 prompt 里约束模型）
    返回:
        切片列表；任何异常情况下会返回降级结果或 None
    """
    # 先检查环境变量：没配 Key 就不硬闯，直接走降级方案
    api_key = os.getenv("AGICTO_API_KEY")
    if not api_key:
        print("警告: 未设置 AGICTO_API_KEY 环境变量，将使用基础语义切片")
        return fallback_semantic_chunking(text, max_chunk_size)

    # OpenAI SDK 只需改 base_url 就能指向 agicto，后续调用写法与 OpenAI 完全一致
    client = OpenAI(
        api_key=api_key,
        base_url=AGICTO_BASE_URL
    )

    # Prompt 三要素：任务 + 约束 + 输出格式
    # 之所以强制 JSON，是为了让程序能稳定解析；否则就得用正则去猜自然语言，极易出错
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
        print("正在调用LLM进行语义切片...")
        response = client.chat.completions.create(
            model=AGICTO_CHAT_MODEL,
            messages=[
                # system 消息负责"定规矩"，user 消息负责"给活儿"
                # 把 JSON 约束同时放在两边，能显著降低模型输出多余文字的概率
                {"role": "system", "content": "你是一个专业的文本切片助手。请严格按照JSON格式返回结果，不要添加任何额外的标记。"},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content
        print(f"LLM返回结果: {result[:200]}...")

        # 清理结果，移除可能的 Markdown 代码块标记
        # 模型很爱把 JSON 包成 ```json ... ```，直接 json.loads 会报错，所以先剥壳
        cleaned_result = result.strip()
        if cleaned_result.startswith('```'):
            # 移除开头的 ```json 或 ```
            cleaned_result = re.sub(r'^```(?:json)?\s*', '', cleaned_result)
        if cleaned_result.endswith('```'):
            # 移除结尾的 ```
            cleaned_result = re.sub(r'\s*```$', '', cleaned_result)

        # 解析JSON结果
        chunks_data = json.loads(cleaned_result)

        # 容错：不同模型/提示词下返回结构可能不同，这里兼容几种常见格式
        if "chunks" in chunks_data:
            return chunks_data["chunks"]
        elif "slice" in chunks_data:
            # 如果返回的是包含"slice"字段的列表
            if isinstance(chunks_data, list):
                return [item.get("slice", "") for item in chunks_data if item.get("slice")]
            else:
                return [chunks_data["slice"]]
        else:
            # 如果直接返回字符串列表
            if isinstance(chunks_data, list):
                return chunks_data
            else:
                print(f"意外的返回格式: {chunks_data}")
                return []

    except json.JSONDecodeError as e:
        # 走到这里说明模型输出的不是合法 JSON（最常见的原因是多说了话）
        print(f"JSON解析失败: {e}")
        print(f"原始结果: {result}")
        # 尝试手动解析：用非贪婪正则从一堆文字里"抠"出 JSON 对象
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                chunks_data = json.loads(json_str)
                if "chunks" in chunks_data:
                    return chunks_data["chunks"]
        except:
            pass
    except Exception as e:
        # 兜底：网络异常、鉴权失败、模型名不存在等都会落到这里
        print(f"LLM切片失败: {e}")


def test_chunking_methods():
    """测试函数：跑一遍 LLM 语义切片并打印每块内容与长度"""
    # 示例文本
    text = """
迪士尼乐园提供多种门票类型以满足不同游客需求。一日票是最基础的门票类型，可在购买时选定日期使用，价格根据季节浮动。两日票需要连续两天使用，总价比购买两天单日票优惠约9折。特定日票包含部分节庆活动时段，需注意门票标注的有效期限。

购票渠道以官方渠道为主，包括上海迪士尼官网、官方App、微信公众号及小程序。第三方平台如飞猪、携程等合作代理商也可购票，但需认准官方授权标识。所有电子票需绑定身份证件，港澳台居民可用通行证，外籍游客用护照，儿童票需提供出生证明或户口本复印件。

生日福利需在官方渠道登记，可获赠生日徽章和甜品券。半年内有效结婚证持有者可购买特别套票，含皇家宴会厅双人餐。军人优惠现役及退役军人凭证件享8折，需至少提前3天登记审批。
"""

    print("\n=== LLM语义切片测试 ===")
    try:
        # max_chunk_size=300：字数上限交给模型自己执行，我们只负责校验结果
        chunks = advanced_semantic_chunking_with_llm(text, max_chunk_size=300)
        print(f"LLM语义切片生成 {len(chunks)} 个切片:")
        for i, chunk in enumerate(chunks):
            print(f"LLM语义块 {i+1} (长度: {len(chunk)}): {chunk}")
    except Exception as e:
        print(f"LLM切片测试失败: {e}")


if __name__ == "__main__":
    test_chunking_methods()
