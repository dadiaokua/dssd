"""
dataset_loader.py - 多数据集加载器

从 3 个数据集中随机选取 prompt:
  1. LongForm          — 长文本生成 (input → output)
  2. python_code_instructions_18k_alpaca — Python 代码生成 (instruction → output)
  3. WizardLM_evol_instruct_V2_196k     — 通用指令 (conversations[0].value → ...)

每次调用 sample_prompts() 时, 随机从三个数据集中抽取指定数量的 prompt。
"""

import os
import json
import random
from typing import List, Optional

# 数据集根目录 (项目根目录下的 dataset/)
_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")

# 数据集配置: (目录名, 文件路径, 格式, prompt 提取函数)
_DATASETS = {
    "LongForm": {
        "file": "LongForm/data/train-00000-of-00001-367270308b568067.parquet",
        "format": "parquet",
    },
    "python_code": {
        "file": "python_code_instructions_18k_alpaca/data/"
                "train-00000-of-00001-8b6e212f3e1ece96.parquet",
        "format": "parquet",
    },
    "WizardLM": {
        "file": "WizardLM_evol_instruct_V2_196k/"
                "WizardLM_evol_instruct_V2_143k.json",
        "format": "json",
    },
}


def _extract_prompt_longform(row: dict) -> str:
    """LongForm: 使用 'input' 字段作为 prompt。"""
    return row.get("input", "")


def _extract_prompt_python_code(row: dict) -> str:
    """python_code_instructions: 使用 'instruction' 字段, 可选拼接 'input'。"""
    inst = row.get("instruction", "")
    inp = row.get("input", "")
    if inp and inp.strip():
        return f"{inst}\n\nInput: {inp}"
    return inst


def _extract_prompt_wizardlm(row: dict) -> str:
    """WizardLM: 从 conversations 列表取第一个 human 消息。"""
    convs = row.get("conversations", [])
    for msg in convs:
        if isinstance(msg, dict) and msg.get("from") == "human":
            return msg.get("value", "")
    return ""


_EXTRACTORS = {
    "LongForm": _extract_prompt_longform,
    "python_code": _extract_prompt_python_code,
    "WizardLM": _extract_prompt_wizardlm,
}


class DatasetPool:
    """
    数据集池: 懒加载三个数据集, 提供 sample_prompts() 接口。
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 随机种子, 用于可复现的采样。
        """
        self._rng = random.Random(seed)
        self._loaded: dict = {}  # dataset_name → list[dict]

    def _load_dataset(self, name: str) -> List[dict]:
        """懒加载指定数据集, 返回 list of dict。"""
        if name in self._loaded:
            return self._loaded[name]

        cfg = _DATASETS[name]
        filepath = os.path.join(_DATASET_ROOT, cfg["file"])

        if not os.path.exists(filepath):
            print(f"[DatasetPool] ⚠ 文件不存在: {filepath}")
            self._loaded[name] = []
            return []

        if cfg["format"] == "parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(filepath)
            rows = table.to_pydict()
            # 转为 list of dict
            n = table.num_rows
            data = [{col: rows[col][i] for col in rows} for i in range(n)]

        elif cfg["format"] == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        self._loaded[name] = data
        print(f"[DatasetPool] 加载 {name}: {len(data)} 条")
        return data

    def sample_prompts(self, n: int = 10, min_length: int = 10,
                       max_length: int = 2000) -> List[dict]:
        """
        从三个数据集中随机采样 n 条 prompt。

        每条 prompt 随机选择来自哪个数据集。
        过滤掉过短或过长的 prompt。

        Args:
            n:          采样数量
            min_length: prompt 最小字符长度
            max_length: prompt 最大字符长度

        Returns:
            list of dict, 每个 dict 包含:
                - "prompt": str (原始 prompt 文本)
                - "source": str (数据集名称)
        """
        dataset_names = list(_DATASETS.keys())
        results = []
        max_attempts = n * 20  # 防止死循环
        attempts = 0

        while len(results) < n and attempts < max_attempts:
            attempts += 1
            # 随机选数据集
            ds_name = self._rng.choice(dataset_names)
            data = self._load_dataset(ds_name)
            if not data:
                continue

            # 随机选一条
            row = self._rng.choice(data)
            extractor = _EXTRACTORS[ds_name]
            prompt = extractor(row)

            # 长度过滤
            if len(prompt) < min_length or len(prompt) > max_length:
                continue

            results.append({
                "prompt": prompt,
                "source": ds_name,
            })

        if len(results) < n:
            print(f"[DatasetPool] ⚠ 仅采样到 {len(results)}/{n} 条 "
                  f"(长度范围 {min_length}-{max_length})")

        # 打印来源分布
        from collections import Counter
        dist = Counter(r["source"] for r in results)
        print(f"[DatasetPool] 采样 {len(results)} 条 prompt, "
              f"来源分布: {dict(dist)}")

        return results


# ---- 便捷函数 ----

def load_prompts(n: int = 10, seed: Optional[int] = None,
                 min_length: int = 10, max_length: int = 2000) -> List[dict]:
    """
    一行代码从三个数据集随机采样 prompt。

    Args:
        n:          采样数量
        seed:       随机种子
        min_length: prompt 最小字符长度
        max_length: prompt 最大字符长度

    Returns:
        list of dict, 每个 dict 包含 "prompt" 和 "source"
    """
    pool = DatasetPool(seed=seed)
    return pool.sample_prompts(n=n, min_length=min_length,
                               max_length=max_length)
