from __future__ import annotations

import re


def preprocess_text(text):
    import opencc

    word_to_num = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for word, num in word_to_num.items():
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(pattern, num, text, flags=re.IGNORECASE)

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    cc = opencc.OpenCC("t2s")
    text = cc.convert(text)

    # Special handling for spaces between Chinese characters:
    # - Keep single spaces between English words/numbers
    # - Remove spaces only when surrounded by Chinese characters on both sides to prevent incorrect word segmentation
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    return text.lower().strip()


def cosine_similarity_text(text1, text2, n: int = 3):
    from collections import Counter

    if not text1 or not text2:
        return 0.0

    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    print(f"cosine similarity text1 is: {text1}, text2 is: {text2}")

    ngrams1 = [text1[i : i + n] for i in range(len(text1) - n + 1)]
    ngrams2 = [text2[i : i + n] for i in range(len(text2) - n + 1)]

    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)

    all_ngrams = set(counter1.keys()) | set(counter2.keys())
    vec1 = [counter1.get(ng, 0) for ng in all_ngrams]
    vec2 = [counter2.get(ng, 0) for ng in all_ngrams]

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)
