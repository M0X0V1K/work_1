"""
Скрипт для проверки
python evaluate.py input.txt gold.txt
В input.txt — исходные предложения, в gold.txt — правильные разборы.
Оба файла должны содержать одинаковое количество строк.
"""

import re
import sys
from pathlib import Path

from lemmatizer import DATA_DIR, load_dictionary, process_line, normalize, candidates


def parse_tagged_line(line):
    result = []
    for part in line.split():
        match = re.fullmatch(r"(.+?)\{([^{}=]+)=([^{}=]+)\}", part)
        if not match:
            raise ValueError(f"Некорректная разметка: {part}")
        token, lemma, pos = match.groups()
        result.append((token, normalize(lemma), pos))
    return result


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Использование: python evaluate.py input.txt gold.txt")

    with open(sys.argv[1], encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(sys.argv[2], encoding="utf-8") as f:
        gold_lines = f.read().splitlines()
    if len(lines) != len(gold_lines):
        raise ValueError("Разное количество предложений в input и gold")

    dictionary = load_dictionary(DATA_DIR)
    total = correct_lemmas = correct_pos = correct_both = unknown = 0

    for line_no, (source, gold_line) in enumerate(zip(lines, gold_lines), 1):
        pred = parse_tagged_line(process_line(source, dictionary))
        gold = parse_tagged_line(gold_line)
        if len(pred) != len(gold):
            raise ValueError(f"Разное количество токенов в строке {line_no}")

        for (token, lemma, pos), (g_token, g_lemma, g_pos) in zip(pred, gold):
            if normalize(token) != normalize(g_token):
                raise ValueError(f"Разные токены: {token} и {g_token}")
            total += 1
            if not candidates(token, dictionary):
                unknown += 1
            good_lemma = lemma == g_lemma
            good_pos = pos == g_pos
            correct_lemmas += good_lemma
            correct_pos += good_pos
            correct_both += good_lemma and good_pos

    if not total:
        raise ValueError("Пустая тестовая выборка")
    print(f"Токенов: {total}")
    print(f"Неизвестных слов: {unknown} ({unknown / total:.1%})")
    print(f"Точность лемм: {correct_lemmas / total:.1%}")
    print(f"Точность POS: {correct_pos / total:.1%}")
    print(f"Лемма + POS одновременно: {correct_both / total:.1%}")


if __name__ == "__main__":
    main()
