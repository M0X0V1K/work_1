"""
Запуск: python lemmatizer.py < input.txt
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

# others.csv не содержит служебные части речи, поэтому их
# разметил самостоятельно. Частицы помечены ADV
PREPOSITIONS = set("""
в во на с со к ко у о об обо от до по за из изо для при
через без под подо над перед между после около возле среди
сквозь вне внутри вдоль вокруг кроме вместо ради согласно
благодаря против напротив помимо насчет насчёт
""".split()) # Предлоги

CONJUNCTIONS = set("""
и а но или либо однако зато чтобы хотя если когда пока
поскольку ибо будто словно как зато также причем причём
""".split()) # Союзы

PRONOUN_LEMMAS = {
    "я": "я", "меня": "я", "мне": "я", "мной": "я", "мною": "я",
    "ты": "ты", "тебя": "ты", "тебе": "ты", "тобой": "ты", "тобою": "ты",
    "он": "он", "его": "он", "него": "он", "ему": "он", "нему": "он",
    "им": "он", "ним": "он", "нем": "он", "нём": "он",
    "она": "она", "ее": "она", "её": "она", "нее": "она", "неё": "она",
    "ей": "она", "ней": "она", "ею": "она", "нею": "она",
    "оно": "оно", "мы": "мы", "нас": "мы", "нам": "мы", "нами": "мы",
    "вы": "вы", "вас": "вы", "вам": "вы", "вами": "вы",
    "они": "они", "их": "они", "них": "они", "ими": "они", "ними": "они",
    "кто": "кто", "кого": "кто", "кому": "кто", "кем": "кто", "ком": "кто",
    "что": "что", "чего": "что", "чему": "что", "чем": "что",
    "себя": "себя", "себе": "себя", "собой": "себя", "собою": "себя",
    "все": "весь", "всё": "весь", "всех": "весь", "всем": "весь",
    "всеми": "весь", "весь": "весь", "вся": "весь", "всю": "весь",
    "всякого": "всякий", "каждый": "каждый", "никто": "никто",
    "ничто": "ничто", "некто": "некто", "нечто": "нечто",
    "никуда": "никуда", "нигде": "нигде", "некуда": "некуда",
    "никогда": "никогда", "откуда": "откуда", "куда": "куда",
} # Местоимения

ADVERBS_AND_PARTICLES = set("""
не ни ли же бы б ведь вот уж даже только лишь почти разве
именно просто пусть пускай ну еще ещё очень уже вчера
сегодня завтра здесь там тут сейчас всегда тогда быстро
медленно позже рано дома вместе рядом чуть совсем далеко
близко впереди назад поэтому так тоже
""".split()) # наречия частицы

# словарь для обработки конкретных случаев, которые не предусмотрены словарями
EXCEPTIONS = {
    "дальше": ("далеко", "ADV"),
    # "лучше": ("хорошо", "ADV"), в ходе тестирования столкнулся со случаями когда лучше может быть сравнительным прилагательным, тогда надо писать `хороший`
    # "хуже": ("плохо", "ADV"),
    # "больше": ("много", "ADV"),
    # "меньше": ("мало", "ADV")
}


def normalize(word):
    """Для поиска игнорируем ударения, регистры и различие е/ё.
    Нужно для нормализации слов
    """
    return word.lower().replace("ё", "е").replace("'", "").replace("\u0301", "") # u0301 - это апостров для ударений


def split_forms(cell):
    """В одном поле словаря встречаются варианты перечислений через запятую, / или //.
    Поэтому мы их так же нормализуем
    """
    for part in re.split(r"[,/;]+", cell or ""):
        word = normalize(part.strip().strip("()[] "))
        if re.fullmatch(r"[а-я]+(?:-[а-я]+)*", word):
            yield word


def add(index, form, lemma, pos):
    form = normalize(form)
    lemma = normalize(lemma)
    item = (lemma, pos)
    if form and item not in index[form]:
        index[form].append(item)


def load_dictionary(folder):
    index = defaultdict(list)

    for filename, pos in (
        ("nouns.csv", "S"),
        ("verbs.csv", "V"),
        ("adjectives.csv", "A"),
    ):
        with (folder / filename).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                lemma = normalize(row.get("bare", "").strip())
                if not lemma:
                    continue

                # У существительных bare — не начальная форма.
                # Например, у слова "кошки" sg_nom = "кошка".
                if pos == "S":
                    nominative = list(split_forms(row.get("sg_nom", "")))
                    if nominative:
                        lemma = nominative[0]

                add(index, lemma, lemma, pos)
                add(index, row["bare"], lemma, pos)

                # Берем только русские словоформы, не переводы и не метаданные.
                for column, value in row.items():
                    if column.startswith(("sg_", "pl_", "past_", "presfut_",
                                          "imperative_", "decl_", "short_")) \
                            or column in ("comparative", "superlative"):
                        for form in split_forms(value):
                            add(index, form, lemma, pos)

                # В словарях иногда нет сравнительной степени.
                # `Частотное` правило русского языка: стабильный -> стабильнее.
                if pos == "A" and lemma.endswith("ный"):
                    add(index, lemma[:-2] + "ее", lemma, "A")

    with (folder / "others.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            word = normalize(row.get("bare", "").strip())
            if not word:
                continue
            if word in PREPOSITIONS:
                add(index, word, word, "PR")
            elif word in CONJUNCTIONS:
                add(index, word, word, "CONJ")
            elif word in PRONOUN_LEMMAS:
                add(index, word, PRONOUN_LEMMAS[word], "NI")
            else:
                # В others.csv нет POS: для прочих слов это лишь теория.
                add(index, word, word, "ADV")

    return index


def candidates(word, index):
    key = normalize(word)
    if key in EXCEPTIONS:
        return [EXCEPTIONS[key]]
    if key in PRONOUN_LEMMAS:
        return [(normalize(PRONOUN_LEMMAS[key]), "NI")]
    if key in PREPOSITIONS:
        return [(key, "PR")]
    if key in CONJUNCTIONS:
        return [(key, "CONJ")]
    if key in ADVERBS_AND_PARTICLES:
        return [(key, "ADV")]
    return index.get(key, [])


def unknown_word(word, next_word, index):
    """Если слов нет в словаре, используем `частотные` правила русского языка"""
    w = normalize(word)
    # Притяжательные прилагательные: Гришины одногрупники -> гришин.
    if w.endswith(("ины", "овы", "евы")) and next_word:
        if any(pos == "S" for _, pos in candidates(next_word, index)):
            return w[:-1], "A"
    if w.endswith("ться") or w.endswith("ть"):
        return w, "V"
    if w.endswith(("ый", "ий", "ая", "ое")):
        return w, "A"
    # Для имени собственного: Пахры -> Пахра.
    if word[:1].isupper() and w.endswith("ы"):
        return w[:-1] + "а", "S"
    return w, "S"


def analyze(word, prev_word, next_word, index):
    variants = candidates(word, index)
    if not variants:
        return unknown_word(word, next_word, index)
    if len(variants) == 1:
        return variants[0]

    # После предлога чаще выбираем часть речи `существительное`:
    # "из стали" -> сталь/S, а не стать/V.
    if normalize(prev_word) in PREPOSITIONS:
        for pos in ("S", "NI", "A"):
            for lemma, tag in variants:
                if tag == pos:
                    return lemma, tag

    # Если у формы есть глагольная форма,чаще всего это будет верный вариант:
    # "они стали работать" -> стать/V.
    for pos in ("V", "S", "A", "NI", "ADV", "CONJ", "PR"):
        for lemma, tag in variants:
            if tag == pos:
                return lemma, tag
    return variants[0]


def process_line(line, index):
    words = re.sub(r"[,.!?]", " ", line).split()
    output = []
    for i, word in enumerate(words):
        prev_word = words[i - 1] if i else ""
        next_word = words[i + 1] if i + 1 < len(words) else ""
        lemma, pos = analyze(word, prev_word, next_word, index)
        output.append(f"{word}{{{lemma}={pos}}}")
    return " ".join(output)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Использование: python lemmatizer.py input.txt")

    index = load_dictionary(DATA_DIR)
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            print(process_line(line, index))


if __name__ == "__main__":
    main()
