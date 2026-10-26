# README

Перед выполнением рекомендую:

```cmd
chcp 65001
set PYTHONIOENCODING=utf-8
```


**`lemmatizer.py`** — лемматизирует русский текст и определяет части речи слов с помощью словарей и правил.

```bash
python lemmatizer.py < test_input.txt
```

**`evaluate.py`** — оценивает точность лемматизации и определения частей речи, сравнивая результат с эталонной разметкой.

```bash
python evaluate.py test_input.txt gold.txt
```

Для запуска оба скрипта и словари `nouns.csv`, `verbs.csv`, `adjectives.csv`, `others.csv` должны находиться в одной папке. `test_input.txt` содержит исходные предложения, `test_gold.txt` — эталонную разметку.