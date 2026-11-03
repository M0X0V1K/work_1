"""Обучение собственной BiGRU-модели лемматизации и POS-теггинга.

Зависимость: pip install torch
Запуск: python train.py --epochs 12 --dict-epochs 2

Данные скачиваются автоматически в dicts/ (только при отсутствии файлов):
  OpenRussian (Badestrand/russian-dictionary), CC BY-SA 4.0;
  UD Russian GSD, CC BY-SA 4.0;
  UD Russian SynTagRus (optional), CC BY-NC-SA 4.0.
Морфологические анализаторы не используются. Разбиение UD train/dev раздельное;
никогда не используйте тестовый gold для обучения!
"""
import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import re
import sys
from urllib.request import urlretrieve

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
TAGS = ("S", "V", "A", "ADV", "NI", "PR", "CONJ", "NUM", "INTJ", "X")
UD_TAGS = {
    "NOUN": "S", "PROPN": "S", "VERB": "V", "AUX": "V", "ADJ": "A",
    "ADV": "ADV", "PART": "ADV", "PRON": "NI", "DET": "NI",
    "ADP": "PR", "CCONJ": "CONJ", "SCONJ": "CONJ", "NUM": "NUM",
    "INTJ": "INTJ", "X": "X", "SYM": "X",
}
WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
FORM_COLUMNS = ("sg_", "pl_", "past_", "presfut_", "imperative_", "decl_", "short_")
DATASETS = {
    "nouns.csv": "https://raw.githubusercontent.com/Badestrand/russian-dictionary/master/nouns.csv",
    "verbs.csv": "https://raw.githubusercontent.com/Badestrand/russian-dictionary/master/verbs.csv",
    "adjectives.csv": "https://raw.githubusercontent.com/Badestrand/russian-dictionary/master/adjectives.csv",
    "others.csv": "https://raw.githubusercontent.com/Badestrand/russian-dictionary/master/others.csv",
    "ru_gsd-ud-train.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-GSD/master/ru_gsd-ud-train.conllu",
    "ru_gsd-ud-dev.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-GSD/master/ru_gsd-ud-dev.conllu",
    "ru_syntagrus-ud-train-a.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-train-a.conllu",
    "ru_syntagrus-ud-train-b.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-train-b.conllu",
    "ru_syntagrus-ud-train-c.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-train-c.conllu",
    "ru_syntagrus-ud-dev.conllu": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-dev.conllu",
}


def norm(word):
    return str(word).lower().replace("ё", "е").replace("\u0301", "").replace("'", "").strip()


def clean_word(word):
    w = norm(word)
    return w if w and WORD_RE.fullmatch(w) and len(w) <= 36 else None


def forms(cell):
    for piece in re.split(r"[,/;]+", cell or ""):
        w = clean_word(piece.strip(" ()[]\""))
        if w:
            yield w


def ensure_data(data_dir, filename, offline=False):
    target = data_dir / filename
    if target.is_file():
        return target
    if offline:
        raise FileNotFoundError(f"Нет {target}. Положите файл в dicts/ или уберите --offline.")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {filename} ...", flush=True)
    try:
        urlretrieve(DATASETS[filename], target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Не получилось скачать {filename}: {exc}\n"
                           f"Загрузите вручную из {DATASETS[filename]} в {data_dir}") from exc
    return target


def load_dictionary(data_dir, offline):
    rows = []
    for fname, pos in (("nouns.csv", "S"), ("verbs.csv", "V"), ("adjectives.csv", "A")):
        path = ensure_data(data_dir, fname, offline)
        with path.open(encoding="utf-8-sig", newline="") as f:
            for record in csv.DictReader(f, delimiter="\t"):
                lemma = clean_word(record.get("bare", ""))
                if pos == "S":
                    nom = list(forms(record.get("sg_nom", "")))
                    if nom:
                        lemma = nom[0]
                if not lemma:
                    continue
                variants = {lemma}
                variants.update(forms(record.get("bare", "")))
                for key, val in record.items():
                    if key and (key.startswith(FORM_COLUMNS) or key in ("comparative", "superlative")):
                        variants.update(forms(val))
                for form in variants:
                    rows.append((form, lemma, pos))

    prepositions = "в во на с со к ко у о об обо от до по за из изо для при через без под подо над перед между после около возле среди сквозь вне внутри вдоль вокруг кроме вместо ради согласно благодаря против напротив помимо насчет насчёт про мимо".split()
    conjunctions = "и а но или либо однако зато чтобы хотя если когда пока поскольку ибо будто словно как также причем причём потому что да".split()
    particles_adv = "не ни ли же бы б ведь вот уж даже только лишь почти разве именно просто пусть пускай ну еще ещё очень уже вчера сегодня завтра здесь там тут сейчас всегда тогда быстро медленно позже рано дома вместе рядом чуть совсем далеко близко впереди назад поэтому так тоже хорошо плохо никогда всегда куда откуда почему зачем сколько да нет".split()
    special = []
    for pos, words in (("PR", prepositions), ("CONJ", conjunctions), ("ADV", particles_adv)):
        special.extend((norm(w), norm(w), pos) for w in words)
    pronouns = {
        "я": "я меня мне мной мною", "ты": "ты тебя тебе тобой тобою",
        "он": "он его него ему нему им ним нем нём", "она": "она ее её нее неё ей ней ею нею",
        "оно": "оно", "мы": "мы нас нам нами", "вы": "вы вас вам вами",
        "они": "они их них им ним ими ними", "себя": "себя себе собой собою",
        "кто": "кто кого кому кем ком", "что": "что чего чему чем чём",
        "весь": "весь вся все всё всего всей всю всем всеми всех",
        "мой": "мой моя мое моё мои моего моей мою моим моими моих",
        "твой": "твой твоя твое твоё твои твоего твоей твою твоим твоими твоих",
        "этот": "этот эта это эти этого этой эту этим этими этих",
        "тот": "тот та то те того той ту тем теми тех",
        "наш": "наш наша наше наши нашего нашей нашу нашим нашими наших",
        "ваш": "ваш ваша ваше ваши вашего вашей вашу вашим вашими ваших",
        "свой": "свой своя свое своё свои своего своей свою своим своими своих",
        "никто": "никто никого никому никем", "ничто": "ничто ничего ничему ничем",
        "каждый": "каждый каждая каждое каждые каждого каждой каждую каждым каждых",
        "никуда": "никуда", "некуда": "некуда", "где": "где", "куда": "куда",
    }
    for lemma, ws in pronouns.items():
        special.extend((norm(w), norm(lemma), "NI") for w in ws.split())
    numbers = {
        "один": "один одна одно одни одного одной одну одним одних одними",
        "два": "два две двух двум двумя", "три": "три трех трёх трем трём тремя",
        "четыре": "четыре четырех четырёх четырем четырём четырьмя",
        "пять": "пять пяти пятью", "шесть": "шесть шести шестью",
        "семь": "семь семи семью", "восемь": "восемь восьми восемью",
        "девять": "девять девяти девятью", "десять": "десять десяти десятью",
    }
    for lemma, ws in numbers.items():
        special.extend((norm(w), norm(lemma), "NUM") for w in ws.split())

    path = ensure_data(data_dir, "others.csv", offline)
    lookup = {(f, p): l for f, l, p in special}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for record in csv.DictReader(f, delimiter="\t"):
            word = clean_word(record.get("bare", ""))
            if word:
                for (w, p), lemma in lookup.items():
                    if w == word:
                        rows.append((word, lemma, p))
    rows.extend(special * 20)
    return list(dict.fromkeys(rows)), special


def load_conllu(path):
    sentences, current = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4 or not cols[0].isdigit():
                continue
            form, lemma, tag = cols[1], cols[2], UD_TAGS.get(cols[3])
            wf, lm = clean_word(form), clean_word(lemma)
            if tag and wf and lm:
                current.append((wf, lm, tag))
        if current:
            sentences.append(current)
    return sentences


def edit(form, lemma):
    k = 0
    while k < min(len(form), len(lemma)) and form[k] == lemma[k]:
        k += 1
    return (len(form) - k, lemma[k:])


def apply_edit(form, op):
    delete_count, add = op
    return form[:len(form) - delete_count] + add if delete_count <= len(form) else None


class ContextBiGRU(nn.Module):
    """Символьная BiGRU -> контекстная BiGRU -> POS и правка окончания."""
    def __init__(self, char_count, operation_count, char_emb=48, char_hidden=64, word_hidden=96):
        super().__init__()
        self.char_embedding = nn.Embedding(char_count, char_emb, padding_idx=0)
        self.char_gru = nn.GRU(char_emb, char_hidden, bidirectional=True, batch_first=True)
        self.context_gru = nn.GRU(char_hidden * 2, word_hidden, bidirectional=True, batch_first=True)
        dim = char_hidden * 2 + word_hidden * 2
        self.pos_head = nn.Linear(dim, len(TAGS))
        self.operation_head = nn.Linear(dim, operation_count)

    def forward(self, chars, char_lengths, sent_lengths):
        # chars: [N, C] для реальных токенов, sent_lengths: [B] (N=sum(sent_lengths)).
        emb = self.char_embedding(chars)
        packed = pack_padded_sequence(emb, char_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.char_gru(packed)
        words = torch.cat([hidden[-2], hidden[-1]], -1)
        pieces = torch.split(words, sent_lengths.tolist())
        padded = nn.utils.rnn.pad_sequence(pieces, batch_first=True)
        sent_packed = pack_padded_sequence(padded, sent_lengths.cpu(), batch_first=True, enforce_sorted=False)
        context_packed, _ = self.context_gru(sent_packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(context_packed, batch_first=True)
        context = torch.cat([context[i, :int(length)] for i, length in enumerate(sent_lengths)], dim=0)
        out = torch.cat([words, context], dim=-1)
        return self.pos_head(out), self.operation_head(out)


@dataclass
class Encoder:
    char_to_id: dict
    operations: list

    def __post_init__(self):
        self.op_to_id = {tuple(op): i for i, op in enumerate(self.operations)}
        self.tag_to_id = {tag: i for i, tag in enumerate(TAGS)}

    def chars(self, word):
        return [self.char_to_id.get(ch, 1) for ch in norm(word)] or [1]

    def collate(self, sentences):
        tokens = [triple for sentence in sentences for triple in sentence]
        enc = [self.chars(form) for form, _, _ in tokens]
        widths = torch.tensor([len(a) for a in enc], dtype=torch.long)
        chars = torch.zeros((len(enc), int(widths.max())), dtype=torch.long)
        for i, seq in enumerate(enc):
            chars[i, :len(seq)] = torch.tensor(seq)
        sizes = torch.tensor([len(s) for s in sentences], dtype=torch.long)
        pos = torch.tensor([self.tag_to_id[tag] for _, _, tag in tokens])
        ops = torch.tensor([self.op_to_id.get(edit(f, lemma), -100) for f, lemma, _ in tokens])
        return chars, widths, sizes, pos, ops, tokens


def build_encoder(dict_rows, train_sents, max_ops):
    chars = set("абвгдеежзийклмнопрстуфхцчшщъыьэюя0123456789-")
    counts = Counter()
    for form, lemma, _ in dict_rows:
        chars.update(form)
        counts[edit(form, lemma)] += 1
    for sent in train_sents:
        for form, lemma, _ in sent:
            chars.update(form)
            counts[edit(form, lemma)] += 8
    operations = [(0, "")] + [op for op, _ in counts.most_common() if op != (0, "")][:max_ops - 1]
    vocab = {ch: i + 2 for i, ch in enumerate(sorted(chars))}
    return Encoder(vocab, operations)


def epoch(model, loader, optimizer, device, is_train=True):
    model.train(is_train)
    n, pos_ok, lemma_ok, both_ok, loss_sum = 0, 0, 0, 0, 0.
    with torch.enable_grad() if is_train else torch.no_grad():
        for chars, lengths, sent_lengths, pos, ops, examples in loader:
            pos = pos.to(device)
            ops = ops.to(device)
            pos_logits, op_logits = model(chars.to(device), lengths, sent_lengths)
            loss = nn.functional.cross_entropy(pos_logits, pos)
            if (ops != -100).any():
                loss = loss + nn.functional.cross_entropy(op_logits, ops, ignore_index=-100)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            pred_pos = pos_logits.argmax(1).cpu()
            pred_op = op_logits.argmax(1).cpu()
            pos_correct = pred_pos == pos.cpu()
            # Валидация лемм — по предсказанному результату, а не по номеру операции.
            lem_correct = torch.tensor([
                apply_edit(form, loader.collate_fn.operations[int(op)]) == lm
                for (form, lm, _), op in zip(examples, pred_op.tolist())
            ], dtype=torch.bool)
            n += len(examples)
            pos_ok += int(pos_correct.sum())
            lemma_ok += int(lem_correct.sum())
            both_ok += int((pos_correct & lem_correct).sum())
            loss_sum += loss.item() * len(examples)
    return (loss_sum / max(1, n), lemma_ok / max(1, n), pos_ok / max(1, n), both_ok / max(1, n))


def collate_with(encoder):
    # Callable с доступными операциями (для вычисления метрик без глобальных переменных).
    class Collator:
        def __init__(self, enc):
            self.operations = enc.operations
            self.enc = enc
        def __call__(self, items):
            return self.enc.collate(items)
    return Collator(encoder)


def main():
    parser = argparse.ArgumentParser(description="Обучить BiGRU на OpenRussian и UD Russian GSD")
    parser.add_argument("--dicts", type=Path, default=ROOT / "dicts")
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument("--treebank", choices=("gsd", "syntagrus"), default="gsd",
                        help="gsd — компактный корпус; syntagrus — значительно больше предложений")
    parser.add_argument("--dict-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--dict-limit", type=int, default=160000, help="0=все словоформы")
    parser.add_argument("--max-operations", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32, help="предложений в батче")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--train-conllu", type=Path, help="свой размеченный корпус CoNLL-U train")
    parser.add_argument("--dev-conllu", type=Path, help="отдельный корпус CoNLL-U dev")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    print(f"Устройство: {device}", flush=True)
    if args.epochs <= 0 or args.max_operations < 2 or args.batch_size < 1:
        parser.error("Параметры --epochs, --batch-size должны быть > 0; --max-operations >= 2")
    rows, special = load_dictionary(args.dicts, args.offline)
    if args.train_conllu:
        train_paths = [args.train_conllu]
    elif args.treebank == "syntagrus":
        train_paths = [ensure_data(args.dicts, f"ru_syntagrus-ud-train-{part}.conllu", args.offline)
                       for part in "abc"]
    else:
        train_paths = [ensure_data(args.dicts, "ru_gsd-ud-train.conllu", args.offline)]
    default_dev = "ru_syntagrus-ud-dev.conllu" if args.treebank == "syntagrus" else "ru_gsd-ud-dev.conllu"
    dev_path = args.dev_conllu or ensure_data(args.dicts, default_dev, args.offline)
    if any(p.resolve() == dev_path.resolve() for p in train_paths):
        raise ValueError("Train и dev должны быть разными файлами — иначе будет утечка данных!")
    training = [s for path in train_paths for s in load_conllu(path) if s]
    validation = [s for s in load_conllu(dev_path) if s]
    if not training or not validation:
        raise ValueError("В train/dev CoNLL-U отсутствуют размеченные предложения")
    rows = list(dict.fromkeys(rows))
    encoder = build_encoder(rows, training, args.max_operations)
    if args.dict_limit and len(rows) > args.dict_limit:
        rows = random.sample(rows, args.dict_limit)
        rows.extend(special * 10)
    dict_sentences = [[row] for row in rows]
    train_collate = collate_with(encoder)
    make_loader = lambda data, batch, shuffle: DataLoader(data, batch_size=batch, shuffle=shuffle,
                                                         collate_fn=train_collate, num_workers=0)
    dl_dict = make_loader(dict_sentences, 256, True)
    dl_train = make_loader(training, args.batch_size, True)
    dl_dev = make_loader(validation, args.batch_size, False)
    config = {"char_emb": 48, "char_hidden": 64, "word_hidden": 96}
    model = ContextBiGRU(len(encoder.char_to_id) + 2, len(encoder.operations), **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Примеров словаря: {len(rows)}; corpus train: {len(training)} предложений; "
          f"dev: {len(validation)}; операций: {len(encoder.operations)}; POS: {TAGS}", flush=True)
    if not any(p in [x[2] for s in training for x in s] for p in ("PR", "NI", "CONJ")):
        print("ВНИМАНИЕ: корпус не содержит служебных POS — качество будет низким", file=sys.stderr)
    for i in range(args.dict_epochs):
        l, la, pa, both = epoch(model, dl_dict, optimizer, device, True)
        print(f"Словарь {i+1}/{args.dict_epochs}: loss={l:.3f} lemma={la:.1%} POS={pa:.1%} BOTH={both:.1%}", flush=True)
    best = -1
    for i in range(args.epochs):
        tr = epoch(model, dl_train, optimizer, device, True)
        va = epoch(model, dl_dev, optimizer, device, False)
        print(f"Corpus {i+1}/{args.epochs}: train lemma={tr[1]:.1%} POS={tr[2]:.1%} "
              f"BOTH={tr[3]:.1%} | DEV lemma={va[1]:.1%} POS={va[2]:.1%} "
              f"BOTH={va[3]:.1%}", flush=True)
        if va[3] > best:
            best = va[3]
            args.model.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "operations": encoder.operations, "char_to_id": encoder.char_to_id,
                        "config": config, "tags": TAGS}, args.model)
            print(f"  Лучшая модель сохранена: {args.model}", flush=True)
    print(f"Готово. Лучшая DEV точность lemma+POS: {best:.2%}. Тестируйте отдельно на gold, не на train.")


if __name__ == "__main__":
    main()
