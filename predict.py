"""Лемматизация + POS исключительно через обученную рекуррентную сеть.

python predict.py --text "Я играю с лабубами"
python predict.py --input test_input.txt --gold test_gold.txt --output result.txt
python predict.py --input test_input.txt --gold test_gold.txt --quiet

Архитектура загружается из train.py, веса — из model.pt. CSV и словари
при обработке НЕ читаются. Точность показывается только при наличии gold.
"""
import argparse
from collections import Counter
from pathlib import Path
import re
import sys

import torch
from train import ROOT, TAGS, Encoder, ContextBiGRU, apply_edit, norm, WORD_RE

GOLD_RE = re.compile(r"(.+?)\{([^{}=]+)=([^{}=]+)\}")


def tokenize(line):
    """Удалить только разрешённую пунктуацию и разбить по пробелам."""
    return re.sub(r"[,.!?]", " ", line).split()


def parse_gold(line, number):
    result = []
    for part in line.split():
        m = GOLD_RE.fullmatch(part)
        if not m:
            raise ValueError(f"Строка {number}, неправильный gold-токен {part!r}; нужен вид слово{{лемма=POS}}")
        result.append(m.groups())
    return result


class RNNTagger:
    def __init__(self, path, device="auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   "cpu" if device == "auto" else device)
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        for field in ("state_dict", "operations", "char_to_id", "config", "tags"):
            if field not in checkpoint:
                raise ValueError(f"Чекпойнт не подходит: отсутствует {field}. Обучите новую model.pt через train.py")
        self.operations = [tuple(op) for op in checkpoint["operations"]]
        self.tags = list(checkpoint["tags"])
        if self.tags != list(TAGS):
            raise ValueError("Несовместимые теги; используйте train.py и predict.py из одного комплекта")
        self.encoder = Encoder(checkpoint["char_to_id"], self.operations)
        self.model = ContextBiGRU(len(self.encoder.char_to_id) + 2, len(self.operations),
                                   **checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.invalid_top = 0

    @torch.inference_mode()
    def process_batch(self, lines):
        """Возвращает массивы (слово, лемма, POS, confidence POS, confidence lemma).
        Контекстная BiGRU должна видеть целое предложение, поэтому НЕ кэшируем слова!
        """
        words_by_line = [tokenize(line) for line in lines]
        valid_indices = [i for i, ws in enumerate(words_by_line) if ws]
        answers = [[] for _ in lines]
        if not valid_indices:
            return answers
        sentences = [[(norm(w), norm(w), "S") for w in words_by_line[i]] for i in valid_indices]
        chars, char_lengths, sent_lengths, _, _, _ = self.encoder.collate(sentences)
        pos_logits, operation_logits = self.model(chars.to(self.device), char_lengths, sent_lengths)
        pos_probs = pos_logits.softmax(-1).cpu()
        operation_logits = operation_logits.cpu()
        forms = [norm(w) for i in valid_indices for w in words_by_line[i]]
        word_lengths = torch.tensor([len(w) for w in forms])
        deletes = torch.tensor([op[0] for op in self.operations])
        append_lengths = torch.tensor([len(op[1]) for op in self.operations])
        # Ограничение сразу для всего батча: нет циклов по 2048 операциям
        # для каждого слова. Также исключаем операции, дающие пустую лемму.
        valid = ((word_lengths[:, None] >= deletes[None, :]) &
                 (word_lengths[:, None] - deletes[None, :] + append_lengths[None, :] > 0))
        masked = operation_logits.masked_fill(~valid, -float("inf"))
        if not bool(torch.isfinite(masked).any(dim=1).all()):
            raise RuntimeError("У части слов нет применимых операций лемматизации")
        first_ops = operation_logits.argmax(dim=-1)
        operation_ids = masked.argmax(dim=-1)
        self.invalid_top += int((first_ops != operation_ids).sum())
        operation_probs = masked.softmax(-1).gather(1, operation_ids[:, None])[:, 0]
        pos_conf, pos_ids = pos_probs.max(dim=-1)
        offset = 0
        for orig_i in valid_indices:
            for word in words_by_line[orig_i]:
                lemma = apply_edit(norm(word), self.operations[int(operation_ids[offset])])
                answers[orig_i].append((word, lemma, self.tags[int(pos_ids[offset])],
                                        float(pos_conf[offset]), float(operation_probs[offset])))
                offset += 1
        return answers


def run(lines, gold_lines, tagger, output=None, quiet=False, show_errors=12, batch_size=32):
    if gold_lines is not None and len(lines) != len(gold_lines):
        raise ValueError(f"Разное количество строк: input={len(lines)}, gold={len(gold_lines)}")
    total = good_lemma = good_pos = good_both = 0
    confidence_pos = confidence_op = 0.0
    by_pos_total, by_pos_correct = Counter(), Counter()
    errors = []
    stream = None
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("w", encoding="utf-8")
    try:
        for begin in range(0, len(lines), batch_size):
            batch = lines[begin:begin + batch_size]
            results = tagger.process_batch(batch)
            for j, predictions in enumerate(results):
                line_no = begin + j + 1
                tagged = " ".join(f"{w}{{{lemma}={pos}}}" for w, lemma, pos, _, _ in predictions)
                if not quiet:
                    print(tagged)
                if stream:
                    stream.write(tagged + "\n")
                for _, _, _, pc, oc in predictions:
                    total += 1
                    confidence_pos += pc
                    confidence_op += oc
                if gold_lines is None:
                    continue
                gold = parse_gold(gold_lines[line_no - 1], line_no)
                if len(gold) != len(predictions):
                    raise ValueError(f"Строка {line_no}: несовпадение токенов: input={len(predictions)}, gold={len(gold)}")
                for (word, lemma, pos, _, _), (gword, glemma, gpos) in zip(predictions, gold):
                    if norm(word) != norm(gword):
                        raise ValueError(f"Строка {line_no}: несовпадение слов {word!r} / {gword!r}")
                    lo = norm(lemma) == norm(glemma)
                    po = pos == gpos
                    good_lemma += lo
                    good_pos += po
                    good_both += lo and po
                    by_pos_total[gpos] += 1
                    by_pos_correct[gpos] += po
                    if not (lo and po) and len(errors) < show_errors:
                        errors.append((line_no, word, lemma, pos, glemma, gpos))
    finally:
        if stream:
            stream.close()
    print("\n" + "=" * 55)
    print("АНАЛИЗ")
    print("=" * 55)
    print(f"Предложений: {len(lines)} | Токенов: {total}")
    print(f"Пропущенных недопустимых первых операций: {tagger.invalid_top}")
    if output:
        print(f"Разбор сохранён в: {output}")
    if not total:
        print("Нет слов для анализа.")
        return
    print(f"Средняя уверенность POS: {confidence_pos/total:.2%} (не точность!)")
    print(f"Средняя уверенность операции: {confidence_op/total:.2%} (не точность!)")
    if gold_lines is None:
        print("Реальная точность недоступна: не передана эталонная разметка --gold.")
        return
    print("\nТОЧНОСТЬ ПО GOLD:")
    print(f"  Лемма: {good_lemma}/{total} = {good_lemma/total:.2%}")
    print(f"  POS: {good_pos}/{total} = {good_pos/total:.2%}")
    print(f"  Лемма + POS: {good_both}/{total} = {good_both/total:.2%}")
    print(f"  Ошибок: {total-good_both}/{total} = {(total-good_both)/total:.2%}")
    print("\nPOS по классам:")
    for pos in sorted(by_pos_total, key=lambda x: -by_pos_total[x]):
        n = by_pos_total[pos]
        ok = by_pos_correct[pos]
        print(f"  {pos:5}: {ok}/{n} = {ok/n:.2%}")
    if errors:
        print("\nПримеры ошибок (модель -> gold):")
        for line_no, word, lemma, pos, true_lemma, true_pos in errors:
            print(f"  строка {line_no}: {word}: {lemma}/{pos} -> {true_lemma}/{true_pos}")


def main():
    parser = argparse.ArgumentParser(description="Разбор предложений RNN, оценка по gold")
    parser.add_argument("text_arg", nargs="?", help="предложение (альтернатива --text)")
    parser.add_argument("--text", help="предложение в командной строке")
    parser.add_argument("--input", type=Path, help="текстовый файл: одно предложение на строку")
    parser.add_argument("--gold", type=Path, help="эталонный файл строка{лемма=POS}")
    parser.add_argument("--gold-text", help="эталон для одного предложения")
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument("--output", type=Path, help="сохранить предсказанную разметку")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--show-errors", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if sum(x is not None for x in (args.text, args.text_arg, args.input)) != 1:
        parser.error("Нужен ровно один вариант: --text, --input или предложение первым аргументом")
    if args.gold and args.gold_text:
        parser.error("Нельзя одновременно --gold и --gold-text")
    if args.batch_size < 1:
        parser.error("--batch-size должен быть положительным")
    lines = (args.input.read_text(encoding="utf-8-sig").splitlines() if args.input else
             (args.text if args.text is not None else args.text_arg).splitlines())
    gold_lines = (args.gold.read_text(encoding="utf-8-sig").splitlines() if args.gold else
                  args.gold_text.splitlines() if args.gold_text is not None else None)
    tagger = RNNTagger(args.model, args.device)
    run(lines, gold_lines, tagger, args.output, args.quiet, args.show_errors, args.batch_size)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"Ошибка: {exc}") from exc
