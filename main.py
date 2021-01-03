import click
import seqeval.metrics.sequence_labeling


@click.command(name="Analyze")
@click.argument("pred_file", type=click.File("r"))
@click.option("--verbose", type=bool, is_flag=True)
def main(pred_file, verbose):
    n_wrong = 0
    n_wrong_unk = 0

    words = []
    gold_seqs = []
    pred_seqs = []

    for line in pred_file.readlines():
        line = line.rstrip("\n")
        if line == "":
            words.append("<SEP>")
            gold_seqs.append("O")
            pred_seqs.append("O")
            continue

        word, gold, pred = line.rstrip("\n").split(" ")

        if gold != pred:
            if "<UNK>" in word:
                n_wrong_unk += 1
            n_wrong += 1

        words.append(word)
        gold_seqs.append(gold)
        pred_seqs.append(pred)

    gold_spans = seqeval.metrics.sequence_labeling.get_entities(gold_seqs)
    pred_spans = seqeval.metrics.sequence_labeling.get_entities(pred_seqs)

    gold_span_reprs = set("-".join(map(str, elems)) for elems in gold_spans)

    n_wrong_spans = 0
    n_wrong_spans_unk = 0

    for label, start, last in pred_spans:
        if f"{label}-{start}-{last}" not in gold_span_reprs:
            if verbose:
                print(label, start, last)
                print(words[start:last + 1])

            if "<UNK>" in "".join(words[start:last + 1]):
                n_wrong_spans_unk += 1
            n_wrong_spans += 1

    print("#wrong tags: ", n_wrong_unk, n_wrong)
    print("#wrong spans: ", n_wrong_spans_unk, n_wrong_spans)


if __name__ == "__main__":
    main()
