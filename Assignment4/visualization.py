#!/usr/bin/env python3
import re, math, argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_log(text: str):
    loss_matches = re.findall(r"Epoch\s+(\d+):\s+loss\s+([0-9.]+)", text)
    valid_matches = re.findall(
        r"Epoch\s+(\d+):\s+valid_loss\s+[0-9.]+\s+\|.*?valid_perplexity\s+([0-9.]+)\s+\|\s+BLEU\s+([0-9.]+)",
        text
    )
    final_bleu = re.search(r"Final Test Set Results:\s+BLEU\s+([0-9.]+)", text)
    loss_by_epoch = {int(e): float(l) for e, l in loss_matches}
    valid_by_epoch = {int(e): (float(p), float(b)) for e, p, b in valid_matches}
    epochs = sorted(set(loss_by_epoch) | set(valid_by_epoch))
    train_ppl, valid_ppl, valid_bleu = [], [], []
    use_epochs = []
    for e in epochs:
        if e in loss_by_epoch:
            use_epochs.append(e)
            train_ppl.append(math.exp(loss_by_epoch[e]))
            if e in valid_by_epoch:
                vp, vb = valid_by_epoch[e]
                valid_ppl.append(vp); valid_bleu.append(vb)
            else:
                valid_ppl.append(float("nan")); valid_bleu.append(float("nan"))
    return use_epochs, train_ppl, valid_ppl, valid_bleu, (float(final_bleu.group(1)) if final_bleu else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./training.out")
    ap.add_argument("--out", default="training_perplexity_bleu")
    ap.add_argument("--title", default="Training/Validation Perplexity and BLEU over Epochs")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    epochs, train_ppl, valid_ppl, valid_bleu, final_bleu = parse_log(text)

    fig, ax_left = plt.subplots(figsize=(8,5), dpi=150)
    ax_left.plot(epochs, train_ppl, marker="o", label="Training Perplexity")
    ax_left.plot(epochs, valid_ppl, marker="s", label="Validation Perplexity")
    ax_left.set_xlabel("Epoch"); ax_left.set_ylabel("Perplexity")
    ax_left.grid(True, linestyle="--", linewidth=0.5)

    ax_right = ax_left.twinx()
    ax_right.plot(epochs, valid_bleu, marker="^", linestyle="-.", label="Validation BLEU")
    if final_bleu is not None:
        ax_right.axhline(final_bleu, linestyle=":", linewidth=1.5, label=f"Final Test BLEU = {final_bleu:.2f}")
    ax_right.set_ylabel("BLEU")

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    plt.title(args.title)
    plt.tight_layout()
    png = f"{args.out}.png"; pdf = f"{args.out}.pdf"
    plt.savefig(png); plt.savefig(pdf)
    print(f"Saved: {png}\nSaved: {pdf}")

if __name__ == "__main__":
    main()
