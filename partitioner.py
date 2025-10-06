import xml.etree.ElementTree as ET
from pathlib import Path

ALIGN_XML = "DGT.en-sk.xml"   # your cesAlign file path
EN_TXT    = "DGT.en-sk.en"            # english sentences, one per line
SK_TXT    = "DGT.en-sk.sk"            # slovak sentences, one per line
N         = 5000

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        # Keep empty lines if your alignments expect them; most corpora trim
        return [line.rstrip("\n") for line in f]

def parse_links(align_xml_path):
    tree = ET.parse(align_xml_path)
    root = tree.getroot()
    links = []
    for link in root.iterfind(".//link"):
        xt = link.attrib.get("xtargets", "").strip()
        if not xt:
            continue
        # xtargets can look like "1;1" or "1 2;3 4" (varies by corpus)
        # Normalize separators to spaces, then split into left/right
        left_raw, right_raw = xt.split(";")
        # Split on whitespace (handles spaces and mixed separators)
        left_idxs  = [int(t) for t in left_raw.replace(";", " ").split() if t.isdigit()]
        right_idxs = [int(t) for t in right_raw.replace(";", " ").split() if t.isdigit()]
        if left_idxs and right_idxs:
            links.append((left_idxs, right_idxs))
    return links

def safe_get_concat(lines, one_based_indices):
    # Convert 1-based -> 0-based; join multiple with a space
    parts = []
    for idx in one_based_indices:
        i = idx - 1
        if 0 <= i < len(lines):
            parts.append(lines[i])
        else:
            # Skip out-of-range; you could also raise if you prefer strictness
            continue
    return " ".join(parts).strip()

def main():
    en_lines = load_lines(EN_TXT)
    sk_lines = load_lines(SK_TXT)
    links = parse_links(ALIGN_XML)

    aligned = []
    for left_idxs, right_idxs in links:
        en_seg = safe_get_concat(en_lines, left_idxs)
        sk_seg = safe_get_concat(sk_lines, right_idxs)
        if en_seg and sk_seg:
            aligned.append((en_seg, sk_seg))
        if len(aligned) >= N:
            break

    if not aligned:
        raise RuntimeError("No aligned pairs found—check file paths/formats.")

    out_en  = Path("batch5000.en")
    out_sk  = Path("batch5000.sk")
    out_tsv = Path("batch5000.tsv")

    with out_en.open("w", encoding="utf-8") as fe, \
         out_sk.open("w", encoding="utf-8") as fs, \
         out_tsv.open("w", encoding="utf-8") as ft:
        for en, sk in aligned:
            fe.write(en + "\n")
            fs.write(sk + "\n")
            ft.write(en.replace("\t", " ") + "\t" + sk.replace("\t", " ") + "\n")

    print(f"Wrote {len(aligned)} pairs to:\n- {out_en}\n- {out_sk}\n- {out_tsv}")

if __name__ == "__main__":
    main()
