import pickle, numpy as np

inp = "test.cz"
out = "test.utf8.cz"

with open(inp, "rb") as f:
    data = pickle.load(f)

def to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    elif isinstance(x, np.ndarray):
        # Flatten array, join elements converted to str
        return " ".join(to_str(el) for el in x.tolist())
    elif isinstance(x, (list, tuple)):
        return " ".join(to_str(el) for el in x)
    else:
        return str(x)

with open(out, "w", encoding="utf-8") as f:
    for x in data:
        f.write(to_str(x).strip() + "\n")

print(f"Wrote {out} with {len(data)} lines.")