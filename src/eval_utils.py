import re
from typing import Any


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_number(s: str):
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def simple_accuracy(pred: str, label: str) -> int:
    """
    Returns 1 if normalized texts match exactly OR numeric values match, else 0.
    """
    p = normalize_text(pred)
    l = normalize_text(label)
    if p == l:
        return 1
    # numeric fallback
    pn = extract_number(p)
    ln = extract_number(l)
    if pn is not None and ln is not None and abs(pn - ln) < 1e-6:
        return 1
    # fuzzy token overlap fallback
    ptoks = set(re.findall(r"[a-z0-9]+", p))
    ltoks = set(re.findall(r"[a-z0-9]+", l))
    if ptoks and ltoks:
        inter = len(ptoks & ltoks)
        union = len(ptoks | ltoks)
        if union > 0 and (inter / union) >= 0.6:
            return 1
    return 0
