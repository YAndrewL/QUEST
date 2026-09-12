from __future__ import annotations
import numpy as np

# markers trusted to name a lineage, and the name each carries when it leads
TRUSTED = {"CD3e": "T cell", "CD4": "helper T cell", "CD8": "cytotoxic T cell",
           "CD68": "macrophage", "CD31": "endothelial cell",
           "aSMA": "smooth muscle/myofibroblast",
           "PanCK": "epithelial cell", "EpCAM": "epithelial cell", "ECad": "epithelial cell",
           "Ki67": "proliferating cell"}
EPI = ["PanCK", "EpCAM", "ECad"]
# never a lineage on these panels, however strongly they lead -- see the docstring
ASSOC = ["CD20", "FoxP3", "CD163", "Podoplanin", "MPO"]
SUPPORT = ["CD45", "HLA-DR", "Vimentin"]


def name_state(sig, z):
    """sig: the centroid's markers at z >= 1, strongest first."""
    if not sig:
        return "unresolved"
    lead_t = max([m for m in sig if m in TRUSTED], key=lambda m: z[m], default=None)
    lead_a = max([m for m in sig if m in ASSOC], key=lambda m: z[m], default=None)
    if lead_t is None:
        if lead_a is None:
            return "unresolved"
        rest = [m for m in sig if m in ASSOC and m != lead_a]
        return (f"{lead_a}/{rest[0]}-associated cell" if rest
                else f"{lead_a}-associated cell")
    if lead_a is not None and z[lead_a] > z[lead_t]:
        rest = [m for m in sig if m in ASSOC and m != lead_a]
        return (f"{lead_a}/{rest[0]}-associated cell" if rest
                else f"{lead_a}-associated cell")
    if lead_t == "Ki67":
        if "MPO" in sig:
            return "MPO-associated proliferating cell"
        return "proliferating epithelial cell" if any(m in sig for m in EPI) \
            else "proliferating cell"
    return TRUSTED[lead_t]


def name_vocabulary(sigs, Z, markers):
    """name a whole vocabulary, then disambiguate repeated epithelial names by their own markers, wh..."""
    zs = [{m: Z[i][markers.index(m)] for m in markers} for i in range(len(sigs))]
    names = [name_state(sigs[i], zs[i]) for i in range(len(sigs))]
    for i, n in enumerate(names):
        if n == "epithelial cell":
            mm = [m for m in sigs[i] if m in EPI]
            if mm:
                names[i] = f"{'/'.join(mm)} epithelial cell"
    # the same disambiguation for repeated `proliferating cell`
    pro = [i for i, n in enumerate(names) if n == "proliferating cell"]
    if len(pro) > 1:
        for i in pro:
            sup = [m for m in sigs[i] if m in SUPPORT or m in ASSOC]
            if sup:
                names[i] = f"{sup[0]}-associated proliferating cell"
    return names


def name_cn(comp, names, lo=0.05, max_mixed=3):
    """A neighbourhood's annotation from its own composition, by PURITY tier."""
    comp = np.asarray(comp, float)
    comp = comp / max(comp.sum(), 1e-9)
    o = np.argsort(-comp)
    top = [(names[j], float(comp[j])) for j in o if comp[j] >= lo]
    if not top:
        return "unassignable"
    a, pa = top[0]
    if pa >= 0.85:
        # deliberately not `-pure`: `A-pure` and `A-rich` differ by one short word and read as
        # near-synonyms in a legend, so the two tiers of the same class were hard to tell apart
        return f"{a}-homogeneous"
    if pa >= 0.60:
        return f"{a}-rich"
    b = top[1][0] if len(top) > 1 else None
    if b is None:
        return f"{a}-rich"
    stroma = {"stromal / smooth muscle", "mesenchymal/fibroblast",
              "smooth muscle/myofibroblast"}
    ves = {"endothelial / lymphatic", "endothelial cell"}
    if (a in stroma and b in ves) or (a in ves and b in stroma):
        return "stromal–vascular mixed"
    imm = {"T cell", "helper T cell", "cytotoxic T cell", "macrophage", "myeloid"}
    if a in imm and b in imm:
        return f"{a}–{b} mixed immune"
    c = top[2][0] if len(top) > 2 and max_mixed > 2 else None
    return f"{a}–{b}–{c} mixed" if c else f"{a}–{b} mixed"


def name_cn_vocabulary(C, names, lo=0.10):
    """name a whole CN vocabulary and break ties."""
    out = [name_cn(C[c], names, lo) for c in range(len(C))]
    seen = {}
    for i, n in enumerate(out):
        seen.setdefault(n, []).append(i)
    for n, idx in seen.items():
        if len(idx) < 2:
            continue
        for i in idx:
            o = np.argsort(-np.asarray(C[i]))
            second = next((names[j] for j in o[1:] if C[i][j] > 0), None)
            out[i] = f"{n}, then {second}" if second else n
        if len(set(out[i] for i in idx)) < len(idx):
            for i in idx:
                out[i] = f"{out[i]} ({C[i].max():.0%})"
    return out


ASSOC_SET = set(ASSOC)


def family(name):
    """(family, detail) for a class name; two classes correspond iff `maps(a, b)`"""
    n = name.lower()
    if n == "unresolved":
        return ("unresolved", frozenset())
    if "epithelial" in n:
        return ("epithelial", frozenset(m for m in EPI if m.lower() in n))
    if "-associated cell" in n:
        head = name.split("-associated")[0]
        return ("assoc", frozenset(q for q in head.split("/") if q in ASSOC_SET))
    if "t cell" in n:
        return ("T cell", frozenset())
    if n in ("macrophage", "granulocyte"):
        return ("myeloid", frozenset())
    if "endothelial" in n or "lymphatic" in n or "vessel" in n:
        return ("vascular", frozenset())
    if "smooth muscle" in n or "mesenchymal" in n or "fibroblast" in n:
        return ("stromal", frozenset())
    if "proliferating" in n:
        return ("proliferating", frozenset())
    if "leukocyte" in n or "antigen-presenting" in n:
        return ("leukocyte", frozenset())
    return (n, frozenset())


def maps(a, b):
    fa, da = family(a)
    fb, db = family(b)
    if fa != fb:
        return False
    return bool(da & db) if fa == "assoc" else True


FAMILY_LABEL = {"epithelial": "epithelial", "T cell": "T cell", "myeloid": "myeloid",
                "vascular": "endothelial / lymphatic", "stromal": "stromal / smooth muscle",
                "proliferating": "proliferating", "leukocyte": "leukocyte",
                "unresolved": "unresolved"}


def family_label(name):
    f, d = family(name)
    if f == "assoc":
        return f"{'/'.join(sorted(d))}-associated" if d else "associated"
    return FAMILY_LABEL.get(f, f)


COMPARTMENT = {
    "T cell": "immune", "helper T cell": "immune", "cytotoxic T cell": "immune",
    "myeloid": "immune", "leukocyte": "immune", "B cell": "immune", "granulocyte": "immune",
    "antigen-presenting cell": "immune",
    "epithelial": "epithelial", "proliferating": "proliferating",
    "endothelial / lymphatic": "vascular",
    "stromal / smooth muscle": "stromal", "mesenchymal/fibroblast": "stromal",
    "unresolved": "unresolved",
}
# markers that place an `-associated` family in a compartment, since those families are named for
# their markers precisely because no lineage could be assigned
ASSOC_COMPARTMENT = {"CD20": "immune", "FoxP3": "immune", "CD163": "immune", "MPO": "immune",
                     "Podoplanin": "vascular"}


def compartment(fam):
    """the coarse tier for a family label"""
    if fam in COMPARTMENT:
        return COMPARTMENT[fam]
    if fam.endswith("-associated"):
        ms = [q for q in fam.replace("-associated", "").split("/") if q]
        votes = [ASSOC_COMPARTMENT[m] for m in ms if m in ASSOC_COMPARTMENT]
        if votes:
            return max(set(votes), key=votes.count)
    return fam


def correspond_two_tier(fams_a, fams_b):
    """label -> (level, key) for each arm; the level is decided PER COMPARTMENT."""
    ga, gb = {}, {}
    for f in set(fams_a):
        ga.setdefault(compartment(f), set()).add(f)
    for f in set(fams_b):
        gb.setdefault(compartment(f), set()).add(f)
    ka, kb = {}, {}
    for f in set(fams_a):
        c = compartment(f)
        ka[f] = ("family", f) if ga.get(c) == gb.get(c) else ("compartment", c)
    for f in set(fams_b):
        c = compartment(f)
        kb[f] = ("family", f) if ga.get(c) == gb.get(c) else ("compartment", c)
    shared = set(ka.values()) & set(kb.values())
    return ka, kb, shared
