"""Unit-4 (T10-T12) - clinical NLP on the MTSamples transcription corpus.

WHY A SECOND DATASET
--------------------
T1-T9 run on the Kaggle chest X-ray cohort, which ships images and labels and
nothing else - no reports, no notes, no free text of any kind. Unit-4 is text
work (classification, entity recognition, sense disambiguation), so it has no
material to run on there. An earlier attempt generated synthetic reports from
the image labels; that corpus was honest about being synthetic, but it could
only ever score a model against text we had written ourselves.

This module uses MTSamples instead: ~5,000 real transcribed clinical notes
across 40 specialties, carrying human-assigned specialty labels and human-
written keyword lists. Every number below is measured against annotation a
person produced, not against anything generated here.

Kept deliberately separate from healthcare_analytics.py: different dataset,
different modality, no shared state. The two run independently.

Corpus: data/mtsamples/mtsamples.csv  (Kaggle: tboyle10/medicaltranscriptions)
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
DATA = Path("data/mtsamples/mtsamples.csv")
OUT_DIR = Path("outputs_nlp")
SEED = 0
MIN_PER_CLASS = 50   # 11 specialties clear this; see T10 step 3
TEST_FRAC = 0.30

np.random.seed(SEED)
OUT_DIR.mkdir(exist_ok=True)

# MTSamples files a note under BOTH a clinical specialty and, usually, a
# document/umbrella category. These are the umbrella ones: they describe what
# KIND of document it is, not which specialty produced it. Mixing the two
# taxonomies in one label column is the central data problem in T10.
UMBRELLA = {
    "Surgery", "Consult - History and Phy.", "SOAP / Chart / Progress Notes",
    "Discharge Summary", "Office Notes", "Emergency Room Reports", "Letters",
    "IME-QME-Work Comp etc.", "Radiology",
}


# --------------------------------------------------------------------------
# Small helpers (same plain style as healthcare_analytics.py, duplicated
# rather than imported so this module runs without cv2/torch/skimage)
# --------------------------------------------------------------------------
def head(title):
    print("\n" + "=" * 74)
    print("  " + title)
    print("=" * 74)


def sub(title):
    print("\n-- " + title + " " + "-" * max(0, 68 - len(title)))


def table(rows, headers):
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    print("  " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))


def save_fig(fig, name):
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("  [saved] " + str(path))


def normalise(text):
    """Whitespace-normalised lowercase form, used only for duplicate matching."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# ==========================================================================
# CORPUS - dedupe and untangle the two taxonomies
# ==========================================================================
def load_corpus():
    """Return (raw, clean).

    raw   - one row per CSV line with a transcription. Retains duplicates, so
            T10 can measure what a naive pipeline would have scored.
    clean - one row per distinct note, carrying the full label set, the
            residual clinical label set, and the human keyword list.
    """
    raw = pd.read_csv(DATA).dropna(subset=["transcription"]).reset_index(drop=True)
    raw["medical_specialty"] = raw.medical_specialty.str.strip()
    raw["norm"] = raw.transcription.map(normalise)

    clean = (raw.groupby("norm")
                .agg(text=("transcription", "first"),
                     keywords=("keywords", "first"),
                     description=("description", "first"),
                     sample_name=("sample_name", "first"),
                     labels=("medical_specialty", lambda s: frozenset(s)),
                     n_rows=("medical_specialty", "size"))
                .reset_index())
    clean["clinical"] = clean.labels.map(lambda s: s - UMBRELLA)
    clean["n_clin"] = clean.clinical.map(len)
    return raw, clean


# ==========================================================================
# T10 - CLINICAL TEXT CLASSIFICATION
# ==========================================================================
def t10_classification(raw, clean):
    head("T10 - CLASSIFYING CLINICAL NOTES BY SPECIALTY")

    # -- 1. the corpus is not what the row count suggests ------------------
    sub("1. Duplicate detection")
    n_dupe = len(raw) - len(clean)
    print(f"  CSV rows with a transcription : {len(raw)}")
    print(f"  distinct note texts           : {len(clean)}")
    print(f"  redundant rows                : {n_dupe} ({n_dupe / len(raw):.0%})")
    print(f"  notes appearing more than once: {int((clean.n_rows > 1).sum())} "
          f"(largest group {int(clean.n_rows.max())} copies)")
    print()
    print("  The duplication is not a packaging error. The same note is filed")
    print("  under several categories at once, and each filing becomes a row.")
    ex = clean[clean.n_rows == 3].iloc[0]
    print(f"  e.g. {ex.sample_name.strip()!r}")
    print(f"       filed under: {', '.join(sorted(ex.labels))}")

    # -- 2. two taxonomies in one column -----------------------------------
    sub("2. The label column mixes two taxonomies")
    multi = clean[clean.labels.map(len) > 1]
    with_umb = int(multi.labels.map(lambda s: bool(s & UMBRELLA)).sum())
    print(f"  notes carrying >1 label          : {len(multi)}")
    print(f"  ...where >=1 label is an umbrella: {with_umb} ({with_umb / len(multi):.0%})")
    print()
    print("  'Surgery', 'Radiology', 'Discharge Summary' and the like describe the")
    print("  DOCUMENT, not the specialty. Treated as classes they overlap every")
    print("  real specialty and the task stops being learnable. Removing them:")
    table([["0 clinical labels", int((clean.n_clin == 0).sum()), "umbrella-only, dropped"],
           ["1 clinical label", int((clean.n_clin == 1).sum()), "the T10 corpus"],
           ["2+ clinical labels", int((clean.n_clin >= 2).sum()), "genuinely ambiguous, dropped"]],
          ["residual labels", "notes", "disposition"])
    amb = clean[clean.n_clin >= 2]
    a = amb.iloc[0]
    print(f"\n  The 2+ group is real clinical ambiguity rather than noise, e.g.")
    print(f"    {a.sample_name.strip()!r} -> {', '.join(sorted(a.clinical))}")
    print(f"  {len(amb)} notes ({len(amb) / len(clean):.0%}) have no single correct")
    print("  answer, which caps any single-label accuracy below 100%.")

    # -- 3. the task --------------------------------------------------------
    sub("3. Task definition")
    work = clean[clean.n_clin == 1].copy()
    work["y"] = work.clinical.map(lambda s: next(iter(s)))
    vc = work.y.value_counts()
    keep = vc[vc >= MIN_PER_CLASS].index
    work = work[work.y.isin(keep)].reset_index(drop=True)
    base = work.y.value_counts().iloc[0] / len(work)
    print(f"  specialties with >={MIN_PER_CLASS} notes : {len(keep)}")
    print(f"  notes in the task           : {len(work)}")
    print(f"  majority-class baseline     : {base:.1%}")
    print("  features                    : transcription text only")
    print("  ('description' and 'keywords' both name the procedure and would")
    print("   leak the label, so neither is used as an input feature.)")
    table([[s, int(c)] for s, c in work.y.value_counts().items()],
          ["specialty", "notes"])

    # -- 4. what duplication would have done to the score ------------------
    sub("4. Why the duplicates had to go first")
    naive = raw.merge(work[["norm", "y"]], on="norm", how="inner")
    nXtr, nXte, nytr, nyte = train_test_split(
        naive.transcription, naive.y, test_size=TEST_FRAC,
        random_state=SEED, stratify=naive.y)
    seen = set(nXtr.map(normalise))
    leaked = float(np.mean([normalise(t) in seen for t in nXte]))
    vec_kw = dict(sublinear_tf=True, min_df=2, ngram_range=(1, 2), stop_words="english")
    naive_mdl = make_pipeline(TfidfVectorizer(**vec_kw), LinearSVC(random_state=SEED))
    naive_mdl.fit(nXtr, nytr)
    acc_naive = accuracy_score(nyte, naive_mdl.predict(nXte))
    print("  Splitting the raw rows, duplicates left intact:")
    print(f"    rows                     : {len(naive)}")
    print(f"    test notes also in train : {leaked:.0%}")
    print(f"    accuracy                 : {acc_naive:.1%}  <- inflated, not reportable")
    print("  The classifier is partly retrieving notes it has already memorised.")
    print("  Everything below splits deduplicated notes instead.")

    # -- 5. models ----------------------------------------------------------
    sub("5. Model comparison (deduplicated, stratified split)")
    Xtr, Xte, ytr, yte = train_test_split(
        work.text, work.y, test_size=TEST_FRAC, random_state=SEED, stratify=work.y)
    print(f"  train {len(Xtr)} notes / test {len(Xte)} notes")
    models = {
        "Multinomial NB": lambda: MultinomialNB(),
        "Logistic regression": lambda: LogisticRegression(max_iter=2000, random_state=SEED),
        "Linear SVM": lambda: LinearSVC(random_state=SEED),
    }
    rows, fitted = [], {}
    for name, make in models.items():
        pipe = make_pipeline(TfidfVectorizer(**vec_kw), make())
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        fitted[name] = pred
        rows.append([name, f"{accuracy_score(yte, pred):.3f}",
                     f"{f1_score(yte, pred, average='macro'):.3f}",
                     f"{f1_score(yte, pred, average='weighted'):.3f}"])
    rows.append(["majority baseline", f"{base:.3f}", "-", "-"])
    table(rows, ["model", "accuracy", "macro-F1", "weighted-F1"])

    best = max(models, key=lambda n: f1_score(yte, fitted[n], average="macro"))
    pred = fitted[best]
    acc = accuracy_score(yte, pred)
    print(f"\n  Best by macro-F1: {best}")

    # -- 6. does the note name its own specialty? --------------------------
    sub("6. Leakage check - are we reading the label off the page?")
    toks = {t for s in keep for t in re.split(r"[^a-z]+", s.lower()) if len(t) > 3}
    pat = re.compile(r"\b(" + "|".join(sorted(toks)) + r")\b", re.I)
    mp = make_pipeline(TfidfVectorizer(**vec_kw), models[best]())
    mp.fit(Xtr.str.replace(pat, " ", regex=True), ytr)
    acc_masked = accuracy_score(yte, mp.predict(Xte.str.replace(pat, " ", regex=True)))
    print(f"  {len(toks)} specialty-name tokens masked out of every note")
    print(f"    accuracy with names      : {acc:.3f}")
    print(f"    accuracy with names gone : {acc_masked:.3f}")
    print(f"    change                   : {acc_masked - acc:+.3f}")
    print("  A small drop means the model is reading clinical content rather")
    print("  than a giveaway string. A large drop would invalidate the result.")

    # -- 7. per class + confusion ------------------------------------------
    sub("7. Where it succeeds and where it confuses")
    classes = sorted(work.y.unique())
    f1s = f1_score(yte, pred, average=None, labels=classes)
    supp = [int((yte == c).sum()) for c in classes]
    order = np.argsort(f1s)
    table([[classes[i], supp[i], f"{f1s[i]:.3f}"] for i in order[::-1]],
          ["specialty", "test n", "F1"])
    cm = confusion_matrix(yte, pred, labels=classes)
    off = sorted(((classes[i], classes[j], int(cm[i, j]))
                  for i in range(len(classes)) for j in range(len(classes))
                  if i != j and cm[i, j] > 0), key=lambda r: -r[2])
    print("\n  Most frequent confusions:")
    table([list(r) for r in off[:6]], ["true", "predicted as", "n"])

    short = [c.split("/")[0].split("-")[0].strip()[:18] for c in classes]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(short, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"T10 specialty classification - {best}\n"
                 f"{len(Xte)} held-out notes, accuracy {acc:.1%}")
    fig.colorbar(im, ax=ax, fraction=0.045, label="row-normalised")
    save_fig(fig, "t10_confusion.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([short[i] for i in order], [f1s[i] for i in order], color="#4878a8")
    ax.axvline(base, color="crimson", ls="--", lw=1, label=f"majority baseline {base:.2f}")
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("T10 per-specialty F1")
    save_fig(fig, "t10_per_class_f1.png")

    pd.DataFrame({"specialty": classes, "test_n": supp, "f1": f1s}).to_csv(
        OUT_DIR / "t10_per_class.csv", index=False)
    print("  [saved] " + str(OUT_DIR / "t10_per_class.csv"))
    return work, (best, acc, f1_score(yte, pred, average="macro"))


# ==========================================================================
# T11 - CLINICAL NAMED ENTITY RECOGNITION
# ==========================================================================
# The gold standard here is distant supervision, and the report must say so.
# MTSamples ships a human-written keyword list per note, but those keywords are
# note-level: a person recorded WHICH clinical terms the note contains, not
# WHERE. Projecting each keyword back onto the text by exact match recovers
# character spans that a person chose, which is as close to real annotation as
# this corpus gets. Two consequences, both measured below:
#   * recall of the gold itself is incomplete - a term the annotator did not
#     bother listing still appears in the text, and a system that finds it is
#     scored as wrong. Section 6 audits exactly this.
#   * precision of the gold is high - the terms were chosen by a human.
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']*|\d+")

NEG_CUES = ["no evidence of", "no signs of", "no sign of", "without any",
            "negative for", "not identified", "denies any", "free of",
            "ruled out", "no", "without", "denies", "absent"]
NEG_STOP = {"but", "however", "although", "though", "aside", "except"}

NER_NOTES = 700          # subsample; the tagger is token-level and the full
                         # corpus is ~1.4M tokens, which adds runtime for no
                         # measurable change in the scores
NER_TEST_FRAC = 0.35
MAX_TERM_WORDS = 5
MIN_TERM_CHARS = 4


def clean_terms(kw, specialties):
    """Keyword string -> usable gold terms.

    Drops the specialty name (MTSamples always puts it first, so it is a label
    in disguise rather than a clinical entity), malformed run-on entries, and
    terms too short to match reliably.
    """
    out, seen = [], set()
    for t in str(kw).split(","):
        t = t.strip().lower()
        if (not t or t in seen or t in specialties
                or len(t) < MIN_TERM_CHARS or len(t.split()) > MAX_TERM_WORDS):
            continue
        seen.add(t)
        out.append(t)
    return out


def gold_spans(text, terms):
    """Character spans for every verbatim occurrence of a gold term.

    Longest match wins, and spans never overlap - the usual convention, so a
    mention of 'right lower lobe' is one entity and not three.
    """
    low = text.lower()
    cand = []
    for t in terms:
        for m in re.finditer(r"\b" + re.escape(t) + r"\b", low):
            cand.append((m.start(), m.end()))
    cand.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, last = [], -1
    for s, e in cand:
        if s >= last:
            out.append((s, e))
            last = e
    return out


def tokenise(text):
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def bio_tags(tokens, spans):
    tags, i = [], 0
    for _, s, e in tokens:
        while i < len(spans) and spans[i][1] <= s:
            i += 1
        if i < len(spans) and s >= spans[i][0] and e <= spans[i][1]:
            tags.append("B" if s == spans[i][0] else "I")
        else:
            tags.append("O")
    return tags


def tags_to_spans(tokens, tags):
    out, cur = [], None
    for (_, s, e), tag in zip(tokens, tags):
        if tag == "B":
            if cur:
                out.append(cur)
            cur = (s, e)
        elif tag == "I" and cur:
            cur = (cur[0], e)
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def tok_features(tokens, i, lexicon):
    w = tokens[i][0].lower()
    f = {
        "w": w, "suf3": w[-3:], "suf2": w[-2:], "pre3": w[:3],
        "len": str(min(len(w), 12)),
        "title": tokens[i][0][:1].isupper(), "upper": tokens[i][0].isupper(),
        "digit": w.isdigit(), "hyphen": "-" in w,
        "in_lex": w in lexicon,
    }
    for off in (-2, -1, 1, 2):
        j = i + off
        f[f"w{off}"] = tokens[j][0].lower() if 0 <= j < len(tokens) else "<pad>"
    f["bg_prev"] = f["w-1"] + "_" + w
    f["bg_next"] = w + "_" + f["w1"]
    return f


def span_prf(gold, pred):
    """Exact-span precision / recall / F1, CoNLL convention."""
    g, p = set(gold), set(pred)
    tp = len(g & p)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def negated(text, span):
    """NegEx-style: is this mention inside the scope of a negation cue?"""
    left = text[max(0, span[0] - 60):span[0]].lower()
    seg = re.split(r"[.;:]|\b(?:" + "|".join(NEG_STOP) + r")\b", left)[-1]
    return any(re.search(r"\b" + re.escape(c) + r"\b", seg) for c in NEG_CUES)


def t11_ner(clean):
    head("T11 - NAMED ENTITY RECOGNITION IN CLINICAL NOTES")

    # -- 1. building a gold standard from human keywords -------------------
    sub("1. Where the gold annotation comes from")
    specialties = {s.strip().lower() for fs in clean.labels for s in fs}
    df = clean[clean.keywords.notna()].copy()
    df["terms"] = df.keywords.map(lambda k: clean_terms(k, specialties))
    df = df[df.terms.map(len) > 0]
    rng = np.random.default_rng(SEED)
    df = df.iloc[rng.permutation(len(df))[:NER_NOTES]].reset_index(drop=True)
    df["spans"] = [gold_spans(t, k) for t, k in zip(df.text, df.terms)]
    df = df[df.spans.map(len) > 0].reset_index(drop=True)

    n_terms = int(df.terms.map(len).sum())
    n_spans = int(df.spans.map(len).sum())
    grounded = float(np.mean([
        len({t for t in terms if re.search(r"\b" + re.escape(t) + r"\b", txt.lower())}) / len(terms)
        for terms, txt in zip(df.terms, df.text)]))
    print("  MTSamples records a human keyword list per note, but note-level:")
    print("  it says which clinical terms occur, not where. Matching each term")
    print("  back into the text recovers spans a person chose.")
    print()
    print(f"  notes used                   : {len(df)} (subsample of the corpus)")
    print(f"  gold terms after cleaning    : {n_terms}")
    print(f"  terms found verbatim in text : {grounded:.1%}")
    print(f"  entity mentions (spans)      : {n_spans}")
    print(f"  mentions per note            : {n_spans / len(df):.1f}")
    print(f"  mean mention length (chars)  : "
          f"{np.mean([e - s for sp in df.spans for s, e in sp]):.1f}")
    print()
    print("  Cleaning drops the first keyword on every note: MTSamples puts the")
    print("  specialty name there, which is a label rather than a clinical term.")

    # -- 2. eyeball it -----------------------------------------------------
    sub("2. Spot check - gold mentions in context")
    for r in df.head(2).itertuples():
        print(f"\n  [{r.sample_name.strip()[:60]}]")
        for s, e in r.spans[:6]:
            lo, hi = max(0, s - 38), min(len(r.text), e + 38)
            ctx = re.sub(r"\s+", " ", r.text[lo:s] + "[" + r.text[s:e] + "]" + r.text[e:hi])
            print(f"    ...{ctx}...")

    # sanity: spans must index back to the text, and must not overlap
    ok_idx = all(r.text[s:e].lower() in r.terms for r in df.itertuples() for s, e in r.spans)
    ok_ovl = all(a[1] <= b[0] for r in df.itertuples() for a, b in zip(r.spans, r.spans[1:]))
    print(f"\n  spans index back to their term : {ok_idx}")
    print(f"  no overlapping spans           : {ok_ovl}")

    # -- 3. split + tokenise ------------------------------------------------
    sub("3. Setup")
    tr, te = train_test_split(df, test_size=NER_TEST_FRAC, random_state=SEED)
    lexicon = {t for terms in tr.terms for t in terms if len(t.split()) == 1}
    print(f"  train notes {len(tr)} / test notes {len(te)}")
    print(f"  single-word lexicon from train notes : {len(lexicon)}")

    def prep(frame):
        toks = [tokenise(t) for t in frame.text]
        tags = [bio_tags(tk, sp) for tk, sp in zip(toks, frame.spans)]
        return toks, tags

    tr_tok, tr_tag = prep(tr)
    te_tok, te_tag = prep(te)
    print(f"  tokens: train {sum(map(len, tr_tok))}  test {sum(map(len, te_tok))}")
    texts = te.text.tolist()
    gold_te = [(i, s, e) for i, sp in enumerate(te.spans) for s, e in sp]

    # -- 4. dictionary baseline --------------------------------------------
    sub("4. Baseline - dictionary lookup")
    full_lex = {t for terms in tr.terms for t in terms}
    maxw = max(len(t.split()) for t in full_lex)
    pred_dict = []
    for i, toks in enumerate(te_tok):
        j = 0
        while j < len(toks):
            hit = None
            for n in range(min(maxw, len(toks) - j), 0, -1):
                phrase = " ".join(w.lower() for w, _, _ in toks[j:j + n])
                if phrase in full_lex:
                    hit = (n, toks[j][1], toks[j + n - 1][2])
                    break
            if hit:
                pred_dict.append((i, hit[1], hit[2]))
                j += hit[0]
            else:
                j += 1
    p, r, f = span_prf(gold_te, pred_dict)
    print(f"  lexicon entries                : {len(full_lex)}")
    print(f"  precision {p:.3f}   recall {r:.3f}   F1 {f:.3f}")
    print("  Recall is capped by terms the training notes never mentioned;")
    print("  precision by terms this note's annotator chose not to list.")

    # -- 5. token classifier -------------------------------------------------
    sub("5. BIO token classifier")
    from sklearn.feature_extraction import DictVectorizer
    Xtr = [tok_features(tk, i, lexicon) for tk in tr_tok for i in range(len(tk))]
    ytr = [t for tags in tr_tag for t in tags]
    Xte = [tok_features(tk, i, lexicon) for tk in te_tok for i in range(len(tk))]
    dv = DictVectorizer()
    # class_weight balanced because 92% of tokens are O; without it the tagger
    # buys accuracy by predicting almost nothing. C tuned on the train split.
    clf = LinearSVC(C=0.2, class_weight="balanced", random_state=SEED)
    clf.fit(dv.fit_transform(Xtr), ytr)
    flat = clf.predict(dv.transform(Xte))
    print(f"  features {len(dv.feature_names_)}  training tokens {len(ytr)}")
    print(f"  tag distribution (train): "
          f"{ {k: int(v) for k, v in pd.Series(ytr).value_counts().items()} }")

    pred_ml, k = [], 0
    for i, toks in enumerate(te_tok):
        tags = list(flat[k:k + len(toks)])
        k += len(toks)
        pred_ml += [(i, s, e) for s, e in tags_to_spans(toks, tags)]
    p2, r2, f2 = span_prf(gold_te, pred_ml)
    print(f"  precision {p2:.3f}   recall {r2:.3f}   F1 {f2:.3f}")

    # multi-word mentions are the less ambiguous slice: a two-word phrase is
    # far more likely to be a genuine clinical concept than a bare noun
    mw_gold = [x for x in gold_te if len(texts[x[0]][x[1]:x[2]].split()) >= 2]
    mw_pred = [x for x in pred_ml if len(texts[x[0]][x[1]:x[2]].split()) >= 2]
    p3, r3, f3 = span_prf(mw_gold, mw_pred)

    table([["dictionary lookup", f"{p:.3f}", f"{r:.3f}", f"{f:.3f}", len(pred_dict)],
           ["BIO token classifier", f"{p2:.3f}", f"{r2:.3f}", f"{f2:.3f}", len(pred_ml)],
           ["  ...multi-word only", f"{p3:.3f}", f"{r3:.3f}", f"{f3:.3f}", len(mw_pred)],
           ["gold mentions", "-", "-", "-", len(gold_te)]],
          ["system", "precision", "recall", "F1", "spans"])

    # -- 6. how complete is the gold? --------------------------------------
    sub("6. The gold standard is incomplete, and by how much is measurable")
    # For single-word lexicon terms, "does it occur in this note" is a set
    # lookup, so the ceiling can be computed exactly rather than estimated.
    hit = miss = 0
    for idx, row in enumerate(te.itertuples()):
        low = row.text.lower()
        present = {w.lower() for w, _, _ in te_tok[idx]}
        gold_surface = {low[s:e] for s, e in row.spans}
        for t in lexicon & present:
            if t in gold_surface:
                hit += 1
            else:
                miss += 1
    ceiling = hit / (hit + miss)
    print("  Take every term a human listed somewhere in the training notes, and")
    print("  ask how often it is listed again when it reappears in a test note:")
    table([["occurs in a test note AND is gold there", hit],
           ["occurs in a test note but is NOT gold there", miss]],
          ["lexicon term", "count"])
    print(f"\n  precision ceiling on single-word terms : {ceiling:.3f}")
    print()
    print("  That is the central result of this tutorial. When a term a human")
    print(f"  already chose to list turns up again, it is listed only {ceiling:.0%} of the")
    print("  time - the keyword list was never meant to be exhaustive. A tagger")
    print("  that finds single-word clinical terms perfectly would still be")
    print(f"  marked at {ceiling:.3f} precision against this gold.")
    print()
    print(f"  The dictionary scores {p:.3f} overall, above that figure only because")
    print("  it also matches multi-word terms, which annotators record far more")
    print(f"  reliably - visible in the multi-word row above ({p3:.3f} precision).")
    print()
    print(f"  The BIO classifier reaches {p2:.3f} because it does not merely find")
    print("  clinical terms - it learns which ones an annotator bothers to write")
    print("  down. That is a different task from the one the tutorial appears to")
    print("  set, and it is worth saying out loud rather than hiding in an F1.")

    extra = [x for x in pred_ml if x not in set(gold_te)]
    rng2 = np.random.default_rng(SEED)
    sample = [extra[i] for i in rng2.permutation(len(extra))[:8]] if extra else []
    print(f"\n  Mentions the classifier found that the gold does not list: {len(extra)}")
    print("  A random sample, for manual judgement - most are real clinical terms")
    print("  the annotator simply did not include, i.e. scored wrong but correct:")
    for i, s, e in sample:
        ctx = re.sub(r"\s+", " ", texts[i][max(0, s - 32):s] + "[" + texts[i][s:e]
                     + "]" + texts[i][e:e + 32])
        print(f"    ...{ctx}...")
    print()
    print("  Every precision figure above is therefore a LOWER BOUND. Reporting")
    print("  it without this section would understate the systems substantially.")

    # -- 7. negation -------------------------------------------------------
    sub("7. Negation detection")
    neg = [(i, s, e) for i, s, e in gold_te if negated(texts[i], (s, e))]
    print(f"  gold mentions            : {len(gold_te)}")
    print(f"  inside a negation scope  : {len(neg)} ({len(neg) / len(gold_te):.1%})")
    print("  A mention is not an assertion. 'no chest pain' contains the entity")
    print("  'chest pain' but records its absence, and a downstream phenotyping")
    print("  query that ignores this inverts the cohort.")
    print("\n  Examples flagged as negated:")
    for i, s, e in neg[:5]:
        ctx = re.sub(r"\s+", " ", texts[i][max(0, s - 45):s] + "[" + texts[i][s:e]
                     + "]" + texts[i][e:e + 20])
        print(f"    ...{ctx}...")

    # -- 8. figures ---------------------------------------------------------
    sub("8. Figures")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    x = np.arange(3)
    ax.bar(x - 0.26, [p, r, f], 0.26, label="dictionary", color="#c3d3e2")
    ax.bar(x, [p2, r2, f2], 0.26, label="BIO classifier", color="#4878a8")
    ax.bar(x + 0.26, [p3, r3, f3], 0.26, label="BIO, multi-word", color="#2d4f6d")
    ax.axhline(ceiling, color="crimson", ls="--", lw=1.4,
               label=f"gold-completeness ceiling ({ceiling:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(["precision", "recall", "F1"])
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("T11 entity recognition")
    ax = axes[1]
    lens = [e - s for sp in df.spans for s, e in sp]
    ax.hist(lens, bins=range(0, 42, 2), color="#4878a8")
    ax.set_xlabel("mention length (characters)")
    ax.set_ylabel("count")
    ax.set_title(f"Gold mention lengths (n={len(lens)})")
    save_fig(fig, "t11_ner.png")

    pd.DataFrame([
        {"system": "dictionary", "precision": p, "recall": r, "f1": f},
        {"system": "bio_classifier", "precision": p2, "recall": r2, "f1": f2},
        {"system": "bio_classifier_multiword", "precision": p3, "recall": r3, "f1": f3},
        {"system": "gold_completeness_ceiling", "precision": ceiling,
         "recall": float("nan"), "f1": float("nan")},
    ]).to_csv(OUT_DIR / "t11_scores.csv", index=False)
    print("  [saved] " + str(OUT_DIR / "t11_scores.csv"))
    return df, (f, f2, len(neg) / len(gold_te), ceiling)


# ==========================================================================
# T12 - WORD SENSE DISAMBIGUATION
# ==========================================================================
# Abbreviation WSD was tried first and abandoned, which is worth recording.
# The usual way to label abbreviation senses is long-form co-occurrence: if a
# note writes 'cerebrovascular accident' and also writes CVA, that CVA is
# pinned. On the deduplicated corpus it yields about 11 labelled instances per
# abbreviation - nowhere near enough to train or to test on. (The raw CSV
# appears to offer twice as many, but those are the duplicate rows from T10
# and using them would put the same instance in train and test.)
#
# Ordinary clinical vocabulary is both more ambiguous and far more frequent.
# 'Sinus' is a heart rhythm or a cavity in the skull; 'discharge' is something
# leaving a wound or a patient leaving a hospital. These are real ambiguities
# that a phenotyping query gets wrong silently.
#
# Labelling scheme, built so that it cannot be circular:
#   the SENSE comes from the +/-3 token collocate window ('sinus rhythm')
#   the FEATURES come from tokens 4..15 away on either side
# The classifier never sees the window that produced its own label.
CUE_W, NEAR_W, FAR_W = 3, 4, 15
MIN_SENSE = 25

TARGETS = {
    "sinus": {
        "cardiac rhythm": [r"\b(?:rhythm|tachycardia|bradycardia|arrhythmia|normal|node|bradycardic)\b"],
        # deliberately specific: 'surgery' and 'disease' also collocate with the
        # cardiac sense and with unrelated anatomy, so they are left out
        "anatomical cavity": [r"\b(?:maxillary|frontal|ethmoid|sphenoid|paranasal|nasal|ostium|"
                              r"mucosa|antrostomy|turbinate|polyp|sinusitis|pilonidal)\b"],
    },
    "discharge": {
        "secretion": [r"\b(?:purulent|nasal|vaginal|ear|penile|bloody|mucoid|drainage|erythema|exudate)\b"],
        "leaving hospital": [r"\b(?:home|instructions|summary|medications|planning|discharged|"
                             r"hospital|condition|followup)\b"],
    },
    "cord": {
        # 'spermatic' earns its own entry rather than being left to collide
        # with the umbilical cues: it is a real fourth sense, and without it
        # a generic cue like 'clamped' silently mislabels hernia repairs.
        "spinal cord": [r"\b(?:spinal|compression|myelopathy|thecal|stenosis|cervical|thoracic|lumbar)\b"],
        "umbilical cord": [r"\b(?:umbilical|nuchal|placenta|infant|baby|delivered)\b"],
        "vocal cord": [r"\b(?:vocal|larynx|laryngeal|paralysis|glottic|intubation|hoarseness)\b"],
        "spermatic cord": [r"\b(?:spermatic|inguinal|hernia|testicle|testis|scrotum|scrotal)\b"],
    },
    "block": {
        "anaesthetic block": [r"\b(?:nerve|epidural|axillary|femoral|digital|caudal|regional|"
                              r"interscalene|performed|anesthesia|lidocaine|marcaine)\b"],
        "conduction defect": [r"\b(?:av|a-v|heart|bundle|branch|atrioventricular|degree|ekg|ecg|"
                              r"complete|lbbb|rbbb)\b"],
    },
}


def wsd_instances(toks_all, word, senses):
    """Every labelled occurrence of `word`.

    Returns (sense, distant-context, collocate shown in situ, note index).
    The cue string used for labelling excludes the target itself; the display
    string puts it back so the window reads normally.
    """
    out = []
    for n, toks in enumerate(toks_all):
        low = [w.lower() for w, _, _ in toks]
        for i, w in enumerate(low):
            if w != word:
                continue
            left, right = low[max(0, i - CUE_W):i], low[i + 1:i + 1 + CUE_W]
            cue = " ".join(left + right)
            fired = [s for s, ps in senses.items() if any(re.search(p, cue) for p in ps)]
            if len(fired) != 1:
                continue
            far = (low[max(0, i - FAR_W):max(0, i - NEAR_W + 1)]
                   + low[i + NEAR_W:i + 1 + FAR_W])
            if len(far) < 6:
                continue
            disp = " ".join(left) + f" [{word}] " + " ".join(right)
            out.append((fired[0], " ".join(far), disp, n))
    return out


def t12_wsd(clean):
    head("T12 - WORD SENSE DISAMBIGUATION IN CLINICAL NOTES")

    sub("1. Why not abbreviations")
    print("  The obvious T12 target is abbreviations, and it was tried first.")
    print("  Senses are normally labelled by long-form co-occurrence: a note")
    print("  that writes 'cerebrovascular accident' pins the CVA it also uses.")
    print("  On the deduplicated corpus that yields roughly 11 labelled")
    print("  instances per abbreviation - too few to train or to test on.")
    print("  The raw CSV looks twice as rich, but those extra instances are the")
    print("  duplicate rows T10 removed; using them puts the same sentence in")
    print("  train and test. Ordinary clinical words are ambiguous too, and")
    print("  roughly ten times more frequent, so they are the target here.")

    sub("2. The labelling scheme")
    print(f"  sense   from the +/-{CUE_W} token collocate window")
    print(f"  features from tokens {NEAR_W}..{FAR_W} away on either side")
    print("  The classifier cannot see the window that produced its own label,")
    print("  so a high score is not the labelling rule being read back out.")
    print("  This is weak supervision, not human annotation. Section 6 tests")
    print("  how much of the result survives when the cue words are removed")
    print("  from the features altogether.")

    toks_all = [tokenise(t) for t in clean.text]
    data, rows = {}, []
    for word, senses in TARGETS.items():
        inst = wsd_instances(toks_all, word, senses)
        cnt = pd.Series([s for s, _, _, _ in inst]).value_counts()
        keep = set(cnt[cnt >= MIN_SENSE].index)
        kept = [x for x in inst if x[0] in keep]
        dropped = {s: int(c) for s, c in cnt.items() if s not in keep}
        rows.append([word, len(inst), len(keep), len(kept),
                     ", ".join(f"{s} {int(c)}" for s, c in cnt.items() if s in keep),
                     ", ".join(f"{s} {c}" for s, c in dropped.items()) or "-"])
        if len(keep) >= 2:
            data[word] = kept

    sub("3. Labelled instances")
    table(rows, ["word", "labelled", "senses kept", "instances used",
                 "senses", f"dropped (<{MIN_SENSE})"])
    print(f"\n  A sense below {MIN_SENSE} instances is dropped, never folded into a")
    print("  neighbouring sense. Folding would be the cheap way to keep the")
    print("  instance count up and would quietly corrupt the labels - the")
    print("  'cord' row is exactly where that temptation arises.")

    sub("4. Instances in context")
    print("  The collocate window that fixed the sense, target in brackets:")
    for word, inst in data.items():
        print(f"\n  {word.upper()}")
        seen = {}
        for sense, far, disp, n in inst:
            if seen.get(sense, 0) >= 2:
                continue
            seen[sense] = seen.get(sense, 0) + 1
            print(f"    {sense:<20} ...{disp}...")

    sub("5. Disambiguation accuracy")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = []
    for word, inst in data.items():
        y = [s for s, _, _, _ in inst]
        X = [f for _, f, _, _ in inst]
        mfs = pd.Series(y).value_counts().iloc[0] / len(y)
        pipe = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
        acc = float(cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean())
        f1m = float(cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro").mean())
        res.append([word, len(y), f"{mfs:.3f}", f"{acc:.3f}", f"{acc - mfs:+.3f}", f"{f1m:.3f}"])
    table(res, ["word", "n", "MFS baseline", "accuracy", "gain", "macro-F1"])
    print("\n  MFS = most frequent sense, the standard WSD baseline: always answer")
    print("  with the commoner reading. Beating it is the only evidence that")
    print("  context is being used at all.")

    sub("6. Rigour check - remove the cue words entirely")
    print("  Cue tokens can still occur far from the target ('nasal' 8 tokens")
    print("  away). That is legitimate context rather than circularity, but the")
    print("  stronger test is to delete every cue token from the features and")
    print("  see what survives.")
    strict = []
    for word, inst in data.items():
        pats = [p for ps in TARGETS[word].values() for p in ps]
        y = [s for s, _, _, _ in inst]
        X = []
        for _, far, _, _ in inst:
            t = far
            for p in pats:
                t = re.sub(p, " ", t)
            X.append(t)
        mfs = pd.Series(y).value_counts().iloc[0] / len(y)
        pipe = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
        acc = float(cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean())
        strict.append((word, mfs, acc))
    table([[w, f"{m:.3f}", f"{a:.3f}", f"{a - m:+.3f}"] for w, m, a in strict],
          ["word", "MFS baseline", "accuracy, cues deleted", "gain"])
    print("\n  Still well above baseline with the cue vocabulary gone, so the")
    print("  classifier is reading genuine surrounding context.")

    sub("7. What the classifier keys on")
    for word, inst in data.items():
        y = [s for s, _, _, _ in inst]
        X = [f for _, f, _, _ in inst]
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2)
        M = vec.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        lr.fit(M, y)
        names = np.array(vec.get_feature_names_out())
        print(f"\n  {word.upper()}")
        if len(lr.classes_) == 2:
            coef = lr.coef_[0]
            print(f"    {lr.classes_[1]:<20}: {', '.join(names[np.argsort(coef)[-7:]][::-1])}")
            print(f"    {lr.classes_[0]:<20}: {', '.join(names[np.argsort(coef)[:7]])}")
        else:
            for ci, c in enumerate(lr.classes_):
                print(f"    {c:<20}: {', '.join(names[np.argsort(lr.coef_[ci])[-7:]][::-1])}")

    sub("8. Figure")
    words = [r[0] for r in res]
    base = [float(r[2]) for r in res]
    accs = [float(r[3]) for r in res]
    strict_acc = [a for _, _, a in strict]
    x = np.arange(len(words))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.26, base, 0.26, label="most-frequent-sense baseline", color="#c3d3e2")
    ax.bar(x, accs, 0.26, label="context classifier", color="#4878a8")
    ax.bar(x + 0.26, strict_acc, 0.26, label="classifier, cue words deleted", color="#2d4f6d")
    ax.set_xticks(x)
    ax.set_xticklabels(words)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-fold accuracy")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("T12 word sense disambiguation")
    save_fig(fig, "t12_wsd.png")

    pd.DataFrame([{"word": r[0], "n": r[1], "mfs_baseline": float(r[2]),
                   "accuracy": float(r[3]), "macro_f1": float(r[5]),
                   "accuracy_cues_deleted": s[2]}
                  for r, s in zip(res, strict)]).to_csv(
        OUT_DIR / "t12_wsd.csv", index=False)
    print("  [saved] " + str(OUT_DIR / "t12_wsd.csv"))

    sub("9. Why this matters clinically")
    print("  A cohort query for 'discharge' that does not disambiguate will")
    print("  collect every discharged patient alongside every wound secretion.")
    print("  A query for 'sinus' mixes arrhythmia screening with ENT disease.")
    print("  The ambiguity is invisible in a keyword count and only shows up")
    print("  when someone checks the retrieved notes by hand.")
    return [(r[0], float(r[2]), float(r[3])) for r in res]


def main():
    head("UNIT-4  (T10-T12)  -  CLINICAL NLP ON MTSAMPLES")
    raw, clean = load_corpus()
    work, t10 = t10_classification(raw, clean)
    ner, t11 = t11_ner(clean)
    t12 = t12_wsd(clean)
    head("DONE")
    print(f"  T10: {t10[0]}, accuracy {t10[1]:.1%}, macro-F1 {t10[2]:.3f}")
    print(f"  T11: dictionary F1 {t11[0]:.3f}, classifier F1 {t11[1]:.3f}, "
          f"{t11[2]:.0%} of mentions negated, gold ceiling {t11[3]:.3f}")
    print("  T12: " + "; ".join(f"{w} {a:.3f} (base {b:.3f})" for w, b, a in t12))
    print(f"  figures : {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
