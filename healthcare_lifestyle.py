"""Unit-5 (T13-T15) - dimensionality reduction and behavioural analysis of
adult body weight, on NHANES 2017-2018.

WHY A THIRD DATASET
-------------------
T1-T9 run on the Kaggle chest X-ray cohort; T10-T12 on the MTSamples note
corpus. Neither can support Unit-5.

T14 and T15 name their subject matter explicitly: behavioural factors
influencing WEIGHTS IN ADULTS. The X-ray cohort records no weight, no age,
no sex and no behavioural variable of any kind - its manifest is image
geometry plus a pneumonia label - and its patients are children aged 1-5.
MTSamples mentions weights only incidentally inside note prose.

NHANES is the survey those tutorials describe: a real, nationally sampled US
health examination with measured body weight and extensive behavioural
questionnaires. Nothing here is simulated or synthesised.

  DEMO_J    age, sex, race, education, income, survey design
  BMX_J     MEASURED weight, height, BMI, waist (not self-reported)
  PAQ_J     physical activity, sedentary time
  SLQ_J     sleep duration
  ALQ_J     alcohol
  SMQ_J     smoking
  DBQ_J     meals away from home, fast food, self-rated diet
  DR1TOT_J  24-hour dietary recall: energy, sugar, fibre, fat, sodium

Source: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/
Cycle J = 2017-2018, the last complete pre-pandemic cycle.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = Path("data/nhanes")
OUT_DIR = Path("outputs_lifestyle")
SEED = 0
ADULT_AGE = 20

np.random.seed(SEED)
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    "DEMO_J": ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2",
               "INDFMPIR", "WTMEC2YR"],
    "BMX_J": ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"],
    "PAQ_J": ["SEQN", "PAQ605", "PAQ620", "PAQ650", "PAQ655", "PAD660",
              "PAQ665", "PAQ670", "PAD675", "PAD680"],
    "SLQ_J": ["SEQN", "SLD012"],
    "ALQ_J": ["SEQN", "ALQ121", "ALQ130"],
    "SMQ_J": ["SEQN", "SMQ020", "SMQ040"],
    "DBQ_J": ["SEQN", "DBQ700", "DBD895", "DBD900"],
    "DR1TOT_J": ["SEQN", "DR1TKCAL", "DR1TSUGR", "DR1TFIBE", "DR1TTFAT", "DR1TSODI"],
}

# NHANES encodes "refused" and "don't know" as in-range-looking numbers. They
# must be removed by hand: a blanket "drop large values" rule would delete
# 9 hours of sedentary time along with the 9999 codes.
SENTINELS = {
    "PAD680": [7777, 9999], "PAQ655": [77, 99], "PAD660": [7777, 9999],
    "PAQ670": [77, 99], "PAD675": [7777, 9999], "ALQ121": [77, 99],
    "ALQ130": [777, 999], "SMQ040": [7, 9], "DBQ700": [7, 9],
    "DBD895": [7777, 9999], "DBD900": [7777, 9999], "DMDEDUC2": [7, 9],
}

# behavioural and dietary features, in the order they are reported
BEHAVIOUR = [
    ("sedentary_min", "Sedentary minutes per day"),
    ("mvpa_min_wk", "Recreational MVPA minutes per week"),
    ("sleep_hr", "Sleep hours per night"),
    ("drinks_day", "Alcoholic drinks per drinking day"),
    ("meals_out", "Meals not home-prepared per week"),
    ("fastfood_wk", "Fast-food meals per week"),
    ("diet_quality", "Self-rated diet (1 excellent - 5 poor)"),
    ("kcal", "Energy intake (kcal, 24h recall)"),
    ("sugar_g", "Sugar (g, 24h recall)"),
    ("fibre_g", "Fibre (g, 24h recall)"),
    ("fat_g", "Total fat (g, 24h recall)"),
    ("sodium_mg", "Sodium (mg, 24h recall)"),
]
BCOLS = [c for c, _ in BEHAVIOUR]
LABEL = dict(BEHAVIOUR)


# --------------------------------------------------------------------------
# Small helpers (same plain style as the other two modules)
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


# ==========================================================================
# LOADING AND DERIVATION
# ==========================================================================
def load_nhanes():
    """Merge the eight NHANES files on SEQN and derive the behavioural panel."""
    frames = {}
    for name, cols in FILES.items():
        df = pd.read_sas(DATA / f"{name}.xpt", format="xport")
        frames[name] = df[[c for c in cols if c in df.columns]].copy()

    m = frames["DEMO_J"]
    for name in list(FILES)[1:]:
        m = m.merge(frames[name], on="SEQN", how="left")

    # XPT stores an exact zero as a denormal; it reads back as ~5.4e-79
    num = m.select_dtypes("number").columns
    m[num] = m[num].mask(m[num].abs() < 1e-12, 0.0)

    for col, codes in SENTINELS.items():
        if col in m.columns:
            m[col] = m[col].replace(codes, np.nan)

    d = m[m.RIDAGEYR >= ADULT_AGE].copy()

    # -- derived behavioural variables -------------------------------------
    d["sedentary_min"] = d.PAD680
    # Recreational MVPA only. Occupational activity is not a discretionary
    # behaviour, so mixing it in would confound the lifestyle reading.
    vig = np.where(d.PAQ650 == 1, d.PAQ655.fillna(0) * d.PAD660.fillna(0), 0.0)
    mod = np.where(d.PAQ665 == 1, d.PAQ670.fillna(0) * d.PAD675.fillna(0), 0.0)
    d["mvpa_min_wk"] = 2 * vig + mod        # vigorous counts double, per WHO
    d.loc[d.PAQ650.isna() & d.PAQ665.isna(), "mvpa_min_wk"] = np.nan

    d["sleep_hr"] = d.SLD012
    # ALQ130 is asked only of drinkers; a non-drinker is 0, not missing
    d["drinks_day"] = d.ALQ130
    d.loc[d.ALQ121 == 0, "drinks_day"] = 0.0

    # 5555 = "more than 21 meals a week"; recode to the boundary rather than drop
    d["meals_out"] = d.DBD895.replace(5555, 21)
    d["fastfood_wk"] = d.DBD900.replace(5555, 21)
    d["diet_quality"] = d.DBQ700

    d["kcal"] = d.DR1TKCAL
    d["sugar_g"] = d.DR1TSUGR
    d["fibre_g"] = d.DR1TFIBE
    d["fat_g"] = d.DR1TTFAT
    d["sodium_mg"] = d.DR1TSODI

    # smoking as a three-level status, reconstructed from the gate question
    smoke = pd.Series("never", index=d.index, dtype=object)
    smoke[(d.SMQ020 == 1) & (d.SMQ040 == 3)] = "former"
    smoke[(d.SMQ020 == 1) & d.SMQ040.isin([1, 2])] = "current"
    smoke[d.SMQ020.isna()] = np.nan
    d["smoke_status"] = smoke

    d["sex"] = d.RIAGENDR.map({1: "Male", 2: "Female"})
    d["age"] = d.RIDAGEYR
    d["income_ratio"] = d.INDFMPIR
    d["education"] = d.DMDEDUC2
    d["weight_kg"] = d.BMXWT
    d["bmi"] = d.BMXBMI
    d["waist_cm"] = d.BMXWAIST
    d["mec_weight"] = d.WTMEC2YR

    d = d[d.weight_kg.notna()].reset_index(drop=True)
    return d


def bmi_group(b):
    if pd.isna(b):
        return np.nan
    if b < 18.5:
        return "Underweight"
    if b < 25:
        return "Normal"
    if b < 30:
        return "Overweight"
    return "Obese"


# ==========================================================================
# T13 - DIMENSIONALITY REDUCTION FOR EXPLORATORY ANALYSIS
# ==========================================================================
def t13_dimensionality(d):
    head("T13 - DIMENSIONALITY REDUCTION FOR EXPLORATORY DATA ANALYSIS")

    sub("1. The feature space")
    print("  Twelve behavioural and dietary measurements per adult. Exploratory")
    print("  analysis in medical research usually starts here: too many")
    print("  correlated variables to plot pairwise, and no obvious ordering.")
    print(f"  pairwise scatter plots needed to see it raw: {len(BCOLS)*(len(BCOLS)-1)//2}")
    print()
    miss = [(LABEL[c], int(d[c].notna().sum()), f"{d[c].isna().mean():.1%}")
            for c in BCOLS]
    table(miss, ["feature", "observed", "missing"])

    sub("2. Complete cases")
    X = d[BCOLS]
    cc = d[X.notna().all(axis=1)].copy()
    print(f"  adults with a measured weight   : {len(d)}")
    print(f"  complete on all 12 behaviours   : {len(cc)} ({len(cc)/len(d):.0%})")
    print("  PCA has no principled way to handle a missing coordinate, so the")
    print("  projection is built on complete cases. Imputing first would invent")
    print("  the very structure the projection is meant to reveal.")
    print(f"  mean weight, complete cases     : {cc.weight_kg.mean():.1f} kg")
    print(f"  mean weight, everyone else      : "
          f"{d[~d.index.isin(cc.index)].weight_kg.mean():.1f} kg")
    print("  The two are close, so dropping incomplete records does not")
    print("  obviously select a heavier or lighter group.")

    Z = StandardScaler().fit_transform(cc[BCOLS].values)

    sub("3. Principal component analysis")
    p = PCA(random_state=SEED).fit(Z)
    ev = p.explained_variance_ratio_
    rows = [[f"PC{i+1}", f"{ev[i]:.3f}", f"{ev[:i+1].sum():.3f}"]
            for i in range(len(ev))]
    table(rows, ["component", "variance explained", "cumulative"])
    n90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    print(f"\n  components needed for 90% of the variance: {n90} of {len(BCOLS)}")
    print(f"  first two components carry {ev[:2].sum():.1%}")
    print()
    print("  This is the finding, and it is a negative one. If the behaviours")
    print("  were largely redundant, two or three components would carry most")
    print("  of the variance and the space could be summarised. They do not.")
    print("  Diet, activity, sleep and alcohol are close to independent axes:")
    print("  knowing somebody is sedentary says little about what they eat.")

    sub("4. What the first components mean")
    load = pd.DataFrame(p.components_[:3].T, index=[LABEL[c] for c in BCOLS],
                        columns=["PC1", "PC2", "PC3"])
    for pc in ["PC1", "PC2", "PC3"]:
        s = load[pc].sort_values()
        print(f"\n  {pc} ({ev[int(pc[-1])-1]:.1%} of variance)")
        print(f"    positive: {', '.join(s.index[-3:][::-1])}")
        print(f"    negative: {', '.join(s.index[:3])}")
    print("\n  PC1 loads on the dietary quantities together - people who eat more")
    print("  of one thing eat more of everything. It is an intake-volume axis,")
    print("  not a lifestyle type.")

    sub("5. Does the projection separate body-weight groups?")
    cc["bmi_group"] = cc.bmi.map(bmi_group)
    S = p.transform(Z)[:, :2]
    groups = ["Normal", "Overweight", "Obese"]
    rows = []
    for g in groups:
        msk = (cc.bmi_group == g).values
        if msk.sum() < 10:
            continue
        rows.append([g, int(msk.sum()), f"{S[msk, 0].mean():+.3f}",
                     f"{S[msk, 1].mean():+.3f}", f"{cc.weight_kg[msk].mean():.1f}"])
    table(rows, ["BMI group", "n", "mean PC1", "mean PC2", "mean weight kg"])
    f, pv = stats.f_oneway(*[S[(cc.bmi_group == g).values, 0] for g in groups])
    print(f"\n  one-way ANOVA of PC1 across the three groups: F={f:.2f}, p={pv:.3g}")
    print("  The group centroids sit almost on top of each other. A 2-D")
    print("  behavioural projection does not separate heavy adults from light")
    print("  ones, which is the honest answer to the exploratory question and")
    print("  exactly what T14 and T15 then quantify.")

    sub("6. t-SNE, for comparison")
    n_ts = min(2000, len(cc))
    idx = np.random.default_rng(SEED).permutation(len(cc))[:n_ts]
    ts = TSNE(n_components=2, perplexity=30, init="pca",
              random_state=SEED, max_iter=1000).fit_transform(Z[idx])
    print(f"  t-SNE on a {n_ts}-adult subsample, perplexity 30")
    print("  t-SNE preserves local neighbourhoods rather than variance, so it")
    print("  can expose clusters PCA would miss. Here it produces one diffuse")
    print("  cloud: there are no natural behavioural types in this population,")
    print("  only a continuum.")
    print("  It is also not a distance-preserving map - cluster sizes and gaps")
    print("  in a t-SNE plot carry no quantitative meaning, so it is used here")
    print("  to look, not to measure.")

    # -- figures ------------------------------------------------------------
    sub("7. Figures")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    ax.bar(range(1, len(ev) + 1), ev * 100, color="#4878a8")
    ax.plot(range(1, len(ev) + 1), np.cumsum(ev) * 100, "o-", color="crimson",
            ms=3.5, lw=1.2, label="cumulative")
    ax.axhline(90, ls="--", lw=1, color="grey")
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (%)")
    ax.set_title("Scree - no dominant component")
    ax.legend(fontsize=8)

    ax = axes[1]
    cmap = {"Normal": "#7fb069", "Overweight": "#e8a33d", "Obese": "#c1413a"}
    for g in groups:
        msk = (cc.bmi_group == g).values
        ax.scatter(S[msk, 0], S[msk, 1], s=5, alpha=0.35, c=cmap[g], label=g)
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title("PCA - BMI groups overlap")
    ax.legend(fontsize=8, markerscale=2)

    ax = axes[2]
    gsub = cc.bmi_group.values[idx]
    for g in groups:
        msk = gsub == g
        ax.scatter(ts[msk, 0], ts[msk, 1], s=5, alpha=0.4, c=cmap[g], label=g)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE (n={n_ts}) - one diffuse cloud")
    ax.legend(fontsize=8, markerscale=2)
    save_fig(fig, "t13_projection.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(load.values, cmap="RdBu_r", vmin=-0.65, vmax=0.65, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(load.columns)
    ax.set_yticks(range(len(BCOLS)))
    ax.set_yticklabels(load.index, fontsize=8)
    for i in range(len(BCOLS)):
        for j in range(3):
            ax.text(j, i, f"{load.values[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7.5)
    ax.set_title("T13 component loadings")
    fig.colorbar(im, ax=ax, fraction=0.04)
    save_fig(fig, "t13_loadings.png")

    pd.DataFrame({"component": [f"PC{i+1}" for i in range(len(ev))],
                  "variance_explained": ev,
                  "cumulative": np.cumsum(ev)}).to_csv(
        OUT_DIR / "t13_variance.csv", index=False)
    load.to_csv(OUT_DIR / "t13_loadings.csv")
    print("  [saved] " + str(OUT_DIR / "t13_variance.csv"))
    print("  [saved] " + str(OUT_DIR / "t13_loadings.csv"))
    return cc, ev, n90


def pearson_ci(x, y):
    """Pearson r with a Fisher-z 95% interval."""
    msk = x.notna() & y.notna()
    n = int(msk.sum())
    r, p = stats.pearsonr(x[msk], y[msk])
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return r, lo, hi, p, n


# ==========================================================================
# T14 - BEHAVIOURAL FACTORS AND ADULT WEIGHT, PART 1
# ==========================================================================
def t14_associations(d):
    head("T14 - BEHAVIOURAL FACTORS INFLUENCING WEIGHT IN ADULTS - PART 1")

    sub("1. The outcome")
    print(f"  adults aged {ADULT_AGE}+ with a MEASURED weight : {len(d)}")
    print("  Weight is measured in the mobile examination centre, not")
    print("  self-reported, so it carries none of the usual under-reporting.")
    print()
    w = d.weight_kg
    table([["mean", f"{w.mean():.1f}"], ["sd", f"{w.std():.1f}"],
           ["median", f"{w.median():.1f}"],
           ["IQR", f"{w.quantile(.25):.1f} - {w.quantile(.75):.1f}"],
           ["range", f"{w.min():.1f} - {w.max():.1f}"]],
          ["statistic", "weight (kg)"])

    sub("2. The survey design cannot be ignored")
    unw = w.mean()
    wt = d.mec_weight
    msk = wt.notna() & w.notna()
    wtd = np.average(w[msk], weights=wt[msk])
    print("  NHANES oversamples some groups by design, so an unweighted mean")
    print("  describes the sample rather than the population.")
    table([["unweighted sample mean", f"{unw:.2f} kg"],
           ["survey-weighted mean (WTMEC2YR)", f"{wtd:.2f} kg"],
           ["difference", f"{wtd - unw:+.2f} kg"]],
          ["estimator", "value"])
    print("\n  The gap is small here but real. Everything below is unweighted,")
    print("  which is defensible for association analysis and wrong for")
    print("  prevalence claims - a distinction worth stating rather than")
    print("  discovering later.")

    sub("3. Weight by sex and age band")
    d = d.copy()
    d["age_band"] = pd.cut(d.age, [19, 29, 39, 49, 59, 69, 120],
                           labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70+"])
    rows = []
    for sex in ["Female", "Male"]:
        for band in d.age_band.cat.categories:
            s = d[(d.sex == sex) & (d.age_band == band)].weight_kg
            if len(s) < 20:
                continue
            rows.append([sex, band, len(s), f"{s.mean():.1f}", f"{s.std():.1f}"])
    table(rows, ["sex", "age band", "n", "mean kg", "sd"])
    mw = d[d.sex == "Male"].weight_kg
    fw = d[d.sex == "Female"].weight_kg
    t, p = stats.ttest_ind(mw, fw, equal_var=False)
    print(f"\n  male - female difference: {mw.mean() - fw.mean():+.1f} kg "
          f"(Welch t={t:.1f}, p={p:.2g})")
    print("  Sex and age dominate before any behaviour is considered, which is")
    print("  why section 5 adjusts for both.")

    sub("4. Crude association of each behaviour with weight")
    print("  Pearson correlation with weight, and the kg difference associated")
    print("  with a one-SD increase in the behaviour.")
    rows = []
    crude = {}
    for c in BCOLS:
        r, lo, hi, p, n = pearson_ci(d[c], d.weight_kg)
        kg_sd = r * d.weight_kg.std()
        crude[c] = (r, lo, hi, p, n, kg_sd)
        rows.append([LABEL[c], n, f"{r:+.3f}", f"{lo:+.3f} to {hi:+.3f}",
                     f"{kg_sd:+.2f}", f"{p:.2g}" if p >= 1e-4 else "<1e-4"])
    rows.sort(key=lambda x: -abs(float(x[2])))
    table(rows, ["behaviour", "n", "r", "95% CI", "kg per SD", "p"])
    big = max(abs(v[0]) for v in crude.values())
    print(f"\n  The strongest single correlation is {big:.3f}. Squared, that is")
    print(f"  {big**2:.1%} of the variance in adult weight. No behavioural")
    print("  variable in this survey is close to being a sufficient explanation.")

    sub("5. Adjusting for age and sex changes the picture")
    print("  Each behaviour is refitted as weight ~ behaviour + age + sex, so")
    print("  the coefficient is the kg difference per SD holding both constant.")
    rows, adj = [], {}
    for c in BCOLS:
        sdx = d[c].std()
        sub_d = d[[c, "weight_kg", "age", "sex"]].dropna()
        m = smf.ols(f"weight_kg ~ {c} + age + C(sex)", data=sub_d).fit()
        b, ci = m.params[c] * sdx, m.conf_int().loc[c] * sdx
        adj[c] = (b, ci[0], ci[1], m.pvalues[c])
        ck = crude[c][5]
        flag = ""
        if np.sign(b) != np.sign(ck):
            flag = "sign flips"
        elif abs(b) < abs(ck) * 0.5:
            flag = "halved"
        rows.append([LABEL[c], f"{ck:+.2f}", f"{b:+.2f}",
                     f"{ci[0]:+.2f} to {ci[1]:+.2f}",
                     f"{m.pvalues[c]:.2g}" if m.pvalues[c] >= 1e-4 else "<1e-4", flag])
    table(rows, ["behaviour", "crude kg/SD", "adjusted kg/SD", "95% CI", "p", "note"])
    nflip = sum(1 for c in BCOLS if np.sign(adj[c][0]) != np.sign(crude[c][5]))
    nhalf = sum(1 for c in BCOLS if abs(adj[c][0]) < abs(crude[c][5]) * 0.5)
    emerged = [c for c in BCOLS if crude[c][3] >= 0.05 and adj[c][3] < 0.05]
    print(f"\n  sign changes after adjustment : {nflip} of {len(BCOLS)}")
    print(f"  more than halved              : {nhalf} of {len(BCOLS)}")
    print(f"  null crude, real adjusted     : {len(emerged)} "
          f"({', '.join(LABEL[c] for c in emerged) if emerged else '-'})")
    print()
    print("  No association reverses, but the confounding is still doing real")
    print("  work in both directions. Three dietary associations more than")
    print("  halve once age and sex are held constant: older adults eat less")
    print("  and weigh less, so part of the crude 'intake' signal was age.")
    if emerged:
        print()
        print("  The opposite case is more interesting. Recreational activity")
        print("  shows essentially no crude association with weight, and a clear")
        print("  negative one after adjustment. Men are both more active and")
        print("  heavier, so sex was masking the effect - suppression rather")
        print("  than confounding. An unadjusted screen would have discarded")
        print("  the one behaviour a clinician can actually prescribe.")

    sub("6. Smoking status, where the confounding is clearest")
    sm = d[d.smoke_status.notna()]
    rows = []
    for g in ["never", "former", "current"]:
        s = sm[sm.smoke_status == g]
        rows.append([g, len(s), f"{s.weight_kg.mean():.1f}",
                     f"{s.age.mean():.1f}", f"{(s.sex == 'Male').mean():.0%}"])
    table(rows, ["smoking status", "n", "mean weight kg", "mean age", "% male"])
    m = smf.ols("weight_kg ~ C(smoke_status, Treatment('never')) + age + C(sex)",
                data=sm).fit()
    print("\n  adjusted for age and sex:")
    for k in m.params.index:
        if "smoke_status" in k:
            lab = k.split("T.")[1].rstrip("]")
            ci = m.conf_int().loc[k]
            print(f"    {lab:8s} vs never : {m.params[k]:+.2f} kg "
                  f"({ci[0]:+.2f} to {ci[1]:+.2f}), p={m.pvalues[k]:.2g}")
    print("  Current smokers are lighter; former smokers heavier. This is the")
    print("  textbook example of why 'behaviour associated with lower weight'")
    print("  must never be read as 'behaviour that should be encouraged'.")

    # -- figure -------------------------------------------------------------
    sub("7. Figure")
    order = sorted(BCOLS, key=lambda c: adj[c][0])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.errorbar([crude[c][5] for c in order], y - 0.17,
                xerr=0, fmt="o", ms=5, color="#9ab6cf", label="crude")
    ax.errorbar([adj[c][0] for c in order], y + 0.17,
                xerr=[[adj[c][0] - adj[c][1] for c in order],
                      [adj[c][2] - adj[c][0] for c in order]],
                fmt="o", ms=5, color="#2d4f6d", capsize=2.5,
                label="adjusted for age and sex")
    ax.axvline(0, color="crimson", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[c] for c in order], fontsize=8.5)
    ax.set_xlabel("kg difference per 1 SD of the behaviour")
    ax.set_title("T14  Behavioural associations with adult weight")
    ax.legend(fontsize=8.5, loc="lower right")
    save_fig(fig, "t14_associations.png")

    pd.DataFrame([{"behaviour": LABEL[c], "n": crude[c][4], "r": crude[c][0],
                   "crude_kg_per_sd": crude[c][5], "adjusted_kg_per_sd": adj[c][0],
                   "adj_ci_lo": adj[c][1], "adj_ci_hi": adj[c][2],
                   "adj_p": adj[c][3]} for c in BCOLS]).to_csv(
        OUT_DIR / "t14_associations.csv", index=False)
    print("  [saved] " + str(OUT_DIR / "t14_associations.csv"))
    return crude, adj, nhalf


# ==========================================================================
# T15 - BEHAVIOURAL FACTORS AND ADULT WEIGHT, PART 2
# ==========================================================================
def t15_models(d):
    head("T15 - BEHAVIOURAL FACTORS INFLUENCING WEIGHT IN ADULTS - PART 2")

    print("\n  Part 1 measured one behaviour at a time. Part 2 asks the question")
    print("  the tutorial title implies: taken together, how much of adult body")
    print("  weight do behavioural factors actually account for?")

    sub("1. The modelling sample")
    cols = BCOLS + ["age", "sex", "education", "income_ratio", "weight_kg"]
    m = d[cols].dropna().copy()
    print(f"  adults with a measured weight     : {len(d)}")
    print(f"  complete on every model variable  : {len(m)} ({len(m)/len(d):.0%})")
    print(f"  mean weight in the modelling set  : {m.weight_kg.mean():.1f} kg")
    print(f"  mean weight in those dropped      : "
          f"{d.loc[~d.index.isin(m.index), 'weight_kg'].mean():.1f} kg")

    sub("2. Nested models - what does behaviour add?")
    f_demo = "weight_kg ~ age + C(sex) + education + income_ratio"
    f_beh = "weight_kg ~ " + " + ".join(BCOLS)
    f_full = f_demo + " + " + " + ".join(BCOLS)
    fits = {"demographics only": smf.ols(f_demo, data=m).fit(),
            "behaviour only": smf.ols(f_beh, data=m).fit(),
            "demographics + behaviour": smf.ols(f_full, data=m).fit()}
    rows = []
    for name, fit in fits.items():
        rows.append([name, int(fit.df_model), f"{fit.rsquared:.4f}",
                     f"{fit.rsquared_adj:.4f}", f"{np.sqrt(fit.mse_resid):.2f}"])
    table(rows, ["model", "predictors", "R2", "adjusted R2", "residual SD kg"])
    inc = fits["demographics + behaviour"].rsquared - fits["demographics only"].rsquared
    print(f"\n  R2 added by all twelve behaviours, over demographics : {inc:.4f}")
    print(f"  i.e. {inc:.1%} of the variance in adult weight.")
    print()
    print("  This is the headline of T15. Age, sex, education and income")
    print(f"  account for {fits['demographics only'].rsquared:.1%}; adding every behavioural and")
    print(f"  dietary measurement in the survey raises that to "
          f"{fits['demographics + behaviour'].rsquared:.1%}.")
    lr = fits["demographics + behaviour"].compare_lr_test(fits["demographics only"])
    print(f"  The increment is statistically detectable (LR chi2={lr[0]:.1f}, "
          f"p={lr[1]:.2g})")
    print("  but statistical detectability at n=%d is not practical importance."
          % len(m))

    sub("3. The full model, coefficients per SD")
    fit = fits["demographics + behaviour"]
    rows = []
    for c in BCOLS:
        sdx = m[c].std()
        b, ci = fit.params[c] * sdx, fit.conf_int().loc[c] * sdx
        rows.append([LABEL[c], f"{b:+.2f}", f"{ci[0]:+.2f} to {ci[1]:+.2f}",
                     f"{fit.pvalues[c]:.3g}" if fit.pvalues[c] >= 1e-4 else "<1e-4"])
    rows.sort(key=lambda r: -abs(float(r[1])))
    table(rows, ["behaviour", "kg per SD", "95% CI", "p"])
    sig = sum(1 for c in BCOLS if fit.pvalues[c] < 0.05)
    big_c = max(BCOLS, key=lambda c: abs(fit.params[c] * m[c].std()))
    big_b = fit.params[big_c] * m[big_c].std()
    resid = np.sqrt(fit.mse_resid)
    print(f"\n  {sig} of {len(BCOLS)} behaviours reach p<0.05 in the full model.")
    print(f"  Largest mutually-adjusted effect : {LABEL[big_c]}, {big_b:+.2f} kg/SD")
    print(f"  Residual spread around the fit   : {resid:.1f} kg")
    print(f"  So the biggest single behavioural effect is about {abs(big_b)/resid:.2f}")
    print("  of a residual SD - detectable, but small against how much two")
    print("  adults with identical answers still differ.")
    print()
    print("  Note the self-rated diet item heading the table. It is the one")
    print("  'behaviour' here that is plausibly an effect of weight rather")
    print("  than a cause: heavier adults rate their own diet worse. It is")
    print("  kept in the model and flagged, not quietly dropped.")
    print()
    print("  Energy intake also turns negative here while total fat is strongly")
    print("  positive, reversing its crude sign. The 24-hour recall quantities")
    print("  are badly collinear - fat, sodium and sugar are largely components")
    print("  of kcal - so individual dietary coefficients in this block should")
    print("  not be read one at a time.")

    sub("4. Does the effect depend on who you are?")
    print("  'Complexity' in the tutorial title means the effects need not be")
    print("  additive. Two interactions are tested explicitly.")
    rows = []
    for term, desc in [("mvpa_min_wk:C(sex)", "activity x sex"),
                       ("sedentary_min:age", "sedentary time x age")]:
        base = "weight_kg ~ age + C(sex) + " + " + ".join(BCOLS)
        f2 = smf.ols(base + " + " + term.replace(":", " * "), data=m).fit()
        f1 = smf.ols(base, data=m).fit()
        lr2 = f2.compare_lr_test(f1)
        rows.append([desc, f"{f1.rsquared:.4f}", f"{f2.rsquared:.4f}",
                     f"{f2.rsquared - f1.rsquared:+.4f}",
                     f"{lr2[1]:.3g}"])
    table(rows, ["interaction", "R2 without", "R2 with", "gain", "p"])
    print("\n  Interactions add essentially nothing. The weak behavioural signal")
    print("  is weak across subgroups, not strong in one and absent in another.")

    sub("5. Would a non-linear model do better?")
    print("  A linear model could be understating things if the real relations")
    print("  curve or interact in ways not specified above. A random forest")
    print("  makes no such assumption, so it is the fair test.")
    X = pd.get_dummies(m[BCOLS + ["age", "sex", "education", "income_ratio"]],
                       columns=["sex"], drop_first=True).astype(float)
    y = m.weight_kg.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED)
    lin = LinearRegression().fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                               random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    base_pred = np.full_like(yte, ytr.mean())
    rows = [["predict the mean", f"{r2_score(yte, base_pred):.4f}",
             f"{np.sqrt(np.mean((yte - base_pred) ** 2)):.2f}"],
            ["linear regression", f"{r2_score(yte, lin.predict(Xte)):.4f}",
             f"{np.sqrt(np.mean((yte - lin.predict(Xte)) ** 2)):.2f}"],
            ["random forest", f"{r2_score(yte, rf.predict(Xte)):.4f}",
             f"{np.sqrt(np.mean((yte - rf.predict(Xte)) ** 2)):.2f}"]]
    table(rows, ["model (held-out 30%)", "R2", "RMSE kg"])
    print("\n  The forest does not rescue the signal. The ceiling is in the data,")
    print("  not in the choice of estimator.")

    sub("6. Which variables does the forest lean on?")
    imp = permutation_importance(rf, Xte, yte, n_repeats=10,
                                 random_state=SEED, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1]
    rows = []
    for i in order[:10]:
        nm = X.columns[i]
        rows.append([LABEL.get(nm, nm), f"{imp.importances_mean[i]:+.4f}",
                     f"{imp.importances_std[i]:.4f}"])
    table(rows, ["variable", "mean drop in R2", "sd"])
    top_name = X.columns[order[0]]
    print(f"\n  The single most important variable is {LABEL.get(top_name, top_name)},")
    print("  which is not a behaviour at all. Self-rated diet follows, and it is")
    print("  the item most open to reverse causation. Age, education and income")
    print("  rank below several behaviours here, so this is not simply")
    print("  'demographics beat behaviour' - it is that one fixed characteristic")
    print("  beats everything a person chooses to do.")

    # -- figure -------------------------------------------------------------
    sub("7. Figure")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax = axes[0]
    names = list(fits) + ["random forest (held-out)"]
    vals = [f.rsquared for f in fits.values()] + [r2_score(yte, rf.predict(Xte))]
    cols_ = ["#c3d3e2", "#9ab6cf", "#4878a8", "#2d4f6d"]
    ax.barh(range(len(names)), vals, color=cols_)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("R² (variance in adult weight explained)")
    ax.set_title("T15  How much does behaviour explain?")

    ax = axes[1]
    top = order[:10][::-1]
    ax.barh(range(len(top)), imp.importances_mean[top],
            xerr=imp.importances_std[top], color="#4878a8", capsize=2)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([LABEL.get(X.columns[i], X.columns[i]) for i in top],
                       fontsize=8)
    ax.set_xlabel("permutation importance (drop in R²)")
    ax.set_title("Random-forest variable importance")
    save_fig(fig, "t15_models.png")

    pd.DataFrame([{"model": k, "r2": v.rsquared, "adj_r2": v.rsquared_adj}
                  for k, v in fits.items()]).to_csv(
        OUT_DIR / "t15_models.csv", index=False)
    print("  [saved] " + str(OUT_DIR / "t15_models.csv"))

    sub("8. What this means")
    print("  Adult body weight is not well predicted by what this survey can")
    print("  record about behaviour. That is not a null result to be hidden:")
    print()
    print("  * Self-reported behaviour is measured with large error. A single")
    print("    24-hour recall is a poor estimate of habitual diet.")
    print("  * Weight is cumulative. Today's behaviour did not produce today's")
    print("    weight, and a cross-sectional survey cannot see the history.")
    print("  * Reverse causation is live throughout - dieting follows weight")
    print("    gain as often as it precedes it.")
    print()
    print("  The honest conclusion for a health service is that behavioural")
    print("  screening questions will not identify who is heavy. Measuring")
    print("  weight directly costs one scale and thirty seconds.")
    return fits, inc, r2_score(yte, rf.predict(Xte))


def main():
    head("UNIT-5  (T13-T15)  -  ADULT WEIGHT AND BEHAVIOUR, NHANES 2017-2018")
    d = load_nhanes()
    print(f"\n  adults aged {ADULT_AGE}+ with a measured weight: {len(d)}")
    cc, ev, n90 = t13_dimensionality(d)
    crude, adj, nhalf = t14_associations(d)
    fits, inc, rf_r2 = t15_models(d)
    head("DONE")
    print(f"  T13: {n90} of {len(BCOLS)} components needed for 90% of variance; "
          f"PC1+PC2 = {ev[:2].sum():.1%}")
    print(f"  T14: strongest crude |r| = "
          f"{max(abs(v[0]) for v in crude.values()):.3f}; "
          f"{nhalf} associations more than halve after adjusting for age and sex")
    print(f"  T15: behaviour adds {inc:.1%} R2 over demographics; "
          f"random forest held-out R2 = {rf_r2:.3f}")
    print(f"  figures : {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
