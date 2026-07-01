import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Quality Summary Soave Raphael", layout="wide")

DATA_DIR = "."


@st.cache_data
def load_data():
    dfcaract = pd.read_csv(f"{DATA_DIR}/caract-2024.csv", sep=";")
    dflieux = pd.read_csv(f"{DATA_DIR}/lieux-2024.csv", sep=";")
    dfusagers = pd.read_csv(f"{DATA_DIR}/usagers-2024.csv", sep=";")
    dfvehicules = pd.read_csv(f"{DATA_DIR}/vehicules-2024.csv", sep=";")
    return {"caract": dfcaract, "lieux": dflieux, "usagers": dfusagers, "vehicules": dfvehicules}


@st.cache_data
def compute_metrics(_dfs):
    dfcaract, dflieux, dfusagers, dfvehicules = (
        _dfs["caract"], _dfs["lieux"], _dfs["usagers"], _dfs["vehicules"],
    )

    missing = {}
    for name, df in _dfs.items():
        n = len(df)
        nan_ = df.isna().sum()
        nan_ = nan_[nan_ > 0].sort_values(ascending=False)
        sentinel = {}
        for col in df.columns:
            if col in ("Num_Acc", "id_usager", "id_vehicule", "num_veh"):
                continue
            c = int((df[col].astype(str).str.strip() == "-1").sum())
            if c > 0:
                sentinel[col] = c
        missing[name] = {
            "n": n,
            "nan": nan_,
            "sentinel": pd.Series(sentinel).sort_values(ascending=False) if sentinel else pd.Series(dtype=int),
        }

    lat = dfcaract["lat"].str.replace(",", ".").astype(float)
    lon = dfcaract["long"].str.replace(",", ".").astype(float)
    dep = dfcaract["dep"].astype(str)
    metro = dep.str.len() <= 2
    out_of_bbox = metro & ((lat < 41) | (lat > 51.5) | (lon < -5.5) | (lon > 9.7))
    swapped = dep.isin(["2A", "2B"]) & (lat < 20)

    age = (dfusagers.merge(dfcaract[["Num_Acc", "an"]], on="Num_Acc", how="left")
           .eval("age = an - an_nais"))["age"]

    impossible_vma = dflieux.loc[dflieux["vma"] > 130, "vma"]

    pr1 = pd.to_numeric(dflieux["pr1"].astype(str).str.replace("\xa0", "").str.strip(), errors="coerce")
    neg_pr1 = pr1[(pr1 < 0) & (pr1 != -1)]

    nbv_bad = int((dflieux["nbv"] == "#VALEURMULTI").sum())

    dup_exact = {name: int(df.duplicated().sum()) for name, df in _dfs.items()}

    counts = dflieux["Num_Acc"].value_counts()
    lieux_multi = int((counts > 1).sum())
    lieux_multi_pct = lieux_multi / dflieux["Num_Acc"].nunique() * 100

    seat_conflicts = dfusagers[dfusagers["catu"].isin([1, 2]) & ~dfusagers["place"].isin([-1, 0])]
    seat_conflict_count = int(seat_conflicts.duplicated(subset=["Num_Acc", "id_vehicule", "place"]).sum())

    consistency = {
        "coord_swapped": int(swapped.sum()),
        "age_min": float(age.min()),
        "age_max": float(age.max()),
        "age_negative": int((age < 0).sum()),
        "age_over_100": int((age > 100).sum()),
        "vma_impossible": len(impossible_vma),
        "vma_values": sorted(impossible_vma.unique().tolist()),
        "pr1_negative": len(neg_pr1),
        "nbv_bad": nbv_bad,
        "dup_exact": dup_exact,
        "lieux_multi": lieux_multi,
        "lieux_multi_pct": lieux_multi_pct,
        "seat_conflict_count": seat_conflict_count,
    }

    return missing, consistency

CANDIDATE_KEYS = {
    "caract": ["Num_Acc"],
    "lieux": ["Num_Acc"],
    "vehicules": ["id_vehicule"],
    "usagers": ["id_usager"],
}


@st.cache_data
def compute_pk_check(_dfs):
    rows = []
    for name, df in _dfs.items():
        key_cols = CANDIDATE_KEYS[name]
        n = len(df)
        n_nulls = int(df[key_cols].isna().any(axis=1).sum())
        n_distinct = int(df[key_cols].drop_duplicates().shape[0])
        is_unique = n_distinct == n
        is_valid = is_unique and n_nulls == 0

        if is_valid:
            reason = f"`{', '.join(key_cols)}` uniquely identifies every row — valid primary key."
        elif name == "lieux":
            reason = (
                "**No valid primary key.** `Num_Acc` *should* identify exactly 1 row per accident "
                "(as it does in `caract`), but it doesn't: 28.8% of accidents have 2 to 5 rows in "
                "`lieux` (see Part C — relational integrity issue). The raw file provides no other "
                "column that disambiguates these duplicate rows (no sequence/version number), so "
                "there is no natural key — a surrogate key can only be created *after* deduplicating "
                "in the Silver layer."
            )
        else:
            reason = (
                f"**No valid primary key.** `{', '.join(key_cols)}` is not unique and/or contains "
                f"nulls ({n_distinct:,} distinct values / {n:,} rows, {n_nulls} nulls)."
            ).replace(",", " ")

        rows.append({
            "File": name,
            "Candidate key": ", ".join(key_cols),
            "Rows": n,
            "Distinct key values": n_distinct,
            "Nulls in key": n_nulls,
            "Valid PK?": "Yes" if is_valid else "No",
            "Explanation": reason,
        })
    return pd.DataFrame(rows)


dfs = load_data()
missing, consistency = compute_metrics(dfs)
pk_check = compute_pk_check(dfs)

st.title("Data Quality Summary Soave Raphael")
st.caption(
    "French Road Safety Open Data 2024 (ONISR / data.gouv.fr) — "
    "consolidated Quality Report & Impact Analysis, based on Parts B and C."
)

tab_overview, tab_report, tab_impact = st.tabs(
    ["Dataset overview", "Quality report", "Impact analysis"]
)


with tab_overview:
    st.subheader("The 4 relational tables (grain: 1 row per accident / vehicle / person)")
    cols = st.columns(4)
    labels = {
        "caract": "caract — accident characteristics",
        "lieux": "lieux — road infrastructure",
        "vehicules": "vehicules — vehicles involved",
        "usagers": "usagers — people involved",
    }
    for col, (name, df) in zip(cols, dfs.items()):
        col.metric(labels[name], f"{len(df):,} rows".replace(",", " "), f"{df.shape[1]} columns")

    st.divider()
    st.subheader("Primary key check")
    st.markdown(
        "For each file, is there a column (or set of columns) that uniquely and completely "
        "identifies each row? Checked as: **distinct values == row count** and **no nulls** "
        "in the candidate key."
    )
    st.dataframe(
        pk_check.drop(columns=["Explanation"]),
        use_container_width=True, hide_index=True,
    )
    for _, r in pk_check.iterrows():
        if r["Valid PK?"] == "No":
            st.warning(f"**{r['File']}** — {r['Explanation']}")
        else:
            st.caption(f"**{r['File']}**: {r['Explanation']}")

    st.divider()
    st.subheader("Missing values — true `NaN` vs. hidden `-1` sentinel")
    st.markdown(
        "ONISR encodes *\"no information\"* two ways: an empty CSV field (`NaN`), and an explicit "
        "sentinel code **`-1`** (*non renseigné*) inside otherwise-categorical columns. "
        "Both are shown below, per file."
    )
    mcols = st.columns(4)
    for col, name in zip(mcols, dfs.keys()):
        with col:
            st.markdown(f"**{name}** (n={missing[name]['n']:,})".replace(",", " "))
            nan_s = missing[name]["nan"]
            sent_s = missing[name]["sentinel"]
            if not nan_s.empty:
                st.caption("NaN")
                st.dataframe((nan_s.to_frame("count")
                              .assign(**{"%": (nan_s / missing[name]["n"] * 100).round(1)})),
                             use_container_width=True, height=140)
            if not sent_s.empty:
                st.caption("'-1' sentinel")
                st.dataframe((sent_s.head(6).to_frame("count")
                              .assign(**{"%": (sent_s.head(6) / missing[name]["n"] * 100).round(1)})),
                             use_container_width=True, height=180)
            if nan_s.empty and sent_s.empty:
                st.success("No missing values")


with tab_report:
    st.subheader("Quality report — main issues discovered")
    st.markdown(
        "This consolidates every issue identified in **Part B** (missing values) and "
        "**Part C** (consistency & validity checks) into a single ranked list."
    )

    issues = [
        dict(sev="High", category="Missing value", file="usagers",
             issue="`an_nais` missing for 2.1% of people",
             metric=f"{missing['usagers']['nan'].get('an_nais', 0):,} rows".replace(",", " "),
             note="Small volume, but it's the only source of age — a key covariate for every severity analysis."),
        dict(sev="High", category="Relational integrity", file="lieux",
             issue="Accidents with more than 1 row in `lieux` (should be exactly 1, like `caract`)",
             metric=f"{consistency['lieux_multi']:,} accidents ({consistency['lieux_multi_pct']:.1f}%)".replace(",", " "),
             note="Any join on `Num_Acc` silently duplicates ~29% of accidents."),
        dict(sev="Moderate", category="Hidden missingness ('-1')", file="lieux",
             issue="`larrout` (roadway width) not specified",
             metric=f"{missing['lieux']['sentinel'].get('larrout', 0):,} rows ({missing['lieux']['sentinel'].get('larrout', 0)/missing['lieux']['n']*100:.1f}%)".replace(",", " "),
             note="Blocks lane/width-based road-infrastructure analysis for most accidents."),
        dict(sev="Moderate", category="Hidden missingness ('-1')", file="lieux",
             issue="`pr` / `pr1` (reference marker / distance) not specified",
             metric="~27,400 rows (~39%) each",
             note="Blocks precise in-road localization; `catr`/`vma`/`infra` remain usable substitutes."),
        dict(sev="Moderate", category="Duplicate / conflicting record", file="usagers",
             issue="Driver/passenger sharing the same seat of the same vehicle",
             metric=f"{consistency['seat_conflict_count']:,} rows".replace(",", " "),
             note="Physically impossible — the correct occupant can't be determined from the data alone."),
        dict(sev="Low", category="Out-of-range value", file="lieux",
             issue="Impossible speed limits (`vma`)",
             metric=f"{consistency['vma_impossible']} rows -> {consistency['vma_values']}",
             note="Likely data-entry typos (e.g. 50 -> 500); easy to filter out."),
        dict(sev="Low", category="Out-of-range value", file="caract",
             issue="Latitude/longitude swapped (Corsica)",
             metric=f"{consistency['coord_swapped']} rows",
             note="Rare, but silently breaks map joins/geocoding for those rows."),
        dict(sev="Low", category="Out-of-range value", file="lieux",
             issue="Negative distance to reference marker (`pr1`)",
             metric=f"{consistency['pr1_negative']} rows",
             note="Physically meaningless (should be ≥ 0)."),
        dict(sev="Low", category="Categorical anomaly", file="lieux",
             issue="`#VALEURMULTI` Excel-export artifact in `nbv`",
             metric=f"{consistency['nbv_bad']} rows",
             note="Spreadsheet corruption, not a real lane count."),
        dict(sev="Low", category="Missing value", file="vehicules",
             issue="`occutc` (public-transport occupants) missing",
             metric=f"{missing['vehicules']['nan'].get('occutc', 0):,} rows (99.0%)".replace(",", " "),
             note="Structural — only applies to buses/coaches, consistent with `catv`."),
        dict(sev="Low", category="Missing value", file="caract",
             issue="`adr` (postal address) missing",
             metric=f"{missing['caract']['nan'].get('adr', 0):,} rows (4.2%)".replace(",", " "),
             note="Low impact — `lat`/`long` (0% missing) is a strictly better location key."),
        dict(sev="Low", category="Out-of-range value", file="usagers",
             issue="Age range (all road users: drivers, passengers, pedestrians)",
             metric=(f"{consistency['age_negative']} negative ages, "
                     f"{consistency['age_over_100']} people > 100 y/o "
                     f"(max {consistency['age_max']:.0f})"),
             note="No negative ages found — clean on that front. The 6 centenarians "
                  "(2 drivers, 2 passengers, 2 pedestrians, ages 101-110) are extreme but not "
                  "impossible; worth a manual spot-check rather than a fix."),
    ]

    sev_filter = st.multiselect(
        "Filter by severity", ["High", "Moderate", "Low"],
        default=["High", "Moderate", "Low"],
    )
    issues_df = pd.DataFrame(issues)
    issues_df = issues_df[issues_df["sev"].isin(sev_filter)]
    issues_df = issues_df.rename(columns={
        "sev": "Severity", "category": "Category", "file": "File",
        "issue": "Issue", "metric": "Metric", "note": "Why it matters",
    })
    st.dataframe(issues_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        """
**Summary.** Across the 4 files, data quality is overall good: the vast majority of columns
(~45 out of ~60) have **zero** true missing values, and every coded categorical column respects
the ONISR codebook except two localized artifacts (`nbv`'s `#VALEURMULTI`, `vma`'s impossible
speeds). The dataset's real weak points are not random noise but **structural**:

- **Conditional fields** (`secu2/3`, `locp`/`actp`/`etatp`, `v2`, `lartpc`, `occutc`) show very
  high `-1`/`NaN` rates simply because they only apply to a subset of rows (pedestrians, multi-
  equipment wearers, buses, divided roads) — not a collection failure.
- **`an_nais`** is the one genuinely concerning gap: low volume, but it is the sole source of age.
- **`lieux`'s 1-to-many relationship with accidents (28.8%)** is the most impactful integrity
  issue, because it breaks the assumed 1-row-per-accident grain silently on every join.
"""
    )

with tab_impact:
    st.subheader("Impact analysis — how these issues affect downstream analytics")

    with st.expander("Demographic / severity analysis (age, gender, injury severity)", expanded=True):
        st.markdown(
            """
- `an_nais` missing for 2.1% of people means any age-stratified statistic (child/elderly risk,
  age vs. severity) silently drops those rows. If missingness correlates with the outcome
  (e.g. incomplete paperwork for fatal cases), estimates can be **biased**, not just smaller-sample.
- **Recommendation:** always report the % of rows excluded whenever age is used; never impute
  a fabricated birth year onto the outcome variable's own covariate.
"""
        )

    with st.expander("Geospatial analysis (mapping, hotspot detection)"):
        st.markdown(
            """
- The 3 Corsica rows with swapped `lat`/`long` will plot **outside France** on any map and could
  poison distance/clustering computations (e.g. nearest-hotspot analysis) if not caught first.
- `adr` missingness (4.2%) has **no impact** on geospatial analysis since `lat`/`long` are 100%
  populated and more precise anyway.
"""
        )

    with st.expander("Road-infrastructure analysis (`lieux`)"):
        st.markdown(
            """
- The **28.8% multi-row accidents** in `lieux` is the highest-impact issue in the whole dataset:
  a naive `caract ⋈ lieux` join on `Num_Acc` will duplicate ~1 in 3 accidents, inflating every
  downstream count (accidents by road category, by speed limit, etc.) unless deduplicated first.
- High `-1` rates on `larrout`, `pr`, `pr1`, `v1` block any analysis needing precise road geometry
  or in-road location, but `catr`/`vma`/`infra`/`situ` remain reliable substitutes for road-context
  analysis (e.g. accidents by road type or speed limit are unaffected).
- The impossible `vma` values (500, 900 km/h...) would silently distort any speed-limit
  aggregation (e.g. average speed limit per road category) if not filtered out first.
"""
        )

    with st.expander("Vehicle / occupant analysis (`vehicules`, `usagers`)"):
        st.markdown(
            """
- `occutc` (99% missing) is fine to use **only** when filtered to `catv` = bus/coach; using it
  unfiltered elsewhere would wrongly suggest almost no vehicles carry occupants.
- The pedestrian-only fields (`locp`, `actp`, `etatp`) and 2nd/3rd equipment fields (`secu2/3`)
  look like 43–92% missing in isolation, but analyzing them **without conditioning on `catu`**
  (pedestrian vs. driver/passenger) would overstate the real gap and could trigger unnecessary
  row-dropping in models that don't even need these fields.
- The 946 seat-conflict rows in `usagers` mean any per-seat occupant analysis (e.g. front vs.
  rear-seat injury severity) has a small but real risk of double-counting or mis-assigning a
  person to the wrong seat.
"""
        )

    st.divider()
    st.markdown(
        "**Bottom line:** none of the issues found are severe enough to block analysis outright, "
        "but each requires an explicit remediation step (see Part B/C) before being used — "
        "otherwise they propagate silently into biased aggregates, inflated joins, or "
        "under/over-stated missingness."
    )
