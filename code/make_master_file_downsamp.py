import pandas as pd
from pathlib import Path

"""
Builds a master csv for training/testing a translation vs. original English classifier.

- Originals: English-language books from Gutenberg
- Translations: Books translated into English, with multiple translation sources
  (e.g., multiple LLMs, multiple human translations, etc)

Each row in the master file represents one paragraph from one source.
The `para_source` column identifies where the paragraph came from.
The `sample_weight` column corrects for the fact that a single paragraph
may have many translation versions but only one original version.

For splitting into train/test/validation folds, use group k-fold on `book_id`
so that whole books are kept together and never split across folds.
"""


Orig_dir       = Path("../../2_anonymization/english_dataset_anon")       # folder with originals
Transl_dir     = Path("../../2_anonymization/par3_books_gemma_anon_key")    # folder with translations

Original_meta   = Path("english_fiction_metadata.csv")
transl_meta = Path("par3_dataset_metadata_final.csv")

output_filee         = Path("master_length20_weights_matched.csv")

glob_patt        = "*.csv"

# filters 
min_Year            = None   # e.g., 1850 = excludes books published before 1850
max_Year            = None   
min_Len             = 20     # min word count for a paragraph to be included
max_Len             = None   # max word count (None = no limit)

### translation sources to include...
# translator_1 through translator_5 are added dynamically.
translation_sources = ["gemma", "gt", "translator"]   # "translator" expands to translator_1..5

rand_seed         = 39


def word_count(text):
    if pd.isna(text):
        return 0
    return len(str(text).split())

def apply_len_filter(df, col):
    """Filter rows by word count of a given column."""
    lengths = df[col].apply(word_count)
    if min_Len is not None:
        df = df[lengths >= min_Len]
    if max_Len is not None:
        df = df[lengths <= max_Len]
    return df.copy()

def apply_year_filter(df, year_col):
    """Filter rows by publication year."""
    if min_Year is not None:
        df = df[df[year_col] >= min_Year]
    if max_Year is not None:
        df = df[df[year_col] <= max_Year]
    return df.copy()


print("Loading metadata...")

orig_meta = pd.read_csv(Original_meta)
orig_meta = orig_meta.rename(columns={
    "Gid_x":                  "book_id",
    "Pubdate":                "pubdate",
    "Volume Title in Gutenberg": "title",
    "Author":                 "author",
})
orig_meta = orig_meta[["book_id", "title", "author", "pubdate"]]
orig_meta["book_id"] = orig_meta["book_id"].astype(str)

trans_meta = pd.read_csv(transl_meta)
trans_meta = trans_meta.rename(columns={
    "Book_id":  "book_id",
    "Pub_Year": "pubdate",
    "Title":    "title",
    "Author":   "author",
})

# drop the duplicate books - these were manually identified before..
if "Duplicate" in trans_meta.columns:
    before = len(trans_meta)
    trans_meta = trans_meta[trans_meta["Duplicate"].str.strip().str.lower() == "n"]
    print(f"  Dropped {before - len(trans_meta)} duplicate books from translation metadata")

trans_meta = trans_meta[["book_id", "title", "author", "pubdate"]]
trans_meta["book_id"] = trans_meta["book_id"].astype(str)

valid_orig_ids  = set(orig_meta["book_id"])
#print(valid_orig_ids)
valid_trans_ids = set(trans_meta["book_id"])

print(f"  Original books in metadata:    {len(valid_orig_ids)}")
print(f"  Translation books in metadata: {len(valid_trans_ids)}")


print("\nProcessing originals...")

orig_required = {
    "book_id", "paragraph_id", "source",
    "gutenberg_para", "gutenberg_para_len",
    "gutenberg_para_anon_ner", "gutenberg_para_anon_pos"
}

orig_parts = []
orig_paths = sorted(Orig_dir.glob(glob_patt))
if orig_paths:
    print("found originals!")
else:
    print("originals aren't in the indicated directory...")

for p in orig_paths:
    df = pd.read_csv(p)
    #print(df.columns.tolist())
    df["book_id"] = df["book_id"].astype(str)

    missing = orig_required - set(df.columns)
    if missing:
        print(f"  WARNING: {p.name} missing columns {sorted(missing)}, skipping")
        continue

    df = df[df["book_id"].isin(valid_orig_ids)]
    if df.empty:
        continue

    # Length filter
    if min_Len is not None:
        df = df[df["gutenberg_para_len"] >= min_Len]
    if max_Len is not None:
        df = df[df["gutenberg_para_len"] <= max_Len]
    if df.empty:
        print(f"{df} is empty.")
        continue

    out = df[[
        "book_id", "paragraph_id",
        "gutenberg_para",
        "gutenberg_para_anon_ner",
        "gutenberg_para_anon_pos",
        "gutenberg_para_len",
    ]].rename(columns={
        "gutenberg_para":         "paragraph",
        "gutenberg_para_anon_ner": "paragraph_anon_ner",
        "gutenberg_para_anon_pos": "paragraph_anon_pos",
        "gutenberg_para_len":     "paragraph_len",
    })

    out["para_source"]  = "original_en"
    out["src_lang_code"] = "en"
    out["translation"]  = 0

    orig_parts.append(out)

originals_all = pd.concat(orig_parts, ignore_index=True) if orig_parts else pd.DataFrame()

#adding this block in for testing...
print("orig_parts:", len(orig_parts))
print("originals_all shape:", originals_all.shape)
print("originals_all columns:", originals_all.columns.tolist())
print("orig_meta columns:", orig_meta.columns.tolist())

originals_all = originals_all.merge(orig_meta, on="book_id", how="left")
originals_all["sample_weight"] = 1.0

print(f"  Original rows kept: {len(originals_all)}")


print("\nProcessing translations...")

# Detect which translator_N columns exist dynamically
translator_slots = [f"translator_{i}" for i in range(1, 6)] # only goes up to 5... FYI

trans_parts = []
trans_paths = sorted(Transl_dir.glob(glob_patt))

for p in trans_paths:
    df = pd.read_csv(p)
    df["book_id"] = df["Book_id"].astype(str) if "Book_id" in df.columns else df["book_id"].astype(str)

    df = df[df["book_id"].isin(valid_trans_ids)]
    if df.empty:
        continue

    # Rename for consistency
    # if "Book_id" in df.columns:
    #     df = df.rename(columns={"Book_id": "book_id"})

    # Build list of (para_col, anon_ner_col, anon_pos_col, source_label) tuples
    sources = []

    if "gemma" in translation_sources and "gemma_para" in df.columns:
        sources.append(("gemma_para", "gemma_para_anon_ner", "gemma_para_anon_pos", "gemma"))

    if "gt" in translation_sources and "gt_para" in df.columns:
        sources.append(("gt_para", "gt_para_anon_ner", "gt_para_anon_pos", "gt"))

    if "translator" in translation_sources:
        for slot in translator_slots:
            para_col     = f"{slot}_para"
            anon_ner_col = f"{slot}_para_anon_ner"
            anon_pos_col = f"{slot}_para_anon_pos"
            if para_col in df.columns:
                sources.append((para_col, anon_ner_col, anon_pos_col, slot))

    n_sources = len(sources)
    if n_sources == 0:
        print(f"  WARNING: {p.name} has no recognised translation columns, skipping")
        continue

    #weight = 1.0 / n_sources #edit out where it calcualates sample weight here... adding it at end so captures after any filtering

    for para_col, anon_ner_col, anon_pos_col, label in sources:
        sub = df[["book_id", "paragraph_id", para_col]].copy()
        sub = sub.dropna(subset=[para_col])

        # Get paragraph_len for translations (already have this for originals)
        sub["paragraph_len"] = sub[para_col].apply(word_count)

        if min_Len is not None:
            sub = sub[sub["paragraph_len"] >= min_Len]
        if max_Len is not None:
            sub = sub[sub["paragraph_len"] <= max_Len]
        if sub.empty:
            continue

        sub = sub.rename(columns={para_col: "paragraph"})

        if anon_ner_col in df.columns:
            sub["paragraph_anon_ner"] = df.loc[sub.index, anon_ner_col].values
        else:
            sub["paragraph_anon_ner"] = None

        if anon_pos_col in df.columns:
            sub["paragraph_anon_pos"] = df.loc[sub.index, anon_pos_col].values
        else:
            sub["paragraph_anon_pos"] = None

        sub["para_source"]    = label
        sub["src_lang_code"]  = df.loc[sub.index, "src_lang_code"].values if "src_lang_code" in df.columns else None
        sub["translation"]    = 1
        #sub["sample_weight"]  = weight

        trans_parts.append(sub)

translations_all = pd.concat(trans_parts, ignore_index=True) if trans_parts else pd.DataFrame()
translations_all = translations_all.merge(trans_meta, on="book_id", how="left")

print(f"  Translation rows kept: {len(translations_all)}")


if min_Year is not None or max_Year is not None:
    before = len(originals_all) + len(translations_all)
    originals_all    = apply_year_filter(originals_all,    "pubdate")
    translations_all = apply_year_filter(translations_all, "pubdate")
    after = len(originals_all) + len(translations_all)
    print(f"\nYear filter dropped {before - after} rows")


# Combine everything, shuffle it, save it.
print("\nCombining and saving...")

master = pd.concat([originals_all, translations_all], ignore_index=True)
master = master.sample(frac=1, random_state=rand_seed).reset_index(drop=True)

######### Downsamping Section ####
# Undersample translations to match the length distribution of originals
# Keeps all originals, samples translations down per length bin

print("\nApplying length-matched downsampling...")

bin_sze = 10  

originals_df    = master[master["translation"] == 0].copy()
translations_df = master[master["translation"] == 1].copy()

originals_df["len_bin"]    = (originals_df["paragraph_len"]    // bin_sze) * bin_sze
translations_df["len_bin"] = (translations_df["paragraph_len"] // bin_sze) * bin_sze

matched_trans_parts = []

for bin_val, orig_group in originals_df.groupby("len_bin"):
    n_needed    = len(orig_group)
    trans_group = translations_df[translations_df["len_bin"] == bin_val]

    if len(trans_group) == 0:
        print(f"  WARNING: no translations in length bin {bin_val}-{bin_val+bin_sze}, skipping")
        continue

    n_sample = min(n_needed, len(trans_group))
    sampled  = trans_group.sample(n=n_sample, random_state=rand_seed)
    matched_trans_parts.append(sampled)

matched_translations = pd.concat(matched_trans_parts, ignore_index=True)


originals_df     = originals_df.drop(columns=["len_bin"])
matched_translations = matched_translations.drop(columns=["len_bin"])

master = pd.concat([originals_df, matched_translations], ignore_index=True)
master = master.sample(frac=1, random_state=rand_seed).reset_index(drop=True)

print(f"  Originals after matching:     {(master['translation']==0).sum():,}")
print(f"  Translations after matching:  {(master['translation']==1).sum():,}")
print(f"\n  Length distribution check:")
print(f"    Originals mean:     {master[master['translation']==0]['paragraph_len'].mean():.1f}")
print(f"    Translations mean:  {master[master['translation']==1]['paragraph_len'].mean():.1f}")


master = master[[
    "book_id",
    "title",
    "author",
    "pubdate",
    "src_lang_code",
    "paragraph_id",
    "para_source",
    "paragraph",
    "paragraph_anon_ner",
    "paragraph_anon_pos",
    "paragraph_len",
    "translation",
    "sample_weight",
]]


##### COMPUTE SAMPLE WEIGHTS
# Must happen AFTER downsampling — some translation versions may have been
# removed, making the original 1/n_sources weights stale

trans_mask = master["translation"] == 1
master.loc[trans_mask, "sample_weight"] = (
    master[trans_mask]
    .groupby(["book_id", "paragraph_id"])["para_source"]
    .transform("count")
    .rdiv(1)
)
master.loc[~trans_mask, "sample_weight"] = 1.0

print("\nSample weight distribution (translated paragraphs):")
print(master[trans_mask]["sample_weight"].value_counts().sort_index())

master.to_csv(output_filee, index=False)

print(f"\nDone. Wrote {len(master)} rows to {output_filee}")
print(f"  translation=0 (original): {(master['translation'] == 0).sum():,}")
print(f"  translation=1 (translated): {(master['translation'] == 1).sum():,}")
print(f"\nBreakdown by para_source:")
print(master["para_source"].value_counts().to_string())
print(f"\nUnique books:")
print(f"  Originals:    {master[master['translation']==0]['book_id'].nunique()}")
print(f"  Translations: {master[master['translation']==1]['book_id'].nunique()}")
print("\nBreakdown by src_lang_code:")
print(master[master['translation']==1].groupby('src_lang_code')['book_id'].nunique().to_string())