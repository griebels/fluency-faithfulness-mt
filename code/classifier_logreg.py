"""
classifier_logreg.py

Trains a classifier to tell apart original English paragraphs from translated
English paragraphs.

The model uses POS n-grams from the paragraph_anon_pos column, rather than the
raw text, so the classifier is mostly learning syntactic patterns instead of
memorizing words, names, or book-specific vocabulary.

The script uses book-level cross-validation, meaning paragraphs from the same
book are never split between train and test. It also stratifies the folds by
source language where possible. Languages with only 1-2 books are grouped into
a shared "rare" category so they get spread across the folds more evenly.

Each paragraph appears in the test set exactly once.

A few important choices:
- class_weight='balanced' helps with original/translated class imbalance
- sample_weight helps correct for paragraphs that have multiple translation versions
- book_id-level splits prevent book-level leakage
- POS-only features reduce lexical leakage
- solver='saga' and n_jobs=-1 make the logistic regression run faster on my machine

Usage:
    python classify_translations_kfold.py --csv master_length10.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)

warnings.filterwarnings("ignore")



textCol   = "paragraph_anon_pos"
labelCol  = "translation"
bookCol   = "book_id"
weightCol = "sample_weight"
langCol   = "src_lang_code"

ngram_range  = (1, 3)
maxFeatures = 20_000   

lr_c        = 10.0       
class_max_iter = 2000
rand_seed = 12
num_folds     = 10

# Languages with <= this many books get bucketed into a single "rare" stratum
rare_threshold = 2


def assign_strata(df):
    """
    Build a stratum label per book for stratified fold assignment.

    - Original English books ="original_en"
    - Translated books with > rare threshold = language code
    - Translated books from rare languages = "rare"
    """
    lang_counts = (
        df[df[labelCol] == 1]
        .drop_duplicates(subset=bookCol)
        .groupby(langCol)[bookCol]
        .count()
    )

    rare_langs = set(lang_counts[lang_counts <= rare_threshold].index)

    print(f"\n  Languages bucketed as 'rare' (n <= {rare_threshold} books):")
    for lang in sorted(rare_langs):
        print(f"    {lang}: {int(lang_counts[lang])} book(s)")

    book_df = (
        df.drop_duplicates(subset=bookCol)
        [[bookCol, labelCol, langCol]]
        .copy()
    )

    def get_stratum(row):
        if row[labelCol] == 0:
            return "original_en"
        lang = row[langCol]
        if pd.isna(lang) or lang in rare_langs:
            return "rare"
        return lang

    book_df["stratum"] = book_df.apply(get_stratum, axis=1)

    print(f"\n  Stratum distribution (books):")
    print(book_df["stratum"].value_counts().to_string())

    return book_df[[bookCol, "stratum"]]



def build_folds(df, book_strata):
    """
    Distribute books across folds using round-robin within each stratum.
    All paragraphs from a book go to the same fold.
    """
    book_fold_map = {}

    for stratum in sorted(book_strata["stratum"].unique()):
        books = book_strata[book_strata["stratum"] == stratum][bookCol].tolist()
        for i, book in enumerate(books):
            book_fold_map[book] = (i % num_folds) + 1

    df = df.copy()
    df["fold"] = df[bookCol].map(book_fold_map)

    unmapped = df["fold"].isna().sum()
    if unmapped > 0:
        print(f"  WARNING: {unmapped} rows could not be assigned to a fold")
    df["fold"] = df["fold"].fillna(1).astype(int)

    print(f"\n  Fold size distribution (paragraphs):")
    print(df["fold"].value_counts().sort_index().to_string())

    print(f"\n  Fold size distribution (books):")
    fold_book_counts = df.drop_duplicates(bookCol).groupby("fold")[bookCol].count()
    print(fold_book_counts.to_string())

    return df



def run_fold(fold_num, train_df, test_df):
    X_train = train_df[textCol].fillna("").tolist()
    y_train = train_df[labelCol].values
    w_train = train_df[weightCol].values

    X_test  = test_df[textCol].fillna("").tolist()
    y_test  = test_df[labelCol].values

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=maxFeatures,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    clf = LogisticRegression(
        class_weight="balanced", 
        C=lr_c,
        solver="saga",       # faster on large sparse matrices?
        max_iter=class_max_iter,
        n_jobs=-1,           # use all CPU cores
        random_state=rand_seed,
    )
    clf.fit(X_train_vec, y_train, sample_weight=w_train)

    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)[:, 1]   # P(translated)

    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")

    para_df = test_df.copy().reset_index(drop=True)
    para_df["predicted_label"] = y_pred
    para_df["logreg_prob"]     = y_prob

    return {
        "fold":           fold_num,
        "n_train":        len(y_train),
        "n_test":         len(y_test),
        "accuracy":       round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":       round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_translation": round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "precision":      round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":         round(recall_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":        round(auc, 4),
        "para_df":        para_df,
    }



def main():
    global num_folds

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True,                 help="Path to master CSV")
    parser.add_argument("--output", default="results_kfold.csv",   help="Base name for output files")
    parser.add_argument("--folds",  default=num_folds, type=int,     help=f"Number of folds (default: {num_folds})")
    args = parser.parse_args()

    #global N_FOLDS
    num_folds = args.folds

    print(f"\nLoading {args.csv} ...")
    df = pd.read_csv(args.csv)
    print(f"  Loaded {len(df):,} rows")

    for col in [textCol, labelCol, bookCol, weightCol]:
        if col not in df.columns:
            raise ValueError(f"Required column {col!r} not found. Available: {df.columns.tolist()}")

    if langCol not in df.columns:
        df[langCol] = None
    df.loc[df[labelCol] == 0, langCol] = df.loc[df[labelCol] == 0, langCol].fillna("en")

    df = df.dropna(subset=[textCol, labelCol, bookCol])
    df[labelCol]  = df[labelCol].astype(int)
    df[weightCol] = df[weightCol].fillna(1.0)

    print(f"\n  translation=0 (original):   {(df[labelCol]==0).sum():,}")
    print(f"  translation=1 (translated): {(df[labelCol]==1).sum():,}")
    print(f"\n  para_source breakdown:")
    print(df["para_source"].value_counts().to_string())
    print(f"\n  Source language breakdown (translated books):")
    print(df[df[labelCol]==1].groupby(langCol)[bookCol].nunique().to_string())

    print(f"Assigning stratified {num_folds}-fold splits ...!!")
    book_strata = assign_strata(df)
    df = build_folds(df, book_strata)

    print(f"Running Logistic Regression ({num_folds}-fold) ...!!")

    all_results  = []
    all_para_dfs = []

    for fold_num in range(1, num_folds + 1):
        train_df = df[df["fold"] != fold_num].copy()
        test_df  = df[df["fold"] == fold_num].copy()

        test_langs = (
            test_df[test_df[labelCol] == 1][langCol]
            .value_counts()
            .to_dict()
        )

        print(f"  Fold {fold_num:>2}/{num_folds}: "
              f"train={len(train_df):,}  test={len(test_df):,}  "
              f"langs={test_langs} ... ",
              end="", flush=True)

        result = run_fold(fold_num, train_df, test_df)
        all_results.append(result)
        all_para_dfs.append(result["para_df"])

        print(f"AUC={result['auc_roc']:.3f}  "
              f"F1_macro={result['f1_macro']:.3f}  "
              f"acc={result['accuracy']:.3f}")

    fold_cols = [
        "fold", "n_train", "n_test",
        "accuracy", "f1_macro", "f1_translation",
        "precision", "recall", "auc_roc",
    ]
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k in fold_cols}
        for r in all_results
    ])

    metric_cols = ["accuracy", "f1_macro", "f1_translation", "precision", "recall", "auc_roc"]
    summary     = results_df[metric_cols].agg(["mean", "std"]).round(4)

    print(f"SUMMARY (mean +/- std across {num_folds} folds)")
    for col in metric_cols:
        m = summary.loc["mean", col]
        s = summary.loc["std",  col]
        print(f"  {col:<20}: {m:.4f} +/- {s:.4f}")


    para_combined = pd.concat(all_para_dfs, ignore_index=True)

    output_cols = [
        bookCol, "paragraph_id", "para_source", langCol,
        "paragraph_len", labelCol,
        "fold", "predicted_label", "logreg_prob", weightCol,
    ]
    output_cols   = [c for c in output_cols if c in para_combined.columns]
    para_combined = para_combined[output_cols]

    # Verify every paragraph scored exactly once
    assert len(para_combined) == len(df), \
        f"Mismatch: {len(para_combined)} scored vs {len(df)} total"
    print(f"\n  Verified: every paragraph scored exactly once ({len(para_combined):,} rows)")

    fold_path    = args.output
    summary_path = args.output.replace(".csv", "_summary.csv")
    para_path    = args.output.replace(".csv", "_paragraphs.csv")

    results_df[fold_cols].to_csv(fold_path,    index=False)
    summary.reset_index().to_csv(summary_path, index=False)
    para_combined.to_csv(para_path,            index=False)

    print(f"\nSaved:")
    print(f"  Per-fold metrics:      {fold_path}")
    print(f"  Summary:               {summary_path}")
    print(f"  Per-paragraph scores:  {para_path}")
    print(f"  (logreg_prob: 0.0 = likely original, 1.0 = likely translated)")


if __name__ == "__main__":
    main()