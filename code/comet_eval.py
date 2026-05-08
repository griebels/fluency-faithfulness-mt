"""
Comet-kiwi Batch Scoring 

Processes a folder of CSVs. For each file, scores all present translation
columns against source_para using "wmt22-cometkiwi-da".

Translation columns scored:
    gt_para, translator_1_para, translator_2_para, translator_3_para,
    translator_4_para, translator_5_para, gemma_para

Scores are 0-1 (1 = perfect translation), added as comet_score_<col> columns.
Output files are written to --output_dir with the same filenames.

Usage:
    python comet_eval.py --input_dir translations/ --output_dir scored/

"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from comet import download_model, load_from_checkpoint


default_model      = "Unbabel/wmt22-cometkiwi-da"
default_BatchSize = 16
source_col         = "source_para"

all_the_translation_cols = [
    "gt_para",
    "translator_1_para",
    "translator_2_para",
    "translator_3_para",
    "translator_4_para",
    "translator_5_para",
    "gemma_para",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COMET QE batch scorer for translation CSVs.")
    parser.add_argument("--input_dir",  required=True,  help="Folder containing input CSVs.")
    parser.add_argument("--output_dir", required=True,  help="Folder to write scored CSVs.")
    parser.add_argument("--model",      default=default_model,
                        help=f"COMET QE model name (default: {default_model}).")
    parser.add_argument("--batch_size", type=int, default=default_BatchSize,
                        help=f"Prediction batch size (default: {default_BatchSize}).")
    return parser.parse_args()


def detect_device():
    if torch.backends.mps.is_available():
        print("  Apple Silicon MPS detected.")
        return "mps"
    elif torch.cuda.is_available():
        print("  CUDA GPU detected.")
        return "cuda"
    else:
        print("  No GPU found, must use CPU.")
        return "cpu"


def get_cols_to_score(df):
    """
    Check all_the_translataion_cols against the actual columns in df.
    """
    present = [col for col in all_the_translation_cols if col in df.columns]
    missing = [col for col in all_the_translation_cols if col not in df.columns]
    return present, missing


def build_samples(df, translation_col):
    """Build the list-of-dicts COMET expects for QE scoring (without a 'ref' key)."""
    samples = []
    for _, row in df.iterrows():
        src = str(row[source_col])      if pd.notna(row[source_col])      else ""
        mt  = str(row[translation_col]) if pd.notna(row[translation_col]) else ""
        samples.append({"src": src, "mt": mt})
    return samples


def score_file(
    csv_path: Path,
    model,
    device: str,
    #gpus: int,
    batch_size: int,
    output_dir: Path,
) -> None:
    print(f"\n{'─' * 60}")
    print(f"  File: {csv_path.name}")

    out_path = output_dir / csv_path.name
    if out_path.exists():
        print(f"  [SKIP] {out_path} Output already exists -- skipping.")
        return

    df = pd.read_csv(csv_path)

    if source_col not in df.columns:
        print(f"  [SKIP] '{source_col}' column not found -- skipping file.")
        return

    present_cols, missing_cols = get_cols_to_score(df)

    if not present_cols:
        print(f"  [SKIP] None of the expected translation columns found -- skipping file.")
        return

    print(f"  Rows: {len(df)}")
    print(f"  Scoring:  {present_cols}")
    print(f"  Skipping: {missing_cols}")
    print(f"  {'─' * 40}")

    for col in present_cols:
        score_col = f"comet_score_{col}"
        print(f"  Scoring '{col}' -> '{score_col}' ...")

        samples = build_samples(df, col)
        #output  = model.predict(samples, batch_size=batch_size, gpus=gpus, num_workers=2)
        output = model.predict(samples, batch_size=batch_size, accelerator=device, num_workers=2)

        df[score_col] = output.scores
        print(f"    system={output.system_score:.4f}  "
              f"mean={df[score_col].mean():.4f}  "
              f"min={df[score_col].min():.4f}  "
              f"max={df[score_col].max():.4f}")

    out_path = output_dir / csv_path.name
    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")


def main():
    args = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'. Exiting.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in '{input_dir}'")

    #device, gpus = detect_device()
    device = detect_device()

    print(f"\nLoading model: {args.model}")
    model_path = download_model(args.model)
    model      = load_from_checkpoint(model_path)

    # if device == "mps":
    #     model = model.to("mps")

    print("Model ready.")

    for csv_path in csv_files:
        score_file(
            csv_path   = csv_path,
            model      = model,
            device     = device,
            #gpus       = gpus,
            batch_size = args.batch_size,
            output_dir = output_dir,
        )

    print(f"\n{'=' * 60}")
    print(f"All done! Scored files are all written to '{output_dir}'")


if __name__ == "__main__":
    main()