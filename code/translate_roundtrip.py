#!/usr/bin/env python3
"""
Batch machine translation using Ollama + LangChain.

Processes files from Par3 dataset (one csv per book with lang name in the filename), 
translating a text column
from the source language (inferred from filename) into a target language.
Uses async to send multiple rows to Ollama simultaneously.

Usage:
    python translate.py --input_dir ./books --output_dir ./books_translated
    python translate.py --input_dir ./books_gemma --output_dir ./books_roundtrip --roundtrip
    python translate.py --input_dir ./books --output_dir ./out --model aya:8b --num_parallel 4

"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm



langMap = {
    "fr": "French",
    "ru" : "Russian",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
    "sv": "Swedish",
    "pt": "Portuguese",
    "es": "Spanish",
    "no": "Norwegian",
    "hu": "Hungarian",
    "it": "Italian",
    "ta": "Tamil",
    "fa": "Persian",
    "nl": "Dutch",
    "cs": "Czech",
    "pl": "Polish",
    "nb": "Norwegian Bokmål",
    "da": "Danish",
    "bn": "Bengali",
    "st": "Southern Sotho",
    "en": "English",
}


filename_regex_forward_transl = re.compile(r"^(?P<stem>.+)_(?P<lang>[a-z]{2,3})$", re.IGNORECASE)
filename_regex_roundtrip_transl = re.compile(r"^(?P<stem>.+)_(?P<lang>[a-z]{2,3})_gemma_anon$", re.IGNORECASE)

prompt_ver = "v1"  # for translategemma prompt style...


def start_ollama_server(num_parallel=1, wait_seconds=5):
    print(f"Starting Ollama server (OLLAMA_NUM_PARALLEL={num_parallel})...")
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": str(num_parallel)}
    subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(wait_seconds)
    print("Ollama server started.")


def pull_model(model):
    print(f"Pulling model '{model}'...")
    result = subprocess.run(["ollama", "pull", model], check=True)
    if result.returncode != 0:
        print(f"WARNING: `ollama pull {model}` exited with code {result.returncode}", file=sys.stderr)


####### TranslateGemma Prompt! Directly from Ollama TranslateGemma#########
def build_translation_prompt(src_code, tgt_code):
    for code in (src_code, tgt_code):
        if code not in langMap:
            raise KeyError(f"Unknown language code '{code}'. Add it to LANG_MAP.")
    src_name, tgt_name = langMap[src_code], langMap[tgt_code]
    return (
        f"You are a professional {src_name} ({src_code}) to {tgt_name} ({tgt_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {src_name} text while adhering "
        f"to {tgt_name} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tgt_name} translation, without any additional explanations or commentary. "
        f"Please translate the following {src_name} text into {tgt_name}:"
    )


def parse_book_and_lang(path, roundtrip):
    pattern = filename_regex_roundtrip_transl if roundtrip else filename_regex_forward_transl
    m = pattern.match(path.stem)
    if not m:
        return None, None
    return m.group("stem"), m.group("lang").lower()


###### async helpers #####

async def translate_row(chain, question, text):
    try:
        return await chain.ainvoke({"question": question, "text": text})
    except Exception as e:
        return f"ERROR: {e}"


async def translate_batch(rows, question, chain, max_concurrent=8):
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(i, text):
        async with sem:
            if pd.isna(text) or str(text).strip() == "":
                return i, ""
            return i, await translate_row(chain, question, str(text))

    return await asyncio.gather(*[bounded(i, text) for i, text in rows])


async def translate_one_file(input_path, output_dir, chain, model_name,
                              text_column, out_col, chunk_size, num_parallel, roundtrip):
    book_stem, lang_code = parse_book_and_lang(input_path, roundtrip)
    if not book_stem:
        print(f"SKIP (filename not parseable): {input_path.name}")
        return

    if roundtrip:
        src_code, tgt_code = "en", lang_code
        out_csv = output_dir / f"{book_stem}_{lang_code}_gemma_roundtrip.csv"
        progress_json = output_dir / f"{book_stem}_{lang_code}_gemma_roundtrip.progress.json"
        meta = {"rt_src_lang_code": "en", "rt_src_lang_name": "English",
                "rt_tgt_lang_code": lang_code, "rt_tgt_lang_name": langMap.get(lang_code, "")}
    else:
        src_code, tgt_code = lang_code, "en"
        out_csv = output_dir / f"{book_stem}_{lang_code}_translated.csv"
        progress_json = output_dir / f"{book_stem}_{lang_code}_translated.progress.json"
        meta = {"src_lang_code": lang_code, "src_lang_name": langMap.get(lang_code, ""),
                "tgt_lang_code": "en", "tgt_lang_name": "English"}

    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        print(f"SKIP (missing column '{text_column}'): {input_path.name}")
        return

    #metadata cols
    for k, v in meta.items():
        df[k] = v
    df["model"] = model_name
    df["prompt_version"] = prompt_ver

    if out_col not in df.columns:
        df[out_col] = ""

    start_i = 0
    if progress_json.exists():
        try:
            prog = json.loads(progress_json.read_text(encoding="utf-8"))
            start_i = int(prog.get("last_completed_row", -1)) + 1
        except Exception:
            start_i = 0

    if out_csv.exists():
        try:
            prev = pd.read_csv(out_csv)
            if out_col in prev.columns and len(prev) == len(df):
                df[out_col] = prev[out_col].fillna("")
                if not progress_json.exists():
                    nonempty = df[out_col].astype(str).str.strip().ne("")
                    if nonempty.any():
                        start_i = int(nonempty[nonempty].index.max()) + 1
        except Exception:
            pass

    if start_i >= len(df):
        print(f"DONE already: {input_path.name}")
        return

    question = build_translation_prompt(src_code, tgt_code)

    mode_label = "roundtrip" if roundtrip else "forward"
    print(f"\nProcessing : {input_path.name}  [{mode_label}: {src_code} → {tgt_code}]")
    print(f"Resume from: row {start_i} / {len(df)}")
    print(f"Concurrency: {num_parallel} parallel requests, chunk size {chunk_size}")

    pending = [
        (i, df.at[i, text_column])
        for i in range(start_i, len(df))
        if str(df.at[i, out_col]).strip() == ""
    ]

    for chunk_start in tqdm(range(0, len(pending), chunk_size), desc=input_path.name):
        chunk = pending[chunk_start : chunk_start + chunk_size]
        results = await translate_batch(chunk, question, chain, max_concurrent=num_parallel)

        for i, translation in results:
            df.at[i, out_col] = translation

        last_i = chunk[-1][0]
        df.to_csv(out_csv, index=False, encoding="utf-8")
        progress_json.write_text(json.dumps({"last_completed_row": last_i}), encoding="utf-8")

    df.to_csv(out_csv, index=False, encoding="utf-8")
    progress_json.write_text(json.dumps({"last_completed_row": len(df) - 1}), encoding="utf-8")

    error_rows = df[df[out_col].astype(str).str.startswith("ERROR:")].index.tolist()
    if error_rows:
        print(f"ERROR: error warning - {len(error_rows)}  errors in {out_csv.name}: rows {error_rows}")
    else:
        print(f"Saved to {out_csv}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-translate CSVs using a locally-served Ollama model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--roundtrip", action="store_true",
                        help="Translate English → target language (expects <book>_<lang>_gemma_anon.csv)")
    parser.add_argument("--model", default="translategemma:4b")
    parser.add_argument("--text_column", default=None,
                        help="Defaults to 'gemma_para' (roundtrip) or 'source_para' (forward)")
    parser.add_argument("--out_col", default=None,
                        help="Defaults to 'gemma_roundtrip_para' (roundtrip) or 'translation' (forward)")
    parser.add_argument("--num_parallel", type=int, default=8,
                        help="Concurrent requests — lower if you run out of VRAM.")
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--skip_pull", action="store_true")
    parser.add_argument("--skip_serve", action="store_true")
    return parser.parse_args()


async def run_all(args, chain, csv_files):
    for path in csv_files:
        try:
            await translate_one_file(
                input_path=path,
                output_dir=args.output_dir,
                chain=chain,
                model_name=args.model,
                text_column=args.text_column,
                out_col=args.out_col,
                chunk_size=args.chunk_size,
                num_parallel=args.num_parallel,
                roundtrip=args.roundtrip,
            )
        except KeyError as e:
            print(f"SKIP (language mapping error): {path.name} → {e}")
        except Exception as e:
            print(f"ERROR on {path.name}: {e}")
            continue

    print("\nAll done.")


def main():
    args = parse_args()
    if args.text_column is None:
        args.text_column = "gemma_para" if args.roundtrip else "source_para"
    if args.out_col is None:
        args.out_col = "gemma_roundtrip_para" if args.roundtrip else "translation"

    if not args.skip_serve:
        start_ollama_server(num_parallel=args.num_parallel)
    if not args.skip_pull:
        pull_model(args.model)

    #langchain/ollama
    template = "{question}\n\n\n{text}"
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=args.model)
    chain = prompt | model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(args.input_dir.glob("*.csv"))

    skip_suffix = "_gemma_roundtrip.csv" if args.roundtrip else "_translated.csv"
    csv_files = [p for p in csv_files if not p.name.endswith(skip_suffix)]

    mode_label = "roundtrip" if args.roundtrip else "forward"
    print(f"\nMode: {mode_label}")
    print(f"Found {len(csv_files)} CSV(s) to process in {args.input_dir}")
    for p in csv_files:
        print(f"  - {p.name}")
    print()

    asyncio.run(run_all(args, chain, csv_files))


if __name__ == "__main__":
    main()
