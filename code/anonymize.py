#!/usr/bin/env python3
"""
Anonymization Script

For each specified column, this script produces:
  - <col>_anon_ner  : text with named entities replaced by generic labels (PERSON, LOCATION, ORG, MISC)
  - <col>_anon_pos  : original text is abstracted to Penn Treebank POS tags

To use, first:
pip install flair pandas spacy 
python -m spacy download en_core_web_sm

How to call this script:
- python anonymize_master.py /path/for/inputFolder /path/for/outputFolder -c col1 col2 col3
"""


from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, List
from flair.data import Sentence
from flair.models import SequenceTagger
import spacy


# Named Entity Anonymizer (Flair)
class FlairEntityAnonymizer:
    """Anonymizes named entities using Flair's NER with batching."""

    def __init__(self, model: str = 'ner', batch_size: int = 32):
        print(f"Loading Flair model: {model}")
        self.tagger = SequenceTagger.load(model)
        self.batch_size = batch_size

    def _get_pseudonym(self, entity_type: str) -> str:
        return {'PER': 'PERSON', 'LOC': 'LOCATION', 'ORG': 'ORG', 'MISC': 'MISC'}.get(entity_type, entity_type)

    def _replace_entities(self, text: str, entities: List[Dict]) -> str:
        if not entities:
            return text
        parts = []
        last_end = 0
        for entity in sorted(entities, key=lambda x: x['start']):
            parts.append(text[last_end:entity['start']])
            parts.append(self._get_pseudonym(entity['type']))
            last_end = entity['end']
        parts.append(text[last_end:])
        return ''.join(parts)

    def anonymize_batch(self, texts: List[str]) -> List[str]:
        valid_indices, valid_texts = [], []
        for i, text in enumerate(texts):
            if pd.notna(text) and isinstance(text, str) and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        if not valid_texts:
            return list(texts)

        sentences = [Sentence(t) for t in valid_texts]
        self.tagger.predict(sentences)

        results = list(texts)
        for idx, sentence in zip(valid_indices, sentences):
            entities = [
                {'start': e.start_position, 'end': e.end_position, 'type': e.tag}
                for e in sentence.get_spans('ner')
            ]
            results[idx] = self._replace_entities(valid_texts[valid_indices.index(idx)], entities)
        return results

    def anonymize_column(self, texts: List[str]) -> List[str]:
        total = len(texts)
        results = []
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            results.extend(self.anonymize_batch(batch))
            processed = min(i + self.batch_size, total)
            print(f"  NER: {processed}/{total} rows ({100 * processed // total}%)")
        return results


# Syntactic (POS) Abstractor (with spaCy)
class POSAbstractor:
    """Convert text to fine-grained Penn Treebank POS tags."""

    def __init__(self, model: str = 'en_core_web_sm'):
        print(f"Loading spaCy model: {model}")
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"\nError: spaCy model '{model}' not found.")
            print(f"Please run: python -m spacy download {model}")
            raise

    def abstract_text(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return text
        doc = self.nlp(text)
        return ' '.join(token.tag_ for token in doc)

    def abstract_column(self, texts: List[str]) -> List[str]:
        total = len(texts)
        results = []
        for i, text in enumerate(texts):
            results.append(self.abstract_text(text))
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  POS: {i+1}/{total} rows ({100*(i+1)//total}%)")
        return results


# pipeline
def process_file(
    filepath: str,
    output_dir: str,
    columns: List[str],
    ner: FlairEntityAnonymizer,
    pos: POSAbstractor,
):
    #filename = os.path.basename(filepath)
    p = Path(filepath)
    filename = p.name
    name= p.stem
    ext = p.suffix
    #name, ext = os.path.splitext(filename)
    out_filename = f"{name}_anon{ext}"
    #out_path = os.path.join(output_dir, out_filename)
    out_path = Path(output_dir) / out_filename

    print(f"\n{'='*70}")
    print(f"Processing: {filename}, which will be saved as {out_filename}")
    print(f"{'='*70}")

    df = pd.read_csv(filepath)
    print(f"  {len(df)} rows, columns: {', '.join(df.columns)}")

    # Filter out rows where the gemma_translate went off the rails, and save them to a separate _skipped_rows CSV
    if 'gt_para' in df.columns and 'gemma_para' in df.columns:
        mask = df.apply(lambda row:(
            not isinstance(row['gemma_para'], str) or
            not isinstance(row['gt_para'], str) or
            len(row['gemma_para']) <= len(row['gt_para']) + 500 # checking for a 500 character difference between gt_para and gemma_para. Usually signals a "runaway translation" by gemma.
        ), axis=1)

        dropped_df = df[~mask]
        dropped = len(dropped_df)

        if dropped > 0:
            print(f"Dropping {dropped} number of rows.")
            skipped_dir = Path(output_dir) / ".skipped_rows"
            skipped_dir.mkdir(exist_ok = True)
            skipped_path = skipped_dir / f"{name}_skipped_rows.csv"
            dropped_df.to_csv(skipped_path, index=False)
            print(f"Skipped rows saved to {skipped_path}")
        
        df = df[mask].reset_index(drop=True)
        print(f"{len(df)} rows remaining...")

    for col in columns:
        if col not in df.columns:
            print(f"  Warning: column '{col}' not found — skipping.")
            continue

        ner_col = f"{col}_anon_ner"
        pos_col = f"{col}_anon_pos"

        texts = df[col].tolist()

        # NER abstraction
        ner_texts = ner.anonymize_column(texts)
        df[ner_col] = ner_texts

        # POS abstraction
        df[pos_col] = pos.abstract_column(texts)

    df.to_csv(out_path, index=False)
    print(f"\n Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Anonymize (Flair NER) then POS-abstract (spaCy) CSV columns.',
    )
    parser.add_argument('input_dir',  help='Folder containing input CSV files')
    parser.add_argument('output_dir', help='Folder to write output CSV files')
    parser.add_argument(
        '-c', '--columns', nargs='+',
        default=['gt_para', 'translator_1_para', 'translator_2_para', 'gemma_para'],
        help='Columns to process (default: gt_para translator_1_para translator_2_para gemma_para)',
    )
    parser.add_argument(
        '--ner-model', default='ner', choices=['ner', 'ner-fast', 'ner-large'],
        help='Flair NER model (default: ner)',
    )
    parser.add_argument(
        '--spacy-model', default='en_core_web_sm',
        choices=['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg'],
        help='spaCy model (default: en_core_web_sm)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Flair batch size (default: 32)',
    )

    args = parser.parse_args()

    #if not os.path.isdir(args.input_dir):
    if not Path(args.input_dir).is_dir():
        print(f"Error: input directory '{args.input_dir}' does not exist.")
        return

    #os.makedirs(args.output_dir, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #csv_files = glob.glob(os.path.join(args.input_dir, '*.csv'))
    csv_files = list(Path(args.input_dir).glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{args.input_dir}'.")
        return
    print(f"Found {len(csv_files)} CSV file(s) in '{args.input_dir}'.")

    ner = FlairEntityAnonymizer(model=args.ner_model, batch_size=args.batch_size)
    pos = POSAbstractor(model=args.spacy_model)

    for filepath in sorted(csv_files):
        p = Path(filepath)
        out_path = Path(args.output_dir) / f"{p.stem}_anon{p.suffix}"
        if out_path.exists():
            print(f"{out_path} already exists, so skipping this file.")
            continue

        process_file(filepath, args.output_dir, args.columns, ner, pos)

    print(f"\n{'='*70}")
    print("ALL FILES PROCESSED")
    print(f"Output folder: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()