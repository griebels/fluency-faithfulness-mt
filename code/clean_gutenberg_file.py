"""
clean_gutenberg_file.py

Takes a txt file from a Gutenberg text.
Splits on double newline (\n\n),
cleans unicode data,
counts length of words in paragraph,
omits blank paragraphs,
omits paragraphs with > 60% punctuation (non-alphanumeric),
omits any paragraph with "project gutenberg",
returns a CSV with one paragraph per row.

Headers = 
book_id : gutenberg id,
source: pg, 
para_id : paragraph index,
gutenberg_para : paragraph text

"""
import csv
import unicodedata
import argparse
#import os
import re
from pathlib import Path

def openTxt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        myfile = f.read()
        return myfile
    
def splitbook(bookfile):
    #booksplit = re.split(r"\n\s*\n", bookfile)
    booksplit = bookfile.split("\n\n")
    return booksplit

def normalizeBook(paragraph):
    replacements = {
        '\u2018': "'",   
        '\u2019': "'",   
        '\u201C': '"',   
        '\u201D': '"',   
        '\u2013': '-',   
        '\u2014': '-',   
        '\u2026': '...', 
        '\u00A0': ' ',   
    }

    paragraph = paragraph.replace("\n", "")
    for k,v in replacements.items():
        paragraph = paragraph.replace(k,v)
    paragraph = unicodedata.normalize("NFKC", paragraph)
    paragraph = paragraph.strip()
    if paragraph:
        return paragraph


def no_gutenberg_junk(paragraph):
    """
    Returns False if the paragraph:
    - contains the words "project gutenberg"
    - is 90% uppercase (often a marker of Table of Contents) 
    """
    if "project gutenberg" in paragraph.lower():
        return False
    
    words = paragraph.split()
    if not words:
        return False
    
    uppercase_words = sum(
        1 for w in words if w.isupper()
    )
    uppercase_ratio = uppercase_words/len(words)
    if uppercase_ratio >0.9:
        return False
    
    alpha_chars = sum(
        1 for c in paragraph if c.isalnum()
    )
    if alpha_chars/len(paragraph) < 0.6:
        return False
    else:
        return True


def saveCSV(bookfile, outputfile, input_filename):
    #book_id = os.path.splitext(os.path.basename(input_filename))[0]
    book_id = input_filename.stem
    with open(outputfile, 'w', newline='', encoding='utf-8') as f:
        fieldnames=['book_id', 'source', 'paragraph_id', 'gutenberg_para', 'gutenberg_para_len']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for para_id, para in enumerate(bookfile):
            writer.writerow({
                'book_id':book_id,
                'source':'pg',
                'paragraph_id': para_id,
                'gutenberg_para': para,
                'gutenberg_para_len': len(para.split()),
                })

def process_one_text(input_txt: Path, output_csv: Path):
    book = openTxt(input_txt)
    bookparts = splitbook(book)

    cleaned_paras = []
    for para in bookparts:
        paraclean = normalizeBook(para)
        if (
            paraclean 
            and no_gutenberg_junk(paraclean) # Checks for front/back matter junk in the para
        ):
            
            cleaned_paras.append(paraclean)
    
    saveCSV(cleaned_paras, output_csv, input_txt)
    return len(cleaned_paras)


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("input_txt")
    parser.add_argument("--input_folder", "--i", required=True)
    parser.add_argument("--output_folder", "--o", required=True)
    #parser.add_argument("output_csv")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_folder.glob("*.txt"))
    if not txt_files:
        raise SystemError(f"No txt files found in {input_folder}")
    
    print(f"Found {len(txt_files)} files.")

    processed = 0
    for txtfile in txt_files:
        out_csv = output_folder / f"{txtfile.stem}.csv"

        n_paras = process_one_text(txtfile, out_csv)
        processed+=1

    print(f"Done. Processed {processed} files.")


if __name__== "__main__":
    main()
