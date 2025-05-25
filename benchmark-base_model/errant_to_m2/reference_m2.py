import pandas as pd
import subprocess

df = pd.read_csv('validation.csv')

src_list = []
tgt_list = []

for index, row in df.iterrows():
    src_text = str(row['src'])
    if ':' in src_text:
        src_text = src_text.split(':',1)[1]
    src_list.append(src_text)
    tgt_list.append(str(row['tgt']))

with open('input.txt', 'w', encoding='utf-8') as f:
    for src in src_list:
        f.write(f"{src}\n")


with open('correct.txt', 'w', encoding='utf-8') as f:
    for tgt in tgt_list:
        f.write(f"{tgt}\n")

subprocess.run(['errant_parallel', '-orig', 'input.txt', '-cor', 'correct.txt', '-out', 'reference.m2'])