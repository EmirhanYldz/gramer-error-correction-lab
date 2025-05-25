from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import errant
import subprocess

annotator = errant.load("en")

tokenizer = AutoTokenizer.from_pretrained("beladrheinz/flan-t5-gec_v1", use_auth_token=False)
model = AutoModelForSeq2SeqLM.from_pretrained("beladrheinz/flan-t5-gec_v1", use_auth_token=False)

input_sentences = []

def correct_text(text):
    inputs = tokenizer(f"gec: {text}", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with open('input.txt', 'r', encoding='utf-8') as file:
    i = 0
    for line in file:
        input_sentences.append(line.strip())
        print(f"{i+1} -> {len(input_sentences)} -> input")
        i += 1

with open('generated.txt', 'w', encoding='utf-8') as f:
    i = 0
    for sentence in input_sentences:
        corrected = correct_text(sentence)
        f.write(corrected + '\n')
        print(f"{i+1} -> {len(input_sentences)} -> generated")
        i += 1

subprocess.run(['errant_parallel', '-orig', 'input.txt', '-cor', 'generated.txt', '-out', 'result_coedit.m2'])


