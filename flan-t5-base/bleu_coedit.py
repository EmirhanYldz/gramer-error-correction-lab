import sacrebleu

def calculate_bleu_sacrebleu(reference_file, candidate_file):
    with open(reference_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f.readlines()]
    
    with open(candidate_file, 'r', encoding='utf-8') as f:
        candidates = [line.strip() for line in f.readlines()]
    
    bleu = sacrebleu.corpus_bleu(candidates, [references])
    print(f"BLEU Score: {bleu.score:.2f}")
    return bleu.score

bleu_score = calculate_bleu_sacrebleu('correct.txt', 'generated.txt')