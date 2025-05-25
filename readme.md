# Grammar Error Correction with FLAN-T5 Models

This repository contains implementations and experiments for grammar error correction using FLAN-T5 models. The project focuses on fine-tuning FLAN-T5 models for grammatical error correction and includes an optimized version using ONNX Runtime for faster inference.

## Models

The project implements grammar error correction using:
- FLAN-T5-Base
- FLAN-T5-Small

The fine-tuned base model is available on Hugging Face Hub:
[beladrheinz/flan-t5-gec_v1](https://huggingface.co/beladrheinz/flan-t5-gec_v1)

## Implementation Details

### Training
- The models are trained on the C4 dataset, coedit dataset and jfleg dataset

### Evaluation
The models are evaluated on CoEdit and FCE (First Certificate in English) datasets. The evaluation process uses:
- ERRANT library for error annotation and evaluation
- BLEU score metrics for overall performance assessment
- M² score for error correction evaluation
- Detailed error type analysis through ERRANT's error classification system

## Dataset Information

The project uses the following datasets:
- C4 Dataset for training
- CoEdit Dataset for training and evaluation
- FCE Dataset for evaluation