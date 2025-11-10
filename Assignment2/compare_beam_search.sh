#!/usr/bin/bash -l
#SBATCH --job-name=beam_translate
#SBATCH --partition teaching
#SBATCH --time=12:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment2.out
#SBATCH --error=err_assignment2.err

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Create a sample of 1000 sentences (only once)
head -n 100 ~/shares/cz-en/data/raw/test.cz > cz-en/data/raw/test_sample.cz
head -n 100 ~/shares/cz-en/data/raw/test.en > cz-en/data/raw/test_sample.en

echo "=== Running translation with beam size 1 ==="
python translate.py \
    --cuda \
    --input cz-en/data/raw/test_sample.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path ~/shares/groups/minsky/checkpoint_best.pt\
    --output cz-en/output_beam1.txt \
    --max-len 300 \
    --beam-size 1 \
    --bleu\
    --reference cz-en/data/raw/test_sample.en 

echo "=== Running translation with beam size 5 ==="
python translate.py \
    --cuda \
    --input cz-en/data/raw/test_sample.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path ~/shares/groups/minsky/checkpoint_best.pt\
    --output cz-en/output_beam5.txt \
    --max-len 300 \
    --beam-size 5\
    --bleu\
    --reference cz-en/data/raw/test_sample.en

# Compute BLEU scores
echo "=== Computing BLEU for beam size 1 ==="
sacrebleu cz-en/data/raw/test_sample.en -i cz-en/output_beam1.txt -m bleu -w 4

echo "=== Computing BLEU for beam size 5 ==="
sacrebleu cz-en/data/raw/test_sample.en -i cz-en/output_beam5.txt -m bleu -w 4

echo "=== Done ==="
