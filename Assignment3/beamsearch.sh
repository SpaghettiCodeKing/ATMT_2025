#!/usr/bin/bash -l
#SBATCH --job-name=beam_translate
#SBATCH --partition teaching
#SBATCH --time=12:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_beamsearch.out
#SBATCH --error=err_beamsearch.err

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

echo "=== Running translation with beam size 5 ==="
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path ~/shares/groups/minsky/checkpoint_best.pt\
    --output cz-en/output_beam5.txt \
    --max-len 300 \
    --beam-size 5\
    --bleu\
    --reference ~/shares/cz-en/data/raw/test.en


echo "=== Computing BLEU for beam size 5 ==="
sacrebleu ~/shares/cz-en/data/raw/test.en -i cz-en/output_beam5.txt -m bleu -w 4

echo "=== Done ==="
