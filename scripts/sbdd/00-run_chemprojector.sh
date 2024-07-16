python sample.py \
    --input data/sbdd/pocket2mol.csv \
    --output results/sbdd/projected.csv \
    --model-path data/trained_weights/original_default.ckpt \
    --num-gpus -1 \
    --num-workers-per-gpu 2 \
    --exhaustiveness 64 \
