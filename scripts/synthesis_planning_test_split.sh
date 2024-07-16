python sample.py \
    --input data/synthesis_planning/test_synthesis.csv \
    --output results/test_split.csv \
    --model-path data/trained_weights/original_split.ckpt \
    --num-gpus -1 \
    --num-workers-per-gpu 2 \
    --exhaustiveness 128 \
