python sample.py \
    --input data/synthesis_planning/test_chembl.csv \
    --output results/chembl.csv \
    --model-path data/trained_weights/original_default.ckpt \
    --num-gpus -1 \
    --num-workers-per-gpu 2 \
    --exhaustiveness 128 \
