python sample.py \
    --input data/synthesis_planning/enamine_smiles_1k.csv \
    --output results/enamine.csv \
    --model-path data/trained_weights/original_default.ckpt \
    --num-gpus -1 \
    --num-workers-per-gpu 2 \
    --exhaustiveness 128 \
