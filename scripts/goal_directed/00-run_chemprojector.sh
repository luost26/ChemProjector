python sample.py \
    --input data/goal_directed/goal_hard_cwo.csv \
    --output results/goal_directed/projected.csv \
    --model-path data/trained_weights/original_default.ckpt \
    --num-gpus -1 \
    --num-workers-per-gpu 2 \
    --exhaustiveness 64 \
