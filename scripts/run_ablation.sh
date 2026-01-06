#!/usr/bin/env bash
set -euo pipefail

# Override these with env vars if needed.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
BASE_OUTPUT="${BASE_OUTPUT:-memory_book_output_kv/ablation}"
SEEDS="${SEEDS:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

BASE_ARGS=(
  --checkpoint tiny_pretrain_output/model_best_eval.pt
  --data_dir book_corpus_output_7e/
  --batch_size 1
  --seq_len 256
  --max_steps 6000
  --n_epochs 1
  --tokenizer_name meta-llama/Meta-Llama-3-8B
  --memory_log_interval 100
  --memory_log_top_k 5
  --memory_log_max_chars 180
  --crystallization_threshold 0.50
  --decay_rate 0.0002
  --min_salience 0.01
  --integration kv_injection
  --cross_attention_top_k 4
  --kv_injection_max_tokens 16
  --retrieval_temperature 1.0
  --retrieval_benefit_salience_weight 0.01
)

CONDITIONS=(
  "none:"
  "refine:--enable_content_refinement"
  "correct:--enable_content_correction"
  "consolidate:--enable_episodic_consolidation"
  "all:--enable_content_refinement --enable_content_correction --enable_episodic_consolidation"
)

for seed in $SEEDS; do
  for entry in "${CONDITIONS[@]}"; do
    name="${entry%%:*}"
    flags="${entry#*:}"
    out_dir="${BASE_OUTPUT}/${name}_seed${seed}"
    log_file="${out_dir}/train.log"
    mkdir -p "$out_dir"

    flag_args=()
    if [[ -n "$flags" ]]; then
      read -r -a flag_args <<< "$flags"
    fi
    extra_args=()
    if [[ -n "${EXTRA_ARGS}" ]]; then
      read -r -a extra_args <<< "${EXTRA_ARGS}"
    fi

    echo "Running ${name} (seed=${seed}) -> ${out_dir}"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      python train_memory_augmented.py \
      "${BASE_ARGS[@]}" \
      --output_dir "$out_dir" \
      --seed "$seed" \
      "${flag_args[@]}" \
      "${extra_args[@]}" \
      2>&1 | tee "$log_file"
  done
done
