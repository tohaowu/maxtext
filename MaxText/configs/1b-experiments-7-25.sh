#!/bin/bash
set -e

STEPS=20000

SAVE_ACCUMULATOR=$1
USE_RHS_NOISE_FUNCTION=$2
PRNG_KEY=$3
RUN_NAME=$4

OUTPUT_FILE=gs://mattdavidow-maxtext-br/${RUN_NAME}.txt


export LIBTPU_INIT_ARGS="--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

command="python3 MaxText/train.py MaxText/configs/base.yml \
    steps=${STEPS} per_device_batch_size=4 learning_rate=0.001 warmup_steps=2000 enable_profiler=false enable_checkpointing=false \
    enable_dropout=false enable_data_shuffling=false run_name=${RUN_NAME}\
    use_int8_training=true metrics_file=metrics.txt\
    remat_policy=full init_prng_key=${PRNG_KEY}\
    gcs_metrics_directory=gs://mattdavidow-maxtext-br/metrics/${RUN_NAME}"

echo "Starting run (${RUN_NAME}) with command: ${command}"
eval ${command}
echo "Finished command"
echo "Now writing to ${OUTPUT_FILE}"
if [[ ${SLICE_ID} -eq 0 && ${WORKER_ID} -eq 0 ]]; then
    gsutil cp metrics.txt ${OUTPUT_FILE}
fi
echo "Done writing to ${OUTPUT_FILE}"