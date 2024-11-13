#!/usr/bin/env bash

# Example:
# bash mixtral_offline_run.sh -r mixtral_int8_kv_192-96-48  -n

# enable profiling using -p option and capture using
# tensorboard --logdir /tmp/tensorboard/

run_name="mixtral_int8_kv_bs_216-108-54"
dry_run=false
skip_warmup=false
test_run=false
enable_profiler=false
performance=true
audit=false
accuracy=false


while getopts "ntspdar:" opt
do
  case "$opt" in
      n ) dry_run=true ;;
      t ) test_run=true ;; 
      s ) skip_warmup=true ;;
      p ) enable_profiler=true ;;
      d ) audit=true ;;
      a ) accuracy=true ;;
      r ) run_name="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
  esac
done


if "$dry_run"; then
    cmd=echo
else
    cmd=''
fi

SKIP_WARMUP_OPTION=""
if "$skip_warmup"; then
    SKIP_WARMUP_OPTION="--skip_warmup"
fi

PROFILER_OPTION=""
if "$enable_profiler"; then
    PROFILER_OPTION="--enable_profile"
fi

if [ -z "$TOKENIZER_PATH" ]; then
  TOKENIZER_PATH=/home/${USER}/maxtext/assets/tokenizer.mistral-v1
fi

BATCH_STR=""
if [ -z "$BATCH_AND_PREFILL_LEN" ];
then
  BATCH_AND_PREFILL_LEN="256,192|512,96|2048,24"
fi

if [ -z "$TOK_OUTLEN_MULTIPLIER" ];
then
  TOK_OUTLEN_MULTIPLIER="2.5"
fi

if [ -z "$MAXENGINE_ARGS" ];
then
  CHECKPOINT="gs://ml-auto-solutions/output/multipod/maxtext/chained_tests_mixtral-8x7b_stable-2024-09-15-04-01-09/unscanned_ckpt/checkpoints/0/items"
  #CHECKPOINT="gs://vipannalla-bkt/checkpoints/quantized/mixtral-8x7b-instruct/"
  BASE_CFG="model_name=mixtral-8x7b tokenizer_path=${TOKENIZER_PATH} load_parameters_path=${CHECKPOINT}"
  QUANT_CFG="quantization=int8 quantize_kvcache=True"
  LAYOUT_CFG="compute_axis_order=0,1,2,3 ar_cache_axis_order=0,1,2,3"
  MOE_CFG="megablox=False capacity_factor=-1"
  MAXENGINE_ARGS="${BASE_CFG} ${QUANT_CFG} ${LAYOUT_CFG} ${MOE_CFG}"
fi
export LOADGEN_RUN_TIMESTAMP=$(TZ=America/Los_Angeles date +%Y%m%d%H%M%S%Z)
export BASEDIR=/home/${USER}/inference
export DATA_DISK_DIR=/home/${USER}/loadgen_run_data
export API_URL=0.0.0.0:9000
if "$test_run"; then
  export DATASET_TYPE=test
  export DATASET_PATH=${DATA_DISK_DIR}/09292024_mixtral_15k_mintoken2_v1.pkl
  export TOTAL_SAMPLE_COUNT=5000
  export USER_CONFIG=user${TOTAL_SAMPLE_COUNT}.conf
else
  export DATASET_TYPE=full
  export DATASET_PATH=${DATA_DISK_DIR}/09292024_mixtral_15k_mintoken2_v1.pkl
  export TOTAL_SAMPLE_COUNT=15000
  export USER_CONFIG=user.conf
fi

export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS

run_loadgen() {

  OUTPUT_LOG_ID=mixtral8x7b-${run_name}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TYPE}_${LOADGEN_RUN_TIMESTAMP}
  OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
  mkdir -p ${OUTPUT_LOG_DIR} && cp ${USER_CONFIG} ${OUTPUT_LOG_DIR}
  OUTPUT_ACCURACY_JSON_PATH=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json


  echo "LOADGEN_RUN_TIMESTAMP: ${LOADGEN_RUN_TIMESTAMP}"
  echo "DATASET_PATH: ${DATASET_PATH}"
  echo "TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
  echo "OUTPUT_LOG_DIR: ${OUTPUT_LOG_DIR}"
  echo "USER_CONFIG: ${USER_CONFIG}"
  echo "BATCH_AND_PREFILL_LEN: ${BATCH_AND_PREFILL_LEN}"
  echo "MAXENGINE_ARGS: ${MAXENGINE_ARGS}"

  ${cmd} python -m offline_mode \
    --mlperf_test_mode=${TEST_MODE} \
	  --input_mode tokenized \
    --output_mode tokenized \
	  --mlperf_conf $BASEDIR/mlperf.conf \
	  --user_conf ${USER_CONFIG} \
	  --audit_conf ${AUDIT_CONF}  \
	  --total_sample_count ${TOTAL_SAMPLE_COUNT} \
	  --dataset_path ${DATASET_PATH} \
    --prefill_lengths_and_batch_sizes ${BATCH_AND_PREFILL_LEN} \
    --maxengine_args "${MAXENGINE_ARGS}" \
	  --output_log_dir ${OUTPUT_LOG_DIR} \
    --tok_outlen_multiplier ${TOK_OUTLEN_MULTIPLIER} \
    ${SKIP_WARMUP_OPTION} ${PROFILER_OPTION} 2>&1 | tee ${OUTPUT_LOG_DIR}/${LOADGEN_RUN_TYPE}_log.log

}

run_loadgen_performance () {
  LOADGEN_RUN_TYPE=offline-performance
  TEST_MODE="performance"
  AUDIT_CONF="no_audit"
  run_loadgen
}

run_loadgen_audit () {
  LOADGEN_RUN_TYPE=offline-audit
  TEST_MODE="performance"
  AUDIT_CONF="$BASEDIR/compliance/nvidia/TEST06/audit.config"
  run_loadgen
}

run_loadgen_accuracy () {
  LOADGEN_RUN_TYPE=offline-accuracy
  TEST_MODE="accuracy"
  AUDIT_CONF="no_audit"
  run_loadgen

  # Eval Run
  if [ -e ${OUTPUT_ACCURACY_JSON_PATH} ]; then
    ${CMD} python3 evaluate-accuracy.py \
      --checkpoint-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
      --mlperf-accuracy-file ${OUTPUT_ACCURACY_JSON_PATH} \
      --dataset-file ${DATASET_PATH} 2>&1 | tee ${OUTPUT_LOG_DIR}/evaluate_offline_accuracy_log.log
  fi
}

performance=true
if "$audit"; then
  performance=false
  echo
  echo "Starting loadgen audit"
  run_loadgen_audit
fi

if "$accuracy"; then
  performance=false
  echo
  echo "Starting loadgen accuracy"
  run_loadgen_accuracy
fi

if "$performance"; then
  echo
  echo "Starting loadgen performance run"
  run_loadgen_performance
fi
