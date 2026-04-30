#!/bin/bash
# usage: bash dart.sh

#config
MODEL_PATH=""
ONTOLOGY_PATH="DART/data/ontology/dbo_cache.json"
DATA_DIR="DART/data/preprocess_table/example_t2dv2"
PREPROCESSED_DIR="DART/data/preprocess_table/example_t2dv2"
OUTPUT_DIR="DART/exp/dart_full"
RETRIEVAL_PKL=""

## retrieve
mkdir -p ${OUTPUT_DIR}


python DART/src/cta/run_retrieve.py \
    --ontology_path      ${ONTOLOGY_PATH} \
    --data_dir      ${DATA_DIR} \
    --output_pkl    ${RETRIEVAL_PKL} \
    --model_path    ${MODEL_PATH} 

if [ $? -ne 0 ]; then
    echo "ERROR: Retrieval failed. Stopping."
    exit 1
fi

## rerank
python DART/src/cta/run_dart.py \
    --ontology ${ONTOLOGY_PATH} \
    --retrieval_pkl ${RETRIEVAL_PKL} \
    --preprocessed_dir ${PREPROCESSED_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_PATH}

if [ $? -ne 0 ]; then
    echo "ERROR: Reranking failed. Stopping."
    exit 1
fi
