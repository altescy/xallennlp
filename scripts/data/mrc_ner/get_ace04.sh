#!/bin/bash

DATASET_DIR=$PWD/data/mrc_ner/ace04
mkdir -p $DATASET_DIR

wget https://wasabisys.com/datasets-altescy/mrc_ner/ace04/train.jsonl -O $DATASET_DIR/train.jsonl
wget https://wasabisys.com/datasets-altescy/mrc_ner/ace04/dev.jsonl -O $DATASET_DIR/dev.jsonl
wget https://wasabisys.com/datasets-altescy/mrc_ner/ace04/test.jsonl -O $DATASET_DIR/test.jsonl
