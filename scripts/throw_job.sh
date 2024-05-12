#!/bin/bash

rootdir=$1
worker_id=$2
input_dir=$3
log_dir=$4
basename=$5

cd $rootdir && python -m preprocessing.dedup.worker --worker_id=$worker_id --input_dir=$input_dir --log_dir=$log_dir --basename=$basename
