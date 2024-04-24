#!/bin/bash

rootdir=$1
worker_id=$2
input_dir=$3
basename=$4

cd $rootdir && python -m preprocessing.dedup.worker --worker_id=$worker_id --input_dir=$input_dir --basename=$basename
