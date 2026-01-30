#!/bin/bash
# Clear any existing preloads and set the new one
export LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/tensorflow/python/platform/../../../tensorflow_cpu_aws.libs/libgomp-cc9055c7.so.1.0.0
python3 repo/model/run_model.py