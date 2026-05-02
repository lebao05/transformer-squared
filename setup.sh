#!/bin/bash
# Cài dependencies
pip install --upgrade pip
pip install pybind11
pip install fasttext-wheel
pip install pyarrow==14.0.2
pip install accelerate==0.27.2 datasets==2.14.6 einops==0.7.0 evaluate==0.4.1 \
    pandas==2.1.2 torch==2.3.1 torchvision==0.18.1 \
    transformers==4.43.1 trl==0.8.6 vllm==0.5.3.post1 \
    hydra-core==1.3.2 peft fire matplotlib \
    "evalplus @ git+https://github.com/evalplus/evalplus@1895d2f6aa8895044a7cf69defc24bd57695e885"

# Cài fishfarm
cd evaluation/fishfarm && pip install -e . && cd ../..

echo "Done!"