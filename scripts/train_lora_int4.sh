#!/usr/bin/env bash

export trainer_backend=pl

train_config="./config/train_${trainer_backend}.yaml"

# 强制覆盖配置文件
export train_config=${train_config}
export enable_deepspeed=false
export enable_ptv2=false
export enable_lora=true
export load_in_bit=4

# export CUDA_VISIBLE_DEVICES=1,2,3,4

usage() { echo "Usage: $0 [-m <train|dataset>]" 1>&2; exit 1; }


while getopts m: opt
do
	case "${opt}" in
		m) mode=${OPTARG};;
    *)
      usage
      ;;
	esac
done

if [ "${mode}" != "dataset" ]  && [ "${mode}" != "train" ] ; then
    usage
fi

if [[ "${mode}" == "dataset" ]] ; then
    python ../data_utils.py
    exit 0
fi

if [[ "${trainer_backend}" == "pl" ]] ; then
  # pl 多卡 修改配置文件 devices

    ### 多机多卡训练

  #  例子 3个机器 每个机器 4个卡
  #  修改train.py Trainer num_nodes = 3
  #  MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=0 python train.py
  #  MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=1 python train.py
  #  MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=2 python train.py

   # pl 多卡 修改配置文件 devices

  python ../train.py
elif [[ "${trainer_backend}" == "cl" ]] ; then
  # 多机多卡
  # colossalai run --nproc_per_node 1 --num_nodes 1 --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../train.py

  colossalai run --nproc_per_node 1 --num_nodes 1 ../train.py
else
  # 多机多卡
  # --nproc_per_node=1 nnodes=1 --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT ../train.py
  torchrun --nproc_per_node 1 --nnodes 1 ../train.py
fi