# 启用shell的调试模式
set -x

# huggingface 缓存路径
export HF_HOME=/mnt/cephfs2/peichao/NLP/huggingface_cache/

pretrained_model=Qwen/Qwen2.5-0.5B-Instruct
save_path=./checkpoint/Qwen2.5-0.5B-Instruct-sft
dataset=Open-Orca/OpenOrca
wandb_tokens=e2425c86d31c5399b1521c9f8bcb94a7f7343e9a

# 读取训练命令存储在traing_commands变量中
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --pretrain ${pretrained_model} \
   --dataset ${dataset} \
   --save_path ${save_path} \
   --max_len 2048 \
   --input_key question \
   --output_key response \
   --train_batch_size 64 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --max_epochs 1 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb ${wandb_tokens}
EOF
# --packing_samples
# --bf16 \
# --flash_attn \

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
