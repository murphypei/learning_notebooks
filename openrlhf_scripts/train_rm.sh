# 启用shell的调试模式
set -x

# huggingface 缓存路径
export HF_HOME=/mnt/cephfs2/peichao/NLP/huggingface_cache/

pretrained_model=Qwen/Qwen2.5-0.5B-Instruct
save_path=./checkpoint/Qwen2.5-0.5B-Instruct-rm
dataset=OpenRLHF/preference_dataset_mixture2_and_safe_pku
wandb_tokens=e2425c86d31c5399b1521c9f8bcb94a7f7343e9a

# 读取训练命令存储在traing_commands变量中
read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --pretrain ${pretrained_model} \
   --dataset ${dataset} \
   --save_path ${save_path} \
   --max_len 8192 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --max_epochs 1 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --learning_rate 9e-6 \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb ${wandb_tokens}
EOF
# --bf16 \
# --flash_attn \
# --packing_samples \

deepspeed --module $training_commands
