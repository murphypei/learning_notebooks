# 启用shell的调试模式
set -x

# huggingface 缓存路径
export HF_HOME=/mnt/cephfs2/peichao/NLP/huggingface_cache/

pretrained_model=Qwen/Qwen2.5-0.5B-Instruct
save_path=./checkpoint/Qwen2.5-0.5B-Instruct-dpo
dataset=OpenRLHF/preference_dataset_mixture2_and_safe_pku
wandb_tokens=e2425c86d31c5399b1521c9f8bcb94a7f7343e9a

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --pretrain $pretrained_model \
   --dataset ${dataset} \
   --save_path ${save_path} \
   --max_len 8192 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --max_epochs 1 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb ${wandb_tokens}
EOF
# --use_wandb [WANDB_TOKENS] or True (use wandb login command)
# --ipo [for IPO]
# --label_smoothing 0.1 [for cDPO]
# --ref_offload
# --packing_samples
# --nll_loss_coef (Regularization with NLL loss)
# --flash_attn \
# --bf16 \

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
