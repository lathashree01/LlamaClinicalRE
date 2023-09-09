lr=2e-4
lora_rank=4
lora_alpha=16
lora_trainable="q_proj,v_proj,k_proj,o_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

PATH_TO_CONVERTED_WEIGHTS=/rds/general/user/l22/home/llama-hf/

dataset_dir=/rds/general/user/l22/home/git_repos/LlamaClinicalRE/datasets/mimic3_note_clm_data/
data_cache=new_temp_data_cache_dir
per_device_train_batch_size=8
per_device_eval_batch_size=8
gradient_accumulation_steps=8
output_dir=new_output_dir
mycachedir=/rds/general/user/l22/home/llama_2/

deepspeed_config_file=/rds/general/user/l22/home/git_repos/LlamaClinicalRE/src/models/ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 3 src/models/run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${PATH_TO_CONVERTED_WEIGHTS} \
    --tokenizer_name_or_path ${PATH_TO_CONVERTED_WEIGHTS} \
    --cache_dir ${mycachedir} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.01 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 1000 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 256 \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing True\
    --ddp_find_unused_parameters True \
    --report_to wandb 
