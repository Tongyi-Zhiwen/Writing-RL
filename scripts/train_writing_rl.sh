export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export DATA_DIR='data'
export HYDRA_FULL_ERROR=1

### api configs
export JUDGE_MODEL="qwen-plus"
export DASHSCOPE_API_KEY='' ### fill your dashscope api key here when using qwen models as JUDGE_MODEL
export OPENAI_API_KEY="" ### fill your openai api key here when using gpt models as JUDGE_MODEL
export REWARD_STRATEGY='position-disadvantage'
export API_PARALLEL=12 ### judge model api parallel; adjust it based on your api rate limit

### wandb
export WANDB_PROJECT='Writing-RL'
export WANDB_API_KEY="" ### wandb login

### training
export PPO_MAX_TOKEN_LEN_PER_GPU=16500
export BASE_MODEL='' ## fill your base model path here; qwen2.5-7b-instruct-writingbench-sft, llama3.1-8b-instruct-writingbench-sft
export EXPERIMENT_NAME=qwen-writing-rl

python scripts/create_index_file.py --experiment $EXPERIMENT_NAME --data_num 5000
echo "Start Training ..."

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=10000 \
    algorithm.adv_estimator=gae \
    reward_model.reward_manager=parallel \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=405 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log