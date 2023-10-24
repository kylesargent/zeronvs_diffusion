#!/bin/bash
python main.py \
    -t \
    --base configs/sd-objaverse-finetune-c_concat-256.yaml \
    --gpus "0,1," \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --finetune_from sd-image-conditioned-v2.ckpt \
    --enable_look_for_checkpoints False \
    data.params.train_config.batch_size=48 \
    data.params.val_config.rate=0.025 \
    lightning.trainer.val_check_interval=100000000 \
    model.params.conditioning_config.params.mode='7dof_qs' \
    model.params.conditioning_config.params.embedding_dim=19 \
    lightning.trainer.accumulate_grad_batches=1 \
    lightning.callbacks.image_logger.params.log_first_step=False \
    lightning.modelcheckpoint.params.every_n_train_steps=100000 \
    lightning.callbacks.image_logger.params.batch_frequency=10