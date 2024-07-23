pwd
date

dataset="imagenet"
num_classes="1000"

usr_group_kl=13.10
load_args="--model.load pretrained_ckpts/mcu/pretrain_13.10_imagenet/version_0/checkpoints/epoch=11-val-acc=0.714.ckpt"

general_config_args="--config configs/mcunet_config.yaml"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var0.9"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 5 --data.val_workers 5 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy --data.batch_size 64"
trainer_args="--trainer.max_epochs 90 --trainer.gradient_clip_val 2.0"
model_args="--model.SVD_var 0.9 --model.with_HOSVD_with_var_compression True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.005 --model.lr_warmup 4 --model.num_classes $num_classes --model.momentum 0.9 --model.anneling_steps 90 --model.scheduler_interval epoch"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args

python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l2_var0.9_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l4_var0.9_${usr_group_kl} --model.num_of_finetune 4