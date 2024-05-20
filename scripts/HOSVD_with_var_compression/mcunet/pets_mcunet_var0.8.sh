pwd
date

dataset="pets"
num_classes="37"

general_config_args="--config configs/mcunet_config.yaml"
usr_group_kl="full_pretrain_imagenet"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var0.8"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24" #--data.batch_size 1"
trainer_args="--trainer.max_epochs 50"
model_args="--model.SVD_var 0.8 --model.with_HOSVD_with_var_compression True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.005 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $logger_args $seed_args"

echo $common_args

python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l1_var0.8_${usr_group_kl} --model.num_of_finetune 1 
python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l2_var0.8_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l3_var0.8_${usr_group_kl} --model.num_of_finetune 3
python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l4_var0.8_${usr_group_kl} --model.num_of_finetune 4
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l5_var0.8_${usr_group_kl} --model.num_of_finetune 5
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l6_var0.8_${usr_group_kl} --model.num_of_finetune 6
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l7_var0.8_${usr_group_kl} --model.num_of_finetune 7
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l8_var0.8_${usr_group_kl} --model.num_of_finetune 8
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l9_var0.8_${usr_group_kl} --model.num_of_finetune 9
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l10_var0.8_${usr_group_kl} --model.num_of_finetune 10