pwd
date

general_config_args="--config configs_new/mcunet_config.yaml"
usr_group_kl="full_pretrain_imagenet"
logger_args="--logger.save_dir runs/cls/mcunet/cifar10"
data_args="--data.name cifar10 --data.data_dir data/cifar10 --data.train_workers 24 --data.val_workers 24"
trainer_args="--trainer.max_epochs 50"
model_args="--model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes 10 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $logger_args $seed_args"

echo $common_args

# Base
python trainer_cls.py ${common_args} --logger.exp_name base_all_${usr_group_kl} --model.with_grad_filter False --model.num_of_finetune "all"
python trainer_cls.py ${common_args} --logger.exp_name base_l1_${usr_group_kl} --model.with_grad_filter False --model.num_of_finetune 1
python trainer_cls.py ${common_args} --logger.exp_name base_l2_${usr_group_kl} --model.with_grad_filter False --model.num_of_finetune 2
python trainer_cls.py ${common_args} --logger.exp_name base_l3_${usr_group_kl} --model.with_grad_filter False --model.num_of_finetune 3
python trainer_cls.py ${common_args} --logger.exp_name base_l4_${usr_group_kl} --model.with_grad_filter False --model.num_of_finetune 4

# R2
python trainer_cls.py ${common_args} --logger.exp_name filt_all_r2_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune "all" --model.filt_radius 2
python trainer_cls.py ${common_args} --logger.exp_name filt_l1_r2_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 1 --model.filt_radius 2
python trainer_cls.py ${common_args} --logger.exp_name filt_l2_r2_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 2 --model.filt_radius 2
python trainer_cls.py ${common_args} --logger.exp_name filt_l3_r2_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 3 --model.filt_radius 2
python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r2_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 4 --model.filt_radius 2

# R4
python trainer_cls.py ${common_args} --logger.exp_name filt_all_r4_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune "all" --model.filt_radius 4
python trainer_cls.py ${common_args} --logger.exp_name filt_l1_r4_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 1 --model.filt_radius 4
python trainer_cls.py ${common_args} --logger.exp_name filt_l2_r4_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 2 --model.filt_radius 4
python trainer_cls.py ${common_args} --logger.exp_name filt_l3_r4_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 3 --model.filt_radius 4
python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r4_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 4 --model.filt_radius 4

# R7
python trainer_cls.py ${common_args} --logger.exp_name filt_all_r7_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune "all" --model.filt_radius 7
python trainer_cls.py ${common_args} --logger.exp_name filt_l1_r7_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 1 --model.filt_radius 7
python trainer_cls.py ${common_args} --logger.exp_name filt_l2_r7_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 2 --model.filt_radius 7
python trainer_cls.py ${common_args} --logger.exp_name filt_l3_r7_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 3 --model.filt_radius 7
python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r7_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 4 --model.filt_radius 7

