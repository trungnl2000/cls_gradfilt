pwd
date

general_config_args="--config configs_new/resnet18_config.yaml"
usr_group_kl="full_pretrain_imagenet"
logger_args="--logger.save_dir runs/cls/resnet18/cifar10"
data_args="--data.name cifar10 --data.data_dir data/cifar10 --data.train_workers 24 --data.val_workers 24"
trainer_args="--trainer.max_epochs 50"
model_args="--model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes 10 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $logger_args $seed_args"

echo $common_args

# R2
python trainer_cls.py ${common_args} --logger.exp_name filt_l3_r2_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 3 --model.filt_radius 2
python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r2_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 4 --model.filt_radius 2
python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r2_${usr_group_kl} --model.with_grad_filter True --model.num_of_finetune 5 --model.filt_radius 2
