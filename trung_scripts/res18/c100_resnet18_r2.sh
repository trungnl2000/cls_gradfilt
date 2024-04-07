pwd
date

usr_group_kl="full_pretrain_imagenet"
logger_args="--logger.save_dir runs/cls/res18/cifar100"
data_args="--data.name cifar100 --data.data_dir data/cifar100 --data.train_workers 24 --data.val_workers 24"
trainer_args="--trainer.max_epochs 50"
model_args="--model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes 100 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"
common_args="$trainer_args $data_args $model_args $logger_args $seed_args"

echo $common_args

python trainer_cls.py --config trung_configs/cls/res18/filt_last1_r2.yaml ${common_args} --logger.exp_name filt_l1_r2_${usr_group_kl} --model.with_grad_filter True
python trainer_cls.py --config trung_configs/cls/res18/filt_last2_r2.yaml ${common_args} --logger.exp_name filt_l2_r2_${usr_group_kl} --model.with_grad_filter True
python trainer_cls.py --config trung_configs/cls/res18/filt_last3_r2.yaml ${common_args} --logger.exp_name filt_l3_r2_${usr_group_kl} --model.with_grad_filter True
python trainer_cls.py --config trung_configs/cls/res18/filt_last4_r2.yaml ${common_args} --logger.exp_name filt_l4_r2_${usr_group_kl} --model.with_grad_filter True