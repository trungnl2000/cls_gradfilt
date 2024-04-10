pwd
date

usr_group_kl="full_pretrain_imagenet"
logger_args="--logger.save_dir runs/cls/mbv2/cifar10"
data_args="--data.name cifar10 --data.data_dir data/cifar10 --data.train_workers 24 --data.val_workers 24"
trainer_args="--trainer.max_epochs 50"
model_args="--model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes 10 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"
common_args="$trainer_args $data_args $model_args $logger_args $seed_args"

echo $common_args

python trainer_cls.py --config trung_configs/cls/mbv2/filt_last1_r7.yaml ${common_args} --logger.exp_name base_l1_${usr_group_kl} --model.with_grad_filter False
python trainer_cls.py --config trung_configs/cls/mbv2/filt_last2_r7.yaml ${common_args} --logger.exp_name base_l2_${usr_group_kl} --model.with_grad_filter False
python trainer_cls.py --config trung_configs/cls/mbv2/filt_last3_r7.yaml ${common_args} --logger.exp_name base_l3_${usr_group_kl} --model.with_grad_filter False
python trainer_cls.py --config trung_configs/cls/mbv2/filt_last4_r7.yaml ${common_args} --logger.exp_name base_l4_${usr_group_kl} --model.with_grad_filter False
