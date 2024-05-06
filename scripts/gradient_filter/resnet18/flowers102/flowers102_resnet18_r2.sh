pwd
date

dataset="flowers102"
num_classes="102"

general_config_args="--config configs/resnet18_config.yaml"
usr_group_kl="full_pretrain_imagenet"
logger_args="--logger.save_dir runs/resnet18/$dataset/gradfilt/r2"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24"
trainer_args="--trainer.max_epochs 50"
model_args="--model.filt_radius 2 --model.with_grad_filter True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $logger_args $seed_args"

echo $common_args

# R2 Có 20 lớp conv2d
# python trainer_cls.py ${common_args} --logger.exp_name filt_l1_r2_${usr_group_kl} --model.num_of_finetune 1
# python trainer_cls.py ${common_args} --logger.exp_name filt_l2_r2_${usr_group_kl} --model.num_of_finetune 2
# python trainer_cls.py ${common_args} --logger.exp_name filt_l3_r2_${usr_group_kl} --model.num_of_finetune 3
# python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r2_${usr_group_kl} --model.num_of_finetune 4
# python trainer_cls.py ${common_args} --logger.exp_name filt_l5_r2_${usr_group_kl} --model.num_of_finetune 5
# python trainer_cls.py ${common_args} --logger.exp_name filt_l6_r2_${usr_group_kl} --model.num_of_finetune 6
# python trainer_cls.py ${common_args} --logger.exp_name filt_l7_r2_${usr_group_kl} --model.num_of_finetune 7
# python trainer_cls.py ${common_args} --logger.exp_name filt_l8_r2_${usr_group_kl} --model.num_of_finetune 8
python trainer_cls.py ${common_args} --logger.exp_name filt_l9_r2_${usr_group_kl} --model.num_of_finetune 9
python trainer_cls.py ${common_args} --logger.exp_name filt_l10_r2_${usr_group_kl} --model.num_of_finetune 10
python trainer_cls.py ${common_args} --logger.exp_name filt_l11_r2_${usr_group_kl} --model.num_of_finetune 11
# python trainer_cls.py ${common_args} --logger.exp_name filt_l12_r2_${usr_group_kl} --model.num_of_finetune 12
# python trainer_cls.py ${common_args} --logger.exp_name filt_l13_r2_${usr_group_kl} --model.num_of_finetune 13
python trainer_cls.py ${common_args} --logger.exp_name filt_l14_r2_${usr_group_kl} --model.num_of_finetune 14
python trainer_cls.py ${common_args} --logger.exp_name filt_l15_r2_${usr_group_kl} --model.num_of_finetune 15
python trainer_cls.py ${common_args} --logger.exp_name filt_l16_r2_${usr_group_kl} --model.num_of_finetune 16
python trainer_cls.py ${common_args} --logger.exp_name filt_l17_r2_${usr_group_kl} --model.num_of_finetune 17
python trainer_cls.py ${common_args} --logger.exp_name filt_l18_r2_${usr_group_kl} --model.num_of_finetune 18
python trainer_cls.py ${common_args} --logger.exp_name filt_l19_r2_${usr_group_kl} --model.num_of_finetune 19
# python trainer_cls.py ${common_args} --logger.exp_name filt_l20_r2_${usr_group_kl} --model.num_of_finetune 20
