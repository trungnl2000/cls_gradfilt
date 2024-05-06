pwd
date

dataset="pets"
num_classes="37"


general_config_args="--config configs/mbv2_config.yaml"
usr_group_kl="full_pretrain_imagenet"
logger_args="--logger.save_dir runs/mbv2/$dataset/gradfilt/r7"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24"
trainer_args="--trainer.max_epochs 50"
model_args="--model.filt_radius 7 --model.with_grad_filter True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $logger_args $seed_args"

echo $common_args

# R7
# python trainer_cls.py ${common_args} --logger.exp_name filt_l1_r7_${usr_group_kl} --model.num_of_finetune 1
# python trainer_cls.py ${common_args} --logger.exp_name filt_l2_r7_${usr_group_kl} --model.num_of_finetune 2
# python trainer_cls.py ${common_args} --logger.exp_name filt_l3_r7_${usr_group_kl} --model.num_of_finetune 3
# python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r7_${usr_group_kl} --model.num_of_finetune 4
# python trainer_cls.py ${common_args} --logger.exp_name filt_l5_r7_${usr_group_kl} --model.num_of_finetune 5
# python trainer_cls.py ${common_args} --logger.exp_name filt_l6_r7_${usr_group_kl} --model.num_of_finetune 6
# python trainer_cls.py ${common_args} --logger.exp_name filt_l7_r7_${usr_group_kl} --model.num_of_finetune 7
# python trainer_cls.py ${common_args} --logger.exp_name filt_l8_r7_${usr_group_kl} --model.num_of_finetune 8
# python trainer_cls.py ${common_args} --logger.exp_name filt_l9_r7_${usr_group_kl} --model.num_of_finetune 9
# python trainer_cls.py ${common_args} --logger.exp_name filt_l10_r7_${usr_group_kl} --model.num_of_finetune 10
# python trainer_cls.py ${common_args} --logger.exp_name filt_l11_r7_${usr_group_kl} --model.num_of_finetune 11
# python trainer_cls.py ${common_args} --logger.exp_name filt_l12_r7_${usr_group_kl} --model.num_of_finetune 12
# python trainer_cls.py ${common_args} --logger.exp_name filt_l13_r7_${usr_group_kl} --model.num_of_finetune 13
python trainer_cls.py ${common_args} --logger.exp_name filt_l14_r7_${usr_group_kl} --model.num_of_finetune 14
python trainer_cls.py ${common_args} --logger.exp_name filt_l15_r7_${usr_group_kl} --model.num_of_finetune 15
python trainer_cls.py ${common_args} --logger.exp_name filt_l16_r7_${usr_group_kl} --model.num_of_finetune 16
# python trainer_cls.py ${common_args} --logger.exp_name filt_l17_r7_${usr_group_kl} --model.num_of_finetune 17
# python trainer_cls.py ${common_args} --logger.exp_name filt_l18_r7_${usr_group_kl} --model.num_of_finetune 18
# python trainer_cls.py ${common_args} --logger.exp_name filt_l19_r7_${usr_group_kl} --model.num_of_finetune 19
# python trainer_cls.py ${common_args} --logger.exp_name filt_l20_r7_${usr_group_kl} --model.num_of_finetune 20
# python trainer_cls.py ${common_args} --logger.exp_name filt_l21_r7_${usr_group_kl} --model.num_of_finetune 21
# python trainer_cls.py ${common_args} --logger.exp_name filt_l22_r7_${usr_group_kl} --model.num_of_finetune 22
# python trainer_cls.py ${common_args} --logger.exp_name filt_l23_r7_${usr_group_kl} --model.num_of_finetune 23
# python trainer_cls.py ${common_args} --logger.exp_name filt_l24_r7_${usr_group_kl} --model.num_of_finetune 24
python trainer_cls.py ${common_args} --logger.exp_name filt_l25_r7_${usr_group_kl} --model.num_of_finetune 25
python trainer_cls.py ${common_args} --logger.exp_name filt_l26_r7_${usr_group_kl} --model.num_of_finetune 26
python trainer_cls.py ${common_args} --logger.exp_name filt_l27_r7_${usr_group_kl} --model.num_of_finetune 27
python trainer_cls.py ${common_args} --logger.exp_name filt_l28_r7_${usr_group_kl} --model.num_of_finetune 28
# python trainer_cls.py ${common_args} --logger.exp_name filt_l29_r7_${usr_group_kl} --model.num_of_finetune 29
# python trainer_cls.py ${common_args} --logger.exp_name filt_l30_r7_${usr_group_kl} --model.num_of_finetune 30
# python trainer_cls.py ${common_args} --logger.exp_name filt_l31_r7_${usr_group_kl} --model.num_of_finetune 31
# python trainer_cls.py ${common_args} --logger.exp_name filt_l32_r7_${usr_group_kl} --model.num_of_finetune 32
# python trainer_cls.py ${common_args} --logger.exp_name filt_l33_r7_${usr_group_kl} --model.num_of_finetune 33
# python trainer_cls.py ${common_args} --logger.exp_name filt_l34_r7_${usr_group_kl} --model.num_of_finetune 34
# python trainer_cls.py ${common_args} --logger.exp_name filt_l35_r7_${usr_group_kl} --model.num_of_finetune 35
# python trainer_cls.py ${common_args} --logger.exp_name filt_l36_r7_${usr_group_kl} --model.num_of_finetune 36
python trainer_cls.py ${common_args} --logger.exp_name filt_l37_r7_${usr_group_kl} --model.num_of_finetune 37
python trainer_cls.py ${common_args} --logger.exp_name filt_l38_r7_${usr_group_kl} --model.num_of_finetune 38
python trainer_cls.py ${common_args} --logger.exp_name filt_l39_r7_${usr_group_kl} --model.num_of_finetune 39
python trainer_cls.py ${common_args} --logger.exp_name filt_l40_r7_${usr_group_kl} --model.num_of_finetune 40
# python trainer_cls.py ${common_args} --logger.exp_name filt_l41_r7_${usr_group_kl} --model.num_of_finetune 41
# python trainer_cls.py ${common_args} --logger.exp_name filt_l42_r7_${usr_group_kl} --model.num_of_finetune 42
# python trainer_cls.py ${common_args} --logger.exp_name filt_l43_r7_${usr_group_kl} --model.num_of_finetune 43
# python trainer_cls.py ${common_args} --logger.exp_name filt_l44_r7_${usr_group_kl} --model.num_of_finetune 44
python trainer_cls.py ${common_args} --logger.exp_name filt_l45_r7_${usr_group_kl} --model.num_of_finetune 45
# python trainer_cls.py ${common_args} --logger.exp_name filt_l46_r7_${usr_group_kl} --model.num_of_finetune 46
# python trainer_cls.py ${common_args} --logger.exp_name filt_l47_r7_${usr_group_kl} --model.num_of_finetune 47
python trainer_cls.py ${common_args} --logger.exp_name filt_l48_r7_${usr_group_kl} --model.num_of_finetune 48
python trainer_cls.py ${common_args} --logger.exp_name filt_l49_r7_${usr_group_kl} --model.num_of_finetune 49
python trainer_cls.py ${common_args} --logger.exp_name filt_l50_r7_${usr_group_kl} --model.num_of_finetune 50
python trainer_cls.py ${common_args} --logger.exp_name filt_l51_r7_${usr_group_kl} --model.num_of_finetune 51
# python trainer_cls.py ${common_args} --logger.exp_name filt_l52_r7_${usr_group_kl} --model.num_of_finetune 52