pwd
date

dataset="imagenet"
num_classes="1000"

usr_group_kl=13.10
load_args="--model.load pretrained_ckpts/mbv2/pretrain_13.10_imagenet/version_0/checkpoints/epoch=155-val-acc=0.743.ckpt"

general_config_args="--config configs/mbv2_config.yaml"
logger_args="--logger.save_dir runs/mbv2/$dataset/avg_batch"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 32 --data.val_workers 32 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy --data.batch_size 64"
trainer_args="--trainer.max_epochs 90 --trainer.gradient_clip_val 2.0"
model_args="--model.with_avg_batch True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.lr_warmup 4 --model.num_classes $num_classes --model.momentum 0.9 --model.anneling_steps 90 --model.scheduler_interval epoch"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args

# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l1_${usr_group_kl} --model.num_of_finetune 1
python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l2_${usr_group_kl} --model.num_of_finetune 2
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l3_${usr_group_kl} --model.num_of_finetune 3
python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l4_${usr_group_kl} --model.num_of_finetune 4
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l5_${usr_group_kl} --model.num_of_finetune 5
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l6_${usr_group_kl} --model.num_of_finetune 6
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l7_${usr_group_kl} --model.num_of_finetune 7
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l8_${usr_group_kl} --model.num_of_finetune 8
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l9_${usr_group_kl} --model.num_of_finetune 9
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l10_${usr_group_kl} --model.num_of_finetune 10
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l11_${usr_group_kl} --model.num_of_finetune 11
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l12_${usr_group_kl} --model.num_of_finetune 12
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l13_${usr_group_kl} --model.num_of_finetune 13
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l14_${usr_group_kl} --model.num_of_finetune 14
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l15_${usr_group_kl} --model.num_of_finetune 15
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l16_${usr_group_kl} --model.num_of_finetune 16
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l17_${usr_group_kl} --model.num_of_finetune 17
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l18_${usr_group_kl} --model.num_of_finetune 18
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l19_${usr_group_kl} --model.num_of_finetune 19
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l20_${usr_group_kl} --model.num_of_finetune 20
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l21_${usr_group_kl} --model.num_of_finetune 21
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l22_${usr_group_kl} --model.num_of_finetune 22
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l23_${usr_group_kl} --model.num_of_finetune 23
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l24_${usr_group_kl} --model.num_of_finetune 24
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l25_${usr_group_kl} --model.num_of_finetune 25
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l26_${usr_group_kl} --model.num_of_finetune 26
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l27_${usr_group_kl} --model.num_of_finetune 27
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l28_${usr_group_kl} --model.num_of_finetune 28
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l29_${usr_group_kl} --model.num_of_finetune 29
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l30_${usr_group_kl} --model.num_of_finetune 30
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l31_${usr_group_kl} --model.num_of_finetune 31
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l32_${usr_group_kl} --model.num_of_finetune 32
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l33_${usr_group_kl} --model.num_of_finetune 33
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l34_${usr_group_kl} --model.num_of_finetune 34
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l35_${usr_group_kl} --model.num_of_finetune 35
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l36_${usr_group_kl} --model.num_of_finetune 36
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l37_${usr_group_kl} --model.num_of_finetune 37
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l38_${usr_group_kl} --model.num_of_finetune 38
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l39_${usr_group_kl} --model.num_of_finetune 39
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l40_${usr_group_kl} --model.num_of_finetune 40
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l41_${usr_group_kl} --model.num_of_finetune 41
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l42_${usr_group_kl} --model.num_of_finetune 42
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l43_${usr_group_kl} --model.num_of_finetune 43
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l44_${usr_group_kl} --model.num_of_finetune 44
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l45_${usr_group_kl} --model.num_of_finetune 45
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l46_${usr_group_kl} --model.num_of_finetune 46
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l47_${usr_group_kl} --model.num_of_finetune 47
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l48_${usr_group_kl} --model.num_of_finetune 48
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l49_${usr_group_kl} --model.num_of_finetune 49
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l50_${usr_group_kl} --model.num_of_finetune 50
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l51_${usr_group_kl} --model.num_of_finetune 51
# python trainer_cls.py ${common_args} --logger.exp_name avg_batch_l52_${usr_group_kl} --model.num_of_finetune 52