pwd
date

dataset="imagenet"
num_classes="1000"

usr_group_kl=13.10
load_args="--model.load pretrained_ckpts/res34/pretrain_13.10_imagenet/version_0/checkpoints/epoch=155-val-acc=0.780.ckpt"

general_config_args="--config configs/resnet34_config.yaml"
logger_args="--logger.save_dir runs/resnet34/$dataset/gradfilt/r2"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 32 --data.val_workers 32 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy --data.batch_size 64"
trainer_args="--trainer.max_epochs 90 --trainer.gpus 3 --trainer.strategy ddp --trainer.gradient_clip_val 2.0"
model_args="--model.with_grad_filter True --model.filt_radius 2 --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.lr_warmup 4 --model.num_classes $num_classes --model.momentum 0.9 --model.anneling_steps 90 --model.scheduler_interval epoch"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args

# R2 Có 20 lớp conv2d
python trainer_cls.py ${common_args} --logger.exp_name filt_l1_r2_${usr_group_kl} --model.num_of_finetune 1
python trainer_cls.py ${common_args} --logger.exp_name filt_l2_r2_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls.py ${common_args} --logger.exp_name filt_l3_r2_${usr_group_kl} --model.num_of_finetune 3
python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r2_${usr_group_kl} --model.num_of_finetune 4
python trainer_cls.py ${common_args} --logger.exp_name filt_l5_r2_${usr_group_kl} --model.num_of_finetune 5
python trainer_cls.py ${common_args} --logger.exp_name filt_l6_r2_${usr_group_kl} --model.num_of_finetune 6
python trainer_cls.py ${common_args} --logger.exp_name filt_l7_r2_${usr_group_kl} --model.num_of_finetune 7
python trainer_cls.py ${common_args} --logger.exp_name filt_l8_r2_${usr_group_kl} --model.num_of_finetune 8
python trainer_cls.py ${common_args} --logger.exp_name filt_l9_r2_${usr_group_kl} --model.num_of_finetune 9
python trainer_cls.py ${common_args} --logger.exp_name filt_l10_r2_${usr_group_kl} --model.num_of_finetune 10
python trainer_cls.py ${common_args} --logger.exp_name filt_l11_r2_${usr_group_kl} --model.num_of_finetune 11
python trainer_cls.py ${common_args} --logger.exp_name filt_l12_r2_${usr_group_kl} --model.num_of_finetune 12
python trainer_cls.py ${common_args} --logger.exp_name filt_l13_r2_${usr_group_kl} --model.num_of_finetune 13
python trainer_cls.py ${common_args} --logger.exp_name filt_l14_r2_${usr_group_kl} --model.num_of_finetune 14
python trainer_cls.py ${common_args} --logger.exp_name filt_l15_r2_${usr_group_kl} --model.num_of_finetune 15
python trainer_cls.py ${common_args} --logger.exp_name filt_l16_r2_${usr_group_kl} --model.num_of_finetune 16
python trainer_cls.py ${common_args} --logger.exp_name filt_l17_r2_${usr_group_kl} --model.num_of_finetune 17
python trainer_cls.py ${common_args} --logger.exp_name filt_l18_r2_${usr_group_kl} --model.num_of_finetune 18
python trainer_cls.py ${common_args} --logger.exp_name filt_l19_r2_${usr_group_kl} --model.num_of_finetune 19
python trainer_cls.py ${common_args} --logger.exp_name filt_l20_r2_${usr_group_kl} --model.num_of_finetune 20
python trainer_cls.py ${common_args} --logger.exp_name filt_l21_r2_${usr_group_kl} --model.num_of_finetune 21
python trainer_cls.py ${common_args} --logger.exp_name filt_l22_r2_${usr_group_kl} --model.num_of_finetune 22
python trainer_cls.py ${common_args} --logger.exp_name filt_l23_r2_${usr_group_kl} --model.num_of_finetune 23
python trainer_cls.py ${common_args} --logger.exp_name filt_l24_r2_${usr_group_kl} --model.num_of_finetune 24
python trainer_cls.py ${common_args} --logger.exp_name filt_l25_r2_${usr_group_kl} --model.num_of_finetune 25
python trainer_cls.py ${common_args} --logger.exp_name filt_l26_r2_${usr_group_kl} --model.num_of_finetune 26
python trainer_cls.py ${common_args} --logger.exp_name filt_l27_r2_${usr_group_kl} --model.num_of_finetune 27
python trainer_cls.py ${common_args} --logger.exp_name filt_l28_r2_${usr_group_kl} --model.num_of_finetune 28
python trainer_cls.py ${common_args} --logger.exp_name filt_l29_r2_${usr_group_kl} --model.num_of_finetune 29
python trainer_cls.py ${common_args} --logger.exp_name filt_l30_r2_${usr_group_kl} --model.num_of_finetune 30
python trainer_cls.py ${common_args} --logger.exp_name filt_l31_r2_${usr_group_kl} --model.num_of_finetune 31
python trainer_cls.py ${common_args} --logger.exp_name filt_l32_r2_${usr_group_kl} --model.num_of_finetune 32
python trainer_cls.py ${common_args} --logger.exp_name filt_l33_r2_${usr_group_kl} --model.num_of_finetune 33
python trainer_cls.py ${common_args} --logger.exp_name filt_l34_r2_${usr_group_kl} --model.num_of_finetune 34
python trainer_cls.py ${common_args} --logger.exp_name filt_l35_r2_${usr_group_kl} --model.num_of_finetune 35
python trainer_cls.py ${common_args} --logger.exp_name filt_l36_r2_${usr_group_kl} --model.num_of_finetune 36