pwd
date

dataset="imagenet"
num_classes="1000"

usr_group_kl=13.10
load_args="--model.load pretrained_ckpts/res34/pretrain_13.10_imagenet/version_0/checkpoints/epoch=155-val-acc=0.780.ckpt"

general_config_args="--config configs/resnet34_config.yaml"
logger_args="--logger.save_dir runs/resnet34/$dataset/SVD/var0.8"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 32 --data.val_workers 32 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy --data.batch_size 64"
trainer_args="--trainer.max_epochs 90 --trainer.gpus 3 --trainer.strategy ddp --trainer.gradient_clip_val 2.0"
model_args="--model.SVD_var 0.8 --model.with_SVD_with_var_compression True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.lr_warmup 4 --model.num_classes $num_classes --model.momentum 0.9 --model.anneling_steps 90 --model.scheduler_interval epoch"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args

python trainer_cls.py ${common_args} --logger.exp_name SVD_l1_var0.8_${usr_group_kl} --model.num_of_finetune 1 
python trainer_cls.py ${common_args} --logger.exp_name SVD_l2_var0.8_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls.py ${common_args} --logger.exp_name SVD_l3_var0.8_${usr_group_kl} --model.num_of_finetune 3
python trainer_cls.py ${common_args} --logger.exp_name SVD_l4_var0.8_${usr_group_kl} --model.num_of_finetune 4
python trainer_cls.py ${common_args} --logger.exp_name SVD_l5_var0.8_${usr_group_kl} --model.num_of_finetune 5
python trainer_cls.py ${common_args} --logger.exp_name SVD_l6_var0.8_${usr_group_kl} --model.num_of_finetune 6
python trainer_cls.py ${common_args} --logger.exp_name SVD_l7_var0.8_${usr_group_kl} --model.num_of_finetune 7
python trainer_cls.py ${common_args} --logger.exp_name SVD_l8_var0.8_${usr_group_kl} --model.num_of_finetune 8
python trainer_cls.py ${common_args} --logger.exp_name SVD_l9_var0.8_${usr_group_kl} --model.num_of_finetune 9
python trainer_cls.py ${common_args} --logger.exp_name SVD_l10_var0.8_${usr_group_kl} --model.num_of_finetune 10
python trainer_cls.py ${common_args} --logger.exp_name SVD_l11_var0.8_${usr_group_kl} --model.num_of_finetune 11
python trainer_cls.py ${common_args} --logger.exp_name SVD_l12_var0.8_${usr_group_kl} --model.num_of_finetune 12
python trainer_cls.py ${common_args} --logger.exp_name SVD_l13_var0.8_${usr_group_kl} --model.num_of_finetune 13
python trainer_cls.py ${common_args} --logger.exp_name SVD_l14_var0.8_${usr_group_kl} --model.num_of_finetune 14
python trainer_cls.py ${common_args} --logger.exp_name SVD_l15_var0.8_${usr_group_kl} --model.num_of_finetune 15
python trainer_cls.py ${common_args} --logger.exp_name SVD_l16_var0.8_${usr_group_kl} --model.num_of_finetune 16
python trainer_cls.py ${common_args} --logger.exp_name SVD_l17_var0.8_${usr_group_kl} --model.num_of_finetune 17
python trainer_cls.py ${common_args} --logger.exp_name SVD_l18_var0.8_${usr_group_kl} --model.num_of_finetune 18
python trainer_cls.py ${common_args} --logger.exp_name SVD_l19_var0.8_${usr_group_kl} --model.num_of_finetune 19
python trainer_cls.py ${common_args} --logger.exp_name SVD_l20_var0.8_${usr_group_kl} --model.num_of_finetune 20
python trainer_cls.py ${common_args} --logger.exp_name SVD_l21_var0.8_${usr_group_kl} --model.num_of_finetune 21
python trainer_cls.py ${common_args} --logger.exp_name SVD_l22_var0.8_${usr_group_kl} --model.num_of_finetune 22
python trainer_cls.py ${common_args} --logger.exp_name SVD_l23_var0.8_${usr_group_kl} --model.num_of_finetune 23
python trainer_cls.py ${common_args} --logger.exp_name SVD_l24_var0.8_${usr_group_kl} --model.num_of_finetune 24
python trainer_cls.py ${common_args} --logger.exp_name SVD_l25_var0.8_${usr_group_kl} --model.num_of_finetune 25
python trainer_cls.py ${common_args} --logger.exp_name SVD_l26_var0.8_${usr_group_kl} --model.num_of_finetune 26
python trainer_cls.py ${common_args} --logger.exp_name SVD_l27_var0.8_${usr_group_kl} --model.num_of_finetune 27
python trainer_cls.py ${common_args} --logger.exp_name SVD_l28_var0.8_${usr_group_kl} --model.num_of_finetune 28
python trainer_cls.py ${common_args} --logger.exp_name SVD_l29_var0.8_${usr_group_kl} --model.num_of_finetune 29
python trainer_cls.py ${common_args} --logger.exp_name SVD_l30_var0.8_${usr_group_kl} --model.num_of_finetune 30
python trainer_cls.py ${common_args} --logger.exp_name SVD_l31_var0.8_${usr_group_kl} --model.num_of_finetune 31
python trainer_cls.py ${common_args} --logger.exp_name SVD_l32_var0.8_${usr_group_kl} --model.num_of_finetune 32
python trainer_cls.py ${common_args} --logger.exp_name SVD_l33_var0.8_${usr_group_kl} --model.num_of_finetune 33
python trainer_cls.py ${common_args} --logger.exp_name SVD_l34_var0.8_${usr_group_kl} --model.num_of_finetune 34
python trainer_cls.py ${common_args} --logger.exp_name SVD_l35_var0.8_${usr_group_kl} --model.num_of_finetune 35
python trainer_cls.py ${common_args} --logger.exp_name SVD_l36_var0.8_${usr_group_kl} --model.num_of_finetune 36