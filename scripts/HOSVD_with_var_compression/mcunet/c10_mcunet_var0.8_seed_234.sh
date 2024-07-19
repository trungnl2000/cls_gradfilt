pwd
date

dataset="cifar10"
num_classes="10"

usr_group_kl="full_pretrain_imagenet"

# usr_group_kl=15.29
# load_args="--model.load pretrained_ckpts/mcu/pretrain_15.29_cifar10/version_0/checkpoints/epoch=49-val-acc=0.950.ckpt"

general_config_args="--config configs/mcunet_config.yaml"
# logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var0.8"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24" # --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy"
trainer_args="--trainer.max_epochs 50"
model_args="--model.with_HOSVD_with_var_compression True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 234"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $seed_args" #$logger_args"

echo $common_args

var="0.7"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.72"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.74"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.76"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.78"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.8"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.82"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.84"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.86"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.88"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.9"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.92"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.94"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.96"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4
var="0.98"
HOSVD_var="--model.SVD_var $var"
logger_args="--logger.save_dir runs/mcunet/$dataset/HOSVD/var$var"
python trainer_cls.py $common_args $HOSVD_var $logger_args --logger.exp_name HOSVD_l4_var$HOSVD_var_${usr_group_kl} --model.num_of_finetune 4


# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l1_var0.8_${usr_group_kl} --model.num_of_finetune 1 
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l2_var0.8_${usr_group_kl} --model.num_of_finetune 2
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l3_var0.8_${usr_group_kl} --model.num_of_finetune 3
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l4_var0.8_${usr_group_kl} --model.num_of_finetune 4
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l5_var0.8_${usr_group_kl} --model.num_of_finetune 5
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l6_var0.8_${usr_group_kl} --model.num_of_finetune 6
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l7_var0.8_${usr_group_kl} --model.num_of_finetune 7
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l8_var0.8_${usr_group_kl} --model.num_of_finetune 8
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l9_var0.8_${usr_group_kl} --model.num_of_finetune 9
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l10_var0.8_${usr_group_kl} --model.num_of_finetune 10
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l11_var0.8_${usr_group_kl} --model.num_of_finetune 11
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l12_var0.8_${usr_group_kl} --model.num_of_finetune 12
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l13_var0.8_${usr_group_kl} --model.num_of_finetune 13
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l14_var0.8_${usr_group_kl} --model.num_of_finetune 14
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l15_var0.8_${usr_group_kl} --model.num_of_finetune 15
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l16_var0.8_${usr_group_kl} --model.num_of_finetune 16
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l17_var0.8_${usr_group_kl} --model.num_of_finetune 17
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l18_var0.8_${usr_group_kl} --model.num_of_finetune 18
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l19_var0.8_${usr_group_kl} --model.num_of_finetune 19
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l20_var0.8_${usr_group_kl} --model.num_of_finetune 20
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l21_var0.8_${usr_group_kl} --model.num_of_finetune 21
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l22_var0.8_${usr_group_kl} --model.num_of_finetune 22
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l23_var0.8_${usr_group_kl} --model.num_of_finetune 23
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l24_var0.8_${usr_group_kl} --model.num_of_finetune 24
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l25_var0.8_${usr_group_kl} --model.num_of_finetune 25
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l26_var0.8_${usr_group_kl} --model.num_of_finetune 26
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l27_var0.8_${usr_group_kl} --model.num_of_finetune 27
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l28_var0.8_${usr_group_kl} --model.num_of_finetune 28
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l29_var0.8_${usr_group_kl} --model.num_of_finetune 29
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l30_var0.8_${usr_group_kl} --model.num_of_finetune 30
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l31_var0.8_${usr_group_kl} --model.num_of_finetune 31
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l32_var0.8_${usr_group_kl} --model.num_of_finetune 32
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l33_var0.8_${usr_group_kl} --model.num_of_finetune 33
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l34_var0.8_${usr_group_kl} --model.num_of_finetune 34
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l35_var0.8_${usr_group_kl} --model.num_of_finetune 35
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l36_var0.8_${usr_group_kl} --model.num_of_finetune 36
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l37_var0.8_${usr_group_kl} --model.num_of_finetune 37
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l38_var0.8_${usr_group_kl} --model.num_of_finetune 38
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l39_var0.8_${usr_group_kl} --model.num_of_finetune 39
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l40_var0.8_${usr_group_kl} --model.num_of_finetune 40
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l41_var0.8_${usr_group_kl} --model.num_of_finetune 41
# python trainer_cls.py ${common_args} --logger.exp_name HOSVD_l42_var0.8_${usr_group_kl} --model.num_of_finetune 42