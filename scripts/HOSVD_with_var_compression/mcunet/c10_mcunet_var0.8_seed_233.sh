pwd
date

dataset="cifar10"
num_classes="10"

usr_group_kl="full_pretrain_imagenet"

# usr_group_kl=15.29
# load_args="--model.load pretrained_ckpts/mcu/pretrain_15.29_cifar10/version_0/checkpoints/epoch=49-val-acc=0.950.ckpt"

general_config_args="--config configs/mcunet_config.yaml"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24" # --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy"
trainer_args="--trainer.max_epochs 50"
model_args="--model.with_HOSVD_with_var_compression True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

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