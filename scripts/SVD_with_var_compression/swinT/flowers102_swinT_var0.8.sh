pwd
date

dataset="flowers102"
num_classes="102"

usr_group_kl="full_pretrained_imagenet"

# usr_group_kl=15.29
# load_args="--model.load pretrained_ckpts/swinT/c10_epoch=24-val-acc=0.809.ckpt"

general_config_args="--config configs/swinT_config.yaml"
logger_args="--logger.save_dir runs/swinT/$dataset/SVD/var0.8"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24" # --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy"
trainer_args="--trainer.max_epochs 50"
model_args="--model.SVD_var 0.8 --model.with_SVD_with_var_compression True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args

python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l1_var0.8_${usr_group_kl} --model.num_of_finetune 1
python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l2_var0.8_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l3_var0.8_${usr_group_kl} --model.num_of_finetune 3
python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l4_var0.8_${usr_group_kl} --model.num_of_finetune 4
# python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l5_var0.8_${usr_group_kl} --model.num_of_finetune 5
# python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l6_var0.8_${usr_group_kl} --model.num_of_finetune 6
# python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l7_var0.8_${usr_group_kl} --model.num_of_finetune 7
# python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l8_var0.8_${usr_group_kl} --model.num_of_finetune 8
# python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l9_var0.8_${usr_group_kl} --model.num_of_finetune 9
# python trainer_cls_linear.py ${common_args} --logger.exp_name SVD_l10_var0.8_${usr_group_kl} --model.num_of_finetune 10