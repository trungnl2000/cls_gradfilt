pwd
date

dataset="imagenet"
num_classes="1000"

usr_group_kl=13.10
load_args="--model.load pretrained_ckpts/res18/pretrain_13.10_imagenet/version_0/checkpoints/epoch=179-val-acc=0.753.ckpt"

general_config_args="--config configs/resnet18_config.yaml"
logger_args="--logger.save_dir runs/resnet18/$dataset/gradfilt/r4"
data_args="--data.name $dataset --data.data_dir data/$dataset --data.train_workers 32 --data.val_workers 32 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy --data.batch_size 64"
trainer_args="--trainer.max_epochs 90 --trainer.gpus 3 --trainer.strategy ddp --trainer.gradient_clip_val 2.0"
model_args="--model.with_grad_filter True --model.filt_radius 4 --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.lr_warmup 4 --model.num_classes $num_classes --model.momentum 0.9 --model.anneling_steps 90 --model.scheduler_interval epoch"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args

# r4 Có 20 lớp conv2d
python trainer_cls.py ${common_args} --logger.exp_name filt_l1_r4_${usr_group_kl} --model.num_of_finetune 1
python trainer_cls.py ${common_args} --logger.exp_name filt_l2_r4_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls.py ${common_args} --logger.exp_name filt_l3_r4_${usr_group_kl} --model.num_of_finetune 3
python trainer_cls.py ${common_args} --logger.exp_name filt_l4_r4_${usr_group_kl} --model.num_of_finetune 4
python trainer_cls.py ${common_args} --logger.exp_name filt_l5_r4_${usr_group_kl} --model.num_of_finetune 5
python trainer_cls.py ${common_args} --logger.exp_name filt_l6_r4_${usr_group_kl} --model.num_of_finetune 6
python trainer_cls.py ${common_args} --logger.exp_name filt_l7_r4_${usr_group_kl} --model.num_of_finetune 7
python trainer_cls.py ${common_args} --logger.exp_name filt_l8_r4_${usr_group_kl} --model.num_of_finetune 8
python trainer_cls.py ${common_args} --logger.exp_name filt_l9_r4_${usr_group_kl} --model.num_of_finetune 9
python trainer_cls.py ${common_args} --logger.exp_name filt_l10_r4_${usr_group_kl} --model.num_of_finetune 10
python trainer_cls.py ${common_args} --logger.exp_name filt_l11_r4_${usr_group_kl} --model.num_of_finetune 11
python trainer_cls.py ${common_args} --logger.exp_name filt_l12_r4_${usr_group_kl} --model.num_of_finetune 12
python trainer_cls.py ${common_args} --logger.exp_name filt_l13_r4_${usr_group_kl} --model.num_of_finetune 13
python trainer_cls.py ${common_args} --logger.exp_name filt_l14_r4_${usr_group_kl} --model.num_of_finetune 14
python trainer_cls.py ${common_args} --logger.exp_name filt_l15_r4_${usr_group_kl} --model.num_of_finetune 15
python trainer_cls.py ${common_args} --logger.exp_name filt_l16_r4_${usr_group_kl} --model.num_of_finetune 16
python trainer_cls.py ${common_args} --logger.exp_name filt_l17_r4_${usr_group_kl} --model.num_of_finetune 17
python trainer_cls.py ${common_args} --logger.exp_name filt_l18_r4_${usr_group_kl} --model.num_of_finetune 18
python trainer_cls.py ${common_args} --logger.exp_name filt_l19_r4_${usr_group_kl} --model.num_of_finetune 19
python trainer_cls.py ${common_args} --logger.exp_name filt_l20_r4_${usr_group_kl} --model.num_of_finetune 20