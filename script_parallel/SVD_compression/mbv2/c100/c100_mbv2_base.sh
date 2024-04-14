pwd
date

general_config_args="--config configs_new/mbv2_config.yaml"
usr_group_kl="full_pretrain_imagenet"
logger_args="--logger.save_dir runs/cls/mbv2/cifar100/base"
data_args="--data.name cifar100 --data.data_dir data/cifar100 --data.train_workers 24 --data.val_workers 24"
trainer_args="--trainer.max_epochs 50"
model_args="--model.truncated_SVD_k 1 --model.with_SVD_compression False --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes 100 --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $logger_args $seed_args"

echo $common_args

# Base
python trainer_cls.py ${common_args} --logger.exp_name base_l1${usr_group_kl} --model.num_of_finetune 1
python trainer_cls.py ${common_args} --logger.exp_name base_l2${usr_group_kl} --model.num_of_finetune 2
python trainer_cls.py ${common_args} --logger.exp_name base_l3${usr_group_kl} --model.num_of_finetune 3
python trainer_cls.py ${common_args} --logger.exp_name base_l4${usr_group_kl} --model.num_of_finetune 4
python trainer_cls.py ${common_args} --logger.exp_name base_l5${usr_group_kl} --model.num_of_finetune 5
python trainer_cls.py ${common_args} --logger.exp_name base_l6${usr_group_kl} --model.num_of_finetune 6
python trainer_cls.py ${common_args} --logger.exp_name base_l7${usr_group_kl} --model.num_of_finetune 7
python trainer_cls.py ${common_args} --logger.exp_name base_l8${usr_group_kl} --model.num_of_finetune 8
python trainer_cls.py ${common_args} --logger.exp_name base_l9${usr_group_kl} --model.num_of_finetune 9
python trainer_cls.py ${common_args} --logger.exp_name base_l10${usr_group_kl} --model.num_of_finetune 10
python trainer_cls.py ${common_args} --logger.exp_name base_l11${usr_group_kl} --model.num_of_finetune 11
python trainer_cls.py ${common_args} --logger.exp_name base_l12${usr_group_kl} --model.num_of_finetune 12
python trainer_cls.py ${common_args} --logger.exp_name base_l13${usr_group_kl} --model.num_of_finetune 13
python trainer_cls.py ${common_args} --logger.exp_name base_l14${usr_group_kl} --model.num_of_finetune 14
python trainer_cls.py ${common_args} --logger.exp_name base_l15${usr_group_kl} --model.num_of_finetune 15
python trainer_cls.py ${common_args} --logger.exp_name base_l16${usr_group_kl} --model.num_of_finetune 16
python trainer_cls.py ${common_args} --logger.exp_name base_l17${usr_group_kl} --model.num_of_finetune 17
python trainer_cls.py ${common_args} --logger.exp_name base_l18${usr_group_kl} --model.num_of_finetune 18
python trainer_cls.py ${common_args} --logger.exp_name base_l19${usr_group_kl} --model.num_of_finetune 19
python trainer_cls.py ${common_args} --logger.exp_name base_l20${usr_group_kl} --model.num_of_finetune 20
python trainer_cls.py ${common_args} --logger.exp_name base_l21${usr_group_kl} --model.num_of_finetune 21
python trainer_cls.py ${common_args} --logger.exp_name base_l22${usr_group_kl} --model.num_of_finetune 22
python trainer_cls.py ${common_args} --logger.exp_name base_l23${usr_group_kl} --model.num_of_finetune 23
python trainer_cls.py ${common_args} --logger.exp_name base_l24${usr_group_kl} --model.num_of_finetune 24
python trainer_cls.py ${common_args} --logger.exp_name base_l25${usr_group_kl} --model.num_of_finetune 25
python trainer_cls.py ${common_args} --logger.exp_name base_l26${usr_group_kl} --model.num_of_finetune 26
python trainer_cls.py ${common_args} --logger.exp_name base_l27${usr_group_kl} --model.num_of_finetune 27
python trainer_cls.py ${common_args} --logger.exp_name base_l28${usr_group_kl} --model.num_of_finetune 28
python trainer_cls.py ${common_args} --logger.exp_name base_l29${usr_group_kl} --model.num_of_finetune 29
python trainer_cls.py ${common_args} --logger.exp_name base_l30${usr_group_kl} --model.num_of_finetune 30
python trainer_cls.py ${common_args} --logger.exp_name base_l31${usr_group_kl} --model.num_of_finetune 31
python trainer_cls.py ${common_args} --logger.exp_name base_l32${usr_group_kl} --model.num_of_finetune 32
python trainer_cls.py ${common_args} --logger.exp_name base_l33${usr_group_kl} --model.num_of_finetune 33
python trainer_cls.py ${common_args} --logger.exp_name base_l34${usr_group_kl} --model.num_of_finetune 34
python trainer_cls.py ${common_args} --logger.exp_name base_l35${usr_group_kl} --model.num_of_finetune 35
python trainer_cls.py ${common_args} --logger.exp_name base_l36${usr_group_kl} --model.num_of_finetune 36
python trainer_cls.py ${common_args} --logger.exp_name base_l37${usr_group_kl} --model.num_of_finetune 37
python trainer_cls.py ${common_args} --logger.exp_name base_l38${usr_group_kl} --model.num_of_finetune 38
python trainer_cls.py ${common_args} --logger.exp_name base_l39${usr_group_kl} --model.num_of_finetune 39
python trainer_cls.py ${common_args} --logger.exp_name base_l40${usr_group_kl} --model.num_of_finetune 40
python trainer_cls.py ${common_args} --logger.exp_name base_l41${usr_group_kl} --model.num_of_finetune 41
python trainer_cls.py ${common_args} --logger.exp_name base_l42${usr_group_kl} --model.num_of_finetune 42
python trainer_cls.py ${common_args} --logger.exp_name base_l43${usr_group_kl} --model.num_of_finetune 43
python trainer_cls.py ${common_args} --logger.exp_name base_l44${usr_group_kl} --model.num_of_finetune 44
python trainer_cls.py ${common_args} --logger.exp_name base_l45${usr_group_kl} --model.num_of_finetune 45
python trainer_cls.py ${common_args} --logger.exp_name base_l46${usr_group_kl} --model.num_of_finetune 46
python trainer_cls.py ${common_args} --logger.exp_name base_l47${usr_group_kl} --model.num_of_finetune 47
python trainer_cls.py ${common_args} --logger.exp_name base_l48${usr_group_kl} --model.num_of_finetune 48
python trainer_cls.py ${common_args} --logger.exp_name base_l49${usr_group_kl} --model.num_of_finetune 49
python trainer_cls.py ${common_args} --logger.exp_name base_l50${usr_group_kl} --model.num_of_finetune 50
python trainer_cls.py ${common_args} --logger.exp_name base_l51${usr_group_kl} --model.num_of_finetune 51
python trainer_cls.py ${common_args} --logger.exp_name base_l52${usr_group_kl} --model.num_of_finetune 52