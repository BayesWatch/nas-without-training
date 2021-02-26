python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 3 --dataset cifar10
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 3 --dataset cifar100 --data_loc ../cifar100/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 3 --dataset ImageNet16-120 --data_loc ../imagenet16/Imagenet16/


python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_pnas --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_enas --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_darts --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_darts_fix-w-d --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_nasnet --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_amoeba --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_resnet --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_resnext-a --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_resnext-b --batch_size 128 --GPU 3



python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace amoeba_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_amoeba_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_darts_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_nasnet_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_pnas_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_enas_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_resnext-a_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/



python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 3 --dataset cifar100 --data_loc ../cifar100/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 3 --dataset ImageNet16-120 --data_loc ../imagenet16/Imagenet16/

python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench101 --batch_size 128 --GPU 3 --api_loc ../nasbench_only108.tfrecord 

