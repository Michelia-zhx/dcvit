# ViT-Base
python imagenet_inversion.py --bs=50 --do_flip --exp_name="vit_base_inversion" --r_feature=0.01 --arch_name="vit_base_patch16_224" --adi_scale=0.0 --setting_id=1 --lr 0.25 --store_best_images --iter_num 2000 --seed 2023 --gpu_id 0 --beta 0.6

# ViT-Small
python imagenet_inversion.py --bs=50 --do_flip --exp_name="vit_small_inversion" --r_feature=0.01 --arch_name="vit_small_patch16_224" --adi_scale=0.0 --setting_id=1 --lr 0.25 --store_best_images --iter_num 2000 --seed 2023 --gpu_id 1 --beta 0.6

# ViT-Tiny
python imagenet_inversion.py --bs=50 --do_flip --exp_name="vit_tiny_inversion" --r_feature=0.01 --arch_name="vit_tiny_patch16_224" --adi_scale=0.0 --setting_id=1 --lr 0.25 --store_best_images --iter_num 2000 --seed 2023 --gpu_id 0 --beta 0.6

# ViT-Large
python imagenet_inversion.py --bs=50 --do_flip --exp_name="vit_large_inversion" --r_feature=0.01 --arch_name="vit_large_patch16_224" --adi_scale=0.0 --setting_id=1 --lr 0.25 --store_best_images --iter_num 2000 --seed 2023 --gpu_id 0 --beta 0.6

# swin
python imagenet_inversion.py --bs=50 --do_flip --exp_name="swin_base_inversion" --r_feature=0.01 --arch_name="swin_base_patch4_window7_224" --adi_scale=0.0 --setting_id=1 --lr 0.25 --store_best_images --iter_num 2000 --seed 2023 --gpu_id 0

# deit
python imagenet_inversion.py --bs=50 --do_flip --exp_name="deit_base_inversion" --r_feature=0.01 --arch_name="deit_base_patch16_224" --adi_scale=0.0 --setting_id=1 --lr 0.25 --store_best_images --iter_num 2000 --seed 2023 --gpu_id 0

# mobilenetv2
`python imagenet_inversion.py --bs=50 --do_flip --exp_name="mobilenetv2_inversion" --r_feature=0.01 --arch_name="mobilenet_v2" --verifier --adi_scale=0.0 --setting_id=0 --lr 0.25 --store_best_images --iter_num 2000 --seed 2023 --gpu_id 0