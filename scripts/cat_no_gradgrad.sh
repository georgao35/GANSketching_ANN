CUDA_VISIBLE_DEVICES=0 python train.py \
--name cat_no_gradgrad --batch 1 \
--dataroot_sketch /mnt/disk/zw/data/sketch/photosketch/standing_cat \
--dataroot_image /mnt/disk/zw/data/image/lmdb/cat --l_image 0.7 \
--g_pretrained /mnt/disk/zw/pretrained/stylegan2-cat/netG.pth \
--d_pretrained /mnt/disk/zw/pretrained/stylegan2-cat/netD.pth \
--max_iter 200000 --disable_eval \
--resume_iter 20000 \
--checkpoints_dir /mnt/disk/zw/checkpoint/ \
--photosketch_path /mnt/disk/zw/pretrained/photosketch.pth \
--display_freq 1000 \
2> ./warning/cat_no_gradgrad.txt
