# SG-GAN

In this work, we propose the use of semantic-guided generative adversarial network (SG-GAN) to automatically synthesize visible face images from their thermal counterparts.

## Steps:

1. Prepare your THM/VIS paired data according to the pix2pix; Use aligned dataset mode. 

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

2. Download the VGG identification model and put it under model/checkpoints/.

https://drive.google.com/open?id=1CllD6zqJH28A6cDYgn5MLhlvk591AtHa

3. Find the face parsing model and protcols:

https://1drv.ms/u/s!AhFf7JiY9UVbgVEPTIeu3d7ey1No

https://github.com/cunjian/face_segmentation (face parsing examples)

3. Train the model:

python train.py --dataroot ./datasets/ARL_Thermal --name SG_GAN --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --niter 150 --niter_decay 250 --display_id 0

4. Test the model:

python test.py --dataroot ./datasets/ARL_Thermal --name SG_GAN --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch --how_many 1500 --loadSize 256 --fineSize 256 --which_epoch 200

5. To evaluate the synthesized model, perform the face matching between synthesized VIS and target VIS:

https://github.com/cunjian/face_rec_amsoftmax (AM-Softmax)

https://github.com/cunjian/face_rec_MobileFaceNet (MobileFaceNet+AM)

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{SGGAN2019,
  title={Matching Thermal to Visible Face Images Using a Semantic-Guided Generative Adversarial Network},
  author={Chen, Cunjian and Ross, Arun},
  booktitle={IEEE International Conference on Automatic Face & Gesture Recognition},
  year={2019}
}


## Reference:

1. https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


