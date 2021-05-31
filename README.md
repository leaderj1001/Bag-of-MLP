# Bag-of-MLP

## Reference
 - MLP-Mixer: An all-MLP Architecture for Vision, [MLPMixer](https://arxiv.org/pdf/2105.01601v1.pdf)
   - Organization: Google Research, Brain Team
 - ResMLP: Feedforward networks for image classification with data-efficient training, [ResMLP](https://arxiv.org/abs/2105.03404)
   - Organization: Facebook AI, Sorbonne University, Inria
 - gMLP: Pay Attention to MLPs, [gMLP](https://arxiv.org/abs/2105.08050)
   - Organization: Google Research, Brain Team

## Summary
 - MLPMixer
 <img width="500" alt="스크린샷 2021-05-10 오후 10 13 36" src="https://user-images.githubusercontent.com/22078438/117664703-0c77d200-b1dd-11eb-9dcd-498c829520a7.png">
 - ResMLP
 <img width="500" alt="스크린샷 2021-05-10 오후 10 13 51" src="https://user-images.githubusercontent.com/22078438/117664706-0da8ff00-b1dd-11eb-9541-308e76680810.png">
 - gMLP
 <img width="500" alt="스크린샷 2021-05-10 오후 10 13 51" src="https://user-images.githubusercontent.com/22078438/120160982-d48b0a00-c231-11eb-8f13-39c4f3de3cf2.png">


## Experiments
| Model | Dataset | # of layers | Patch | Hidden | Params (M) | Acc (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ResNet50 baseline ([ref](https://github.com/kuangliu/pytorch-cifar)) | CIFAR10 | | | | 23.5M | 93.62 % |
| MLPMixer (S/32) | CIFAR10 | 8 | 32 | 512 | 18.4M | wip |
| ResMLP | CIFAR10 | 12 | 32 | 512 | 26.8M | wip 
| gMLP | CIFAR10 | 20 | 16 | 256 | 19.7M | 85.74 % 

