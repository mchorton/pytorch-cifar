# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Pros & cons
Pros:
- Built-in data loading and augmentation, very nice!
- Training is fast, maybe even a little bit faster.
- Very memory efficient!

Cons:
- No progress bar, sad :(
- No built-in log.

## Accuracy
| Model            | Acc.        |
| ------------     | ----------- |
| VGG16            | 92.64%      |
| ResNet18         | 93.02%      |
| ResNet50         | 93.62%      |
| ResNet101        | 93.75%      |
| ResNeXt29(32x4d) | 94.73%      |
| ResNeXt29(2x64d) | 94.82%      |
| DenseNet121      | 95.04%      |

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,150)`
- `0.01` for epoch `[150,250)`
- `0.001` for epoch `[250,350)`

Resume the training with `python main.py --resume --lr=0.01`

## Run Notes
Train the code using:

`python main.py`

Test the code on the GPU using:

`python main.py -t -r`

Test the code on the CPU using:

`python main.py -t -r -d`

When testing the CPU code, note that the GPU may be used to load part of the
CUDA model before work is handed to the GPU. Be sure not to count this as part
of the power used during training. (The amount of time over which this power
usage occurs is small compared to the test time of running on CPU).
