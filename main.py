'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *
from utils import progress_bar
from torch.autograd import Variable

def ckpt_name(namehint):
    return "./checkpoint/%s_ckpt.t7" % namehint

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument(
        '--testonly', '-t', action='store_true',
        help='run inference instead of training')
parser.add_argument(
        '--disablecuda', '-d', action='store_true',
        help='do not use CUDA and GPUs; run on CPU only')
parser.add_argument(
        '--net', '-n', default="resnet18", help='resume from checkpoint')
parser.add_argument(
        '--gpus', '-g', default=-1, type=long,
        help='if using gpus, how many to use (-1 = detect number and use all')
parser.add_argument(
        '--split', '-s', default='test', type=str,
        help='the split to when testing: "train" or "test"')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and (not args.disablecuda)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Timing: data loading
raw_input('Press enter to start DATA loading...')
dataload_start = time.time()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Timing: finish data loading and start model loading (this is probably auxillary)
dataload_end = time.time()
raw_input('DATA loading finished. Press enter to start MODEL loading...')
modelload_start = time.time()

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(ckpt_name(args.net))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    netmap = {
            "resnet18": ResNet18,
            "googlenet": GoogLeNet,
            "densenet121": DenseNet121,
            "resnext29_2x64d": ResNeXt29_2x64d,
            "mobilenet": MobileNet}
    if args.net == "vgg":
        net = VGG('VGG19')
    else:
        net = netmap[args.net]()

if use_cuda:
    net.cuda()
    max_gpus = torch.cuda.device_count()
    # NOTE: This kind of checking should be done at argparse time, but
    # whatever...
    assert args.gpus != 0 and args.gpus <= max_gpus, 'Error: Cannot use 0' \
        ' GPUs or more than the device has (%d)' % (max_gpus,)
    # Spec says -1 means detect; we just detect with anything < 0.
    if args.gpus < 0:
        n_gpus = max_gpus
    else:
        n_gpus = args.gpus
    net = torch.nn.DataParallel(net, device_ids=range(n_gpus))
    cudnn.benchmark = True

if args.disablecuda:
    n_gpus = 0  # for consistency in reporting
    net.cpu()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Timing: model finished loading
modelload_end = time.time()
raw_input('MODEL loading finished. Press enter to continue...')

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, **kwargs):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    raw_input('Press enter to start BUILD NET...')
    buildnet_start = time.time()
    # print('---> START [overall]')

    # NOTE: This kind of checking should be done at argparse time, but
    # whatever...
    assert args.split in ['train', 'test'], 'Error: split must be "train" or '\
        '"test". Got: %s' % (args.split,)
    if args.split == 'train':
        loader, split = trainloader, 'train'
    else:
        loader, split = testloader, 'test'

    for batch_idx, (inputs, targets) in enumerate(loader):
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
        # if batch_idx == 0:
        #     print('---> inputs/targets.cuda()')
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)

        # if batch_idx == 0:
        #     print('---> outputs = net(inputs)')
        outputs = net(inputs)
        if batch_idx == 0:
            # print('---> START [inference]')
            buildnet_end = time.time()
            raw_input('BUILD NET. Finished. Press enter to start INFERENCE...')
            inference_start = time.time()
        # if batch_idx == 0:
        #     print('---> loss = criterion(...)')
        loss = criterion(outputs, targets)

        # if batch_idx == 0:
        #     print('---> (computing loss)')
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    inference_end = time.time()
    print('INFERENCE finished.')
    # print('---> FINISH')
    acc = 100.*correct/total

    # Compute output
    mode = 'gpu' if use_cuda else 'cpu'
    dataload_duration = dataload_end - dataload_start
    modelload_duration = modelload_end - modelload_start
    buildnet_duration = buildnet_end - buildnet_start
    inference_duration = inference_end - inference_start

    # More-human-readable output
    print('Split: ', split)
    print('Accuracy: ', acc)
    print('Timing breakdown:')
    print('  - %0.6f -- loading data' % (dataload_duration))
    print('  - %0.6f -- loading model' % (modelload_duration))
    print('  - %0.6f -- building net' % (buildnet_duration))
    print('  - %0.6f -- inference' % (inference_duration))

    # csv output
    print('split,mode,n-gpus,loading-data,loading-model,building-net,inference')
    print('%s,%s,%d,%0.6f,%0.6f,%0.6f,%0.6f' % (
        split,
        mode,
        n_gpus,
        dataload_duration,
        modelload_duration,
        buildnet_duration,
        inference_duration,
    ))

    # Save checkpoint.
    if kwargs["save"] and acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, ckpt_name(args.net))
        best_acc = acc

if args.testonly:
    print("Testing...")
    test(0, save=False)

else:
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch, save=True)
