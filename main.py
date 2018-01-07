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

import model
import loader
from utils import progress_bar
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--init_lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--validate', '-validate', action='store_true', help='apply current network for validating')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
adjust_epoch = 30 # adjust learning rate by the number of epoches

# Data
print('-- Preparing Training Data')

dir1 = './Misaka-Data/train'
dir2 = './Misaka-Data/test'
if not args.validate:
    trainloader = loader.get_loader(dir1)
    testloader = loader.get_loader(dir2)

classes = ('yes', 'no')

# Model
if args.resume or args.validate:
    # Load checkpoint.
    print('-- Resuming From Checkpoint')
    assert os.path.isdir('Checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./Checkpoint/misaka.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('-- Building Model and Allocating Memory')
    net = model.LeNet()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=5e-4)


# Training
def adjust_learning_rate(optimizer, epoch):
    lr_now = args.init_lr * (0.5 ** (epoch // adjust_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_now

def train(epoch):
    print('-- Current Training Epoch %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
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
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.2f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    print('-- Current Testing Epoch %d' %(epoch))
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.2f%% (%d/%d)'
           % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and not args.validate:
        print('-- Got Best Result. Saving Model. Test Accuracy: %.2f%%' %(acc))
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('Checkpoint'):
            os.mkdir('Checkpoint')
        torch.save(state, './Checkpoint/misaka.t7')
        best_acc = acc

import cv2
import sys
import os.path
import time
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

if args.validate:
    #test(start_epoch)
    cascade_file = "lbpcascade_animeface.xml"
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    dir = './Misaka-Data/Frames/'
    filenum = 1
    ok = False
    for filename in os.listdir(dir):
        print('-- File Name: %s'%filename)
        try:
            cascade = cv2.CascadeClassifier(cascade_file)
            image = cv2.imread(dir + filename, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (128, 128))
        except:
            continue
        height, width, _ = image.shape
        if len(faces) == 0:
            continue
        facenum = 1
        crop_center = 0
        for (x, y, w, h) in faces:
            crop_center = 0
            add = w // 4
            if w != h:
                continue
            x_new = x - add
            y_new = y - add
            w_new = w + (2*add)
            h_new = h + (2*add)
            if x_new > width or y_new > height or x_new < 0 or y_new < 0:
                face = image[y:y+h, x:x+w]
                crop_center = x + w//2
            else:
                face = image[y_new:y_new+h_new, x_new:x_new+w_new]
                crop_center = x_new + w_new//2

            if crop_center - height//2 < 0:
                crop_center = height//2
            if crop_center + height//2 > width:
                crop_center = width - height//2

            face = cv2.resize(face, (128, 128))
            face_out = face
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            tmp = [0]
            tmp = torch.LongTensor(tmp)

            a,b,c = face.shape
            input = torch.from_numpy(np.array(face, np.int32, copy=False))
            input = input.transpose(0, 1).transpose(0, 2).contiguous()
            input = input.view(1, 3, b, a)
            input = Variable(input, volatile=True)
            if use_cuda:
                input = input.cuda()

            input = input.float().div(255)
            input = (input - 0.5) / 0.5
            output = net(input)
            _, predicted = torch.max(output.data, 1)
            result = predicted.eq(tmp).cpu().sum()
            facenum += 1
            #torchvision.utils.save_image(loader.denorm(input).data, './test/tmp-%d.jpg'%filenum)
            if result == 0:
                ok = True
                cv2.imwrite('./Misaka-Data/Face-A/image%d-%d.jpg'%(facenum, filenum), face_out)
                break
            else:
                cv2.imwrite('./Misaka-Data/Face-B/image%d-%d.jpg'%(facenum, filenum), face_out)
        
        if ok:
            ok = False
            image = image[0:height, (crop_center-height//2):(crop_center + height//2)]
            cv2.imwrite('./Misaka-Data/W/image-%d.jpg'%filenum, image)
        filenum += 1

if not args.validate:
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)