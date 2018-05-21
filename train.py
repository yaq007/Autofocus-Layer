import torch
import torch.nn as nn
import numpy as np
from models import ModelBuilder
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset import TrainDataset
import os
import nibabel as nib
import argparse
from utils import AverageMeter
from distutils.version import LooseVersion
import math

def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()

    model.train()
    for iteration, sample in enumerate(train_loader):
        image = sample['images'].float()
        target = sample['labels'].long()
        image = Variable(image).cuda()
        label = Variable(target).cuda()       
        
        # The dimension of out should be in the dimension of B,C,W,H,D
        # transform the prediction and label
        out = model(image)    
        out = out.permute(0,2,3,4,1).contiguous().view(-1, args.num_classes)
        
        # extract the center part of the labels
        start_index = []
        end_index = []
        for i in range(3):
            start = int((args.crop_size[i] - args.center_size[i])/2)
            start_index.append(start)
            end_index.append(start + args.center_size[i])
        label = label[:, start_index[0]:end_index[0], start_index[1]: end_index[1], start_index[2]: end_index[2]]
        label = label.contiguous().view(-1).cuda()       
        
        loss = criterion(out, label)
        losses.update(loss.data[0],image.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adjust learning rate
        cur_iter = iteration + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizer, cur_iter, args)

        print('   * i {} |  lr: {:.6f} | Training Loss: {losses.avg:.3f}'.format(iteration, args.running_lr, losses=losses))
      
    print('   * EPOCH {epoch} | Training Loss: {losses.avg:.3f}'.format(epoch=epoch, losses=losses))  

def save_checkpoint(state, epoch, args):   
    filename = args.ckpt + '/' + str(epoch) + '_checkpoint.pth.tar'
    print(filename)
    torch.save(state, filename)

def adjust_learning_rate(optimizer, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr

def main(args):
    # import network architecture
    builder = ModelBuilder()
    model = builder.build_net(
            arch=args.id, 
            num_input=args.num_input, 
            num_classes=args.num_classes, 
            num_branches=args.num_branches,
            padding_list=args.padding_list, 
            dilation_list=args.dilation_list)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
    cudnn.benchmark = True

    # collect the number of parameters in the network
    print("------------------------------------------")
    print("Network Architecture of Model %s:" % (args.id))
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul

    print(model)
    print("Number of trainable parameters %d in Model %s" % (num_para, args.id))
    print("------------------------------------------")

    # set the optimizer and loss
    optimizer = optim.RMSprop(model.parameters(), args.lr, alpha=args.alpha, eps=args.eps, weight_decay=args.weight_decay, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))
     
    # loading data
    tf = TrainDataset(train_dir, args)    
    train_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=True)    
    
    print("Start training ...")
    for epoch in range(args.start_epoch + 1, args.num_epochs + 1):
        train(train_loader, model, criterion, optimizer, epoch, args)

        # save models
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, epoch, args)
    
    print("Training Done")  

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.3.0'), \
        'PyTorch>=0.3.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='AFN1', 
                        help='a name for identitying the model. Choose from the following options: AFN1-6, Basic, ASPP_c, ASPP_s.')
    parser.add_argument('--padding_list', default=[0,4,8,12], nargs='+', type=int,
                        help='list of the paddings in the parallel convolutions')
    parser.add_argument('--dilation_list', default=[2,6,10,14], nargs='+', type=int,
                        help='list of the dilation rates in the parallel convolutions')
    parser.add_argument('--num_branches', default=4, type=int,
                        help='the number of parallel convolutions in autofocus layer')

    # Path related arguments
    parser.add_argument('--train_path', default='datalist/train_list.txt',
                        help='text file of the name of training data')
    parser.add_argument('--root_path', default='./',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')

    # Data related arguments
    parser.add_argument('--crop_size', default=[75,75,75], nargs='+', type=int,
                        help='crop size of the input image (int or list)')
    parser.add_argument('--center_size', default=[47,47,47], nargs='+', type=int,
                        help='the corresponding output size of the input image (int or list)')
    parser.add_argument('--num_classes', default=5, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=5, type=int,
                        help='number of input image for each patient plus the mask')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--random_flip', default=True, type=bool,
                        help='if horizontally flip images when training')
    parser.add_argument('--normalization', default=True, type=bool,
                        help='normalizae the data before training')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='if shuffle the data during training')
    parser.add_argument('--mask', default=True, type=bool,
                        help='if have the mask')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=4, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=10, type=int,
                        help='training batch size')
    parser.add_argument('--num_epochs', default=400, type=int,
                        help='epochs for training')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='start learning rate')
    parser.add_argument('--lr_pow', default=0.9, type=float, 
                        help='power in poly to drop learning rate')
    parser.add_argument('--optim', default='RMSprop', help='optimizer')
    parser.add_argument('--alpha', default='0.9', type=float, help='alpha in RMSprop')
    parser.add_argument('--eps', default=10**(-4), type=float, help='eps in RMSprop')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')
    parser.add_argument('--momentum', default=0.6, type=float, help='momentum for RMSprop')
    parser.add_argument('--save_epochs_steps', default=10, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--particular_epoch', default=200, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')
    parser.add_argument('--num_round', default=1, type=int)

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    train_file = open(args.train_path, 'r')
    train_dir = train_file.readlines()

    args.ckpt = os.path.join(args.ckpt, args.id, str(args.num_round))
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_checkpoint.pth.tar'

    args.running_lr = args.lr
    args.epoch_iters = math.ceil(int(len(train_dir)/args.num_input)/args.batch_size)
    args.max_iters = args.epoch_iters * args.num_epochs

    assert len(args.padding_list) == args.num_branches, \
        '# parallel convolutions should be the same as the length of padding list'

    assert len(args.dilation_list) == args.num_branches, \
        '# parallel convolutions should be the same as # dilation rates'

    assert isinstance(args.crop_size, (int, list))
    if isinstance(args.crop_size, int):
        args.crop_size = [args.crop_size, args.crop_size, args.crop_size]

    assert isinstance(args.center_size, (int, list))
    if isinstance(args.center_size, int):
        args.center_size = [args.center_size, args.center_size, args.center_size]

    main(args)
