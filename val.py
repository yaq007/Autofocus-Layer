import torch
import torch.nn as nn
import numpy as np
from models import ModelBuilder
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import random
import nibabel as nib
from utils import AverageMeter
from distutils.version import LooseVersion
import argparse
from dataset import ValDataset

# compute the number of segments of  the validation images
def segment(image, mask, label, args):
    # find the left, right, bottom, top, forward, backward limit of the mask   
    boundary = np.nonzero(mask)
    boundary = [np.unique(boundary[i]) for i in range(3)]
    limit = [(min(boundary[i]), max(boundary[i])) for i in range(3)]
        
    # compute the number of image segments in an image 
    num_segments = [int(np.ceil((limit[i][1] - limit[i][0]) / args.center_size[i])) for i in range(3)]   

    # compute the margin and the shape of the padding image 
    margin = [args.crop_size[i] - args.center_size[i] for i in range(3)]
    padding_image_shape = [num_segments[i] * args.center_size[i] + margin[i] for i in range(3)]  
    
    # start_index is corresponding to the location in the new padding images
    start_index = [limit[i][0] - int(margin[i]/2) for i in range(3)]
    start_index = [int(-index) if index < 0 else 0 for index in start_index]
    
    # start and end is corresponding to the location in the original images
    start = [int(max(limit[i][0] - int(margin[i]/2), 0)) for i in range(3)]
    end = [int(min(start[i] + padding_image_shape[i], mask.shape[i])) for i in range(3)]

    # compute the end_index corresponding to the new padding images
    end_index =[int(start_index[i] + end[i] - start[i]) for i in range(3)]
      
    # initialize the padding images
    size = [start_index[i] if start_index[i] > 0 and end[i] < mask.shape[i] else 0 for i in range(3)]
    if sum(size) != 0:
        padding_image_shape = [sum(x) for x in zip(padding_image_shape, size)]

    mask_pad = np.zeros(padding_image_shape) 
    label_pad = np.zeros(padding_image_shape) 
    padding_image_shape.insert(0, args.num_input - 1)
    image_pad = np.zeros(padding_image_shape) 
       
    # assign the original images to the padding images
    image_pad[:, start_index[0] : end_index[0], start_index[1] : end_index[1], start_index[2] : end_index[2]] = image[:, start[0]: end[0], start[1]:end[1], start[2]:end[2]]
    label_pad[start_index[0] : end_index[0], start_index[1] : end_index[1], start_index[2] : end_index[2]] = label[start[0]: end[0], start[1]:end[1], start[2]:end[2]]
    mask_pad[start_index[0] : end_index[0], start_index[1] : end_index[1], start_index[2] : end_index[2]] = mask[start[0]: end[0], start[1]:end[1], start[2]:end[2]]
    return image_pad, mask_pad, label_pad, num_segments, (start_index, end_index), (start, end)
                
def accuracy(pred, mask, label):
    # columns in score is (# pred, # label, pred and label)
    score = np.zeros([3,3])

    # compute Enhance score (label==4) in the first line
    score[0,0] = np.count_nonzero(pred * mask == 4)
    score[0,1] = np.count_nonzero(label == 4)
    score[0,2] = np.count_nonzero(pred * mask * label == 16)
    
    # compute Core score (label == 1,3,4) in the second line
    pred[pred > 2] = 1
    label[label > 2] = 1
    score[1,0] = np.count_nonzero(pred * mask == 1)
    score[1,1] = np.count_nonzero(label == 1)
    score[1,2] = np.count_nonzero(pred * mask * label == 1)
    
    # compute Whole score (all labels) in the third line
    pred[pred > 1] = 1
    label[label > 1] = 1
    score[2,0] = np.count_nonzero(pred * mask == 1)
    score[2,1] = np.count_nonzero(label == 1)
    score[2,2] = np.count_nonzero(pred * mask * label == 1)
    return score
    
def val(val_loader, model, num_segments, args):   
    # switch to evaluate mode
    model.eval()
    
    # columns in score is (# pred, # label, pred and label)
    # lines in score is (Enhance, Core, Whole)
    score = np.zeros([3, 3])   
    h_c, w_c, d_c = args.center_size    
    pred_seg = np.zeros([num_segments[0] * h_c, num_segments[1] * w_c, d_c, args.num_classes])
    
    for i, sample in enumerate(val_loader):
        image = sample['images'].float().cuda()
        target = sample['labels'].long().cuda()
        mask = sample['mask'].long().cuda()
        
        image = torch.squeeze(image, 0)
        target = torch.squeeze(target, 0)
        mask = torch.squeeze(mask, 0)
        
        with torch.no_grad():      
            image = Variable(image)
            label = Variable(target)
            mask = Variable(mask)
        
            # The dimension of out should be in the dimension of B,C,H,W,D
            out = model(image)
            out_size = out.size()[2:]      
            out = out.permute(0,2,3,4,1).contiguous().cuda()    
                
            out_data = (out.data).cpu().numpy()
           
            # make the prediction
            out = out.view(-1, args.num_classes).cuda()    
            prediction = torch.max(out, 1)[1].cuda().data.squeeze()
        
            # extract the center part of the label and mask
            start = [int((args.crop_size[k] - out_size[k])/2) for k in range(3)]
            end = [sum(x) for x in zip(start, out_size)]
            label = label[:, start[0]: end[0], start[1]: end[1], start[2]: end[2]]
            label = label.contiguous().view(-1)       
            mask = mask[:, start[0]: end[0], start[1]: end[1], start[2]: end[2]]
            mask = mask.contiguous().view(-1)
        
        for j in range(num_segments[0]):
            pred_seg[j*h_c:(j+1)*h_c, i*d_c: (i+1)*d_c, :, :] = out_data[j, :]    

        # compute the dice score
        score += accuracy(prediction.data.cpu().numpy(), mask.data.cpu().numpy(), label.data.cpu().numpy())    

    return score, pred_seg

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
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)           
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            raise Exception("=> No checkpoint found at '{}'".format(args.resume))         
    
    # initialization      
    num_ignore = 0
    margin = [args.crop_size[k] - args.center_size[k] for k in range(3)]
    num_images = int(len(val_dir)/args.num_input)
    dice_score = np.zeros([num_images, 3]).astype(float)

    for i in range(num_images):
        # load the images, label and mask
        im = []
        for j in range(args.num_input):
            direct, _ = val_dir[args.num_input * i + j].split("\n")
            name = direct
            if j < args.num_input - 1:
                image = nib.load(args.root_path + direct + '.gz').get_data()
                image = np.expand_dims(image, axis=0)
                im.append(image)
                if j == 0:
                    mask = nib.load(args.root_path + direct + "/mask.nii.gz").get_data()
            else:
                labels = nib.load(args.root_path + direct + '.gz').get_data()
        
        images = np.concatenate(im, axis=0).astype(float)

        # divide the input images input small image segments
        # return the padding input images which can be divided exactly
        image_pad, mask_pad, label_pad, num_segments, padding_index, index = segment(images, mask, labels, args)

        # initialize prediction for the whole image as background
        labels_shape = list(labels.shape)
        labels_shape.append(args.num_classes)
        pred = np.zeros(labels_shape)
        pred[:,:,:,0] = 1
            
        # initialize the prediction for a small segmentation as background
        pad_shape = [int(num_segments[k] * args.center_size[k]) for k in range(3)]
        pad_shape.append(args.num_classes)
        pred_pad = np.zeros(pad_shape)  
        pred_pad[:,:,:,0] = 1 

        # score_per_image stores the sum of each image
        score_per_image = np.zeros([3, 3])
        # iterate over the z dimension
        for idz in range(num_segments[2]):
            tf = ValDataset(image_pad, label_pad, mask_pad, num_segments, idz, args)
            val_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=False)
            score_seg, pred_seg = val(val_loader, model, num_segments, args)
            pred_pad[:, :, idz*args.center_size[2]:(idz+1)*args.center_size[2], :] = pred_seg        
            score_per_image += score_seg
                
        # decide the start and end point in the original image
        for k in range(3):
            if index[0][k] == 0:
                index[0][k] = int(margin[k]/2 - padding_index[0][k])
            else:
                index[0][k] = int(margin[k]/2 + index[0][k])

            index[1][k] = int(min(index[0][k] + num_segments[k] * args.center_size[k], labels.shape[k]))

        dist = [index[1][k] - index[0][k] for k in range(3)]
        pred[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_pad[:dist[0], :dist[1], :dist[2]]
            
        if np.sum(score_per_image[0,:]) == 0 or np.sum(score_per_image[1,:]) == 0 or np.sum(score_per_image[2,:]) == 0:
            num_ignore += 1
            continue 
        # compute the Enhance, Core and Whole dice score
        dice_score_per = [2 * np.sum(score_per_image[k,2]) / (np.sum(score_per_image[k,0]) + np.sum(score_per_image[k,1])) for k in range(3)]   
        print('Image: %d, Enhance score: %.4f, Core score: %.4f, Whole score: %.4f' % (i, dice_score_per[0], dice_score_per[1], dice_score_per[2]))           
        
        dice_score[i, :] = dice_score_per
        
    count_image = num_images - num_ignore
    dice_score = dice_score[:count_image,:]
    mean_dice = np.mean(dice_score, axis=0)
    std_dice = np.std(dice_score, axis=0)
    print('Evalution Done!')
    print('Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (mean_dice[0], mean_dice[1], mean_dice[2], np.mean(mean_dice)))
    print('Enhance std: %.4f, Core std: %.4f, Whole std: %.4f, Mean Std: %.4f' % (std_dice[0], std_dice[1], std_dice[2], np.mean(std_dice)))                      
    if np.mean(mean_dice) > args.best_mean:
        args.best_epoch = args.epoch_index
        args.best_mean = np.mean(mean_dice)


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
    parser.add_argument('--val_path', default='datalist/test_list.txt',
                        help='txt file of the name of validation data')
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
    parser.add_argument('--normalization', default=True, type=bool,
                        help='normalizae the data before running the validation')
    parser.add_argument('--shuffle', default=False, type=bool,
                        help='if shuffle the data in validation')
    parser.add_argument('--mask', default=True, type=bool,
                        help='if have the mask')

    # validation related arguments
    parser.add_argument('--num_gpus', default=4, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='validation batch size')
    parser.add_argument('--num_epochs', default=20, type=int,
                        help='epochs for validation')
    parser.add_argument('--start_epoch', default=210, type=int,
                        help='epoch to start validation.')
    parser.add_argument('--steps_spoch', default=10, type=int, 
                        help='number of epochs to do the validation')
    parser.add_argument('--best_epoch', default=0, type=int, 
                        help='number of epochs has the best validation result')
    parser.add_argument('--best_mean', default=0.0, type=float, 
                        help='the best mean dice score')
    parser.add_argument('--num_round', default=1, type=int,
                        help='restore the models from which run')
     
    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    val_file = open(args.val_path, 'r')
    val_dir = val_file.readlines()

    args.ckpt = os.path.join(args.ckpt, args.id, str(args.num_round))

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

    # do the validation on a series of models
    for i in range(args.num_epochs):
        args.epoch_index = args.start_epoch + args.steps_spoch * i
        args.resume = args.ckpt + '/' + str(args.epoch_index) + '_checkpoint.pth.tar'
        main(args)

    print('Best Epoch: %d, Best mean epoch: %.4f' % (args.best_epoch, args.best_mean))











