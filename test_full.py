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
from dataset import ValDataset_full as ValDataset
import SimpleITK as sitk 

# save prediction results in the format of online submission
def visualize_result(name, pred, args):
    _, name1, _, name2 = name.split("/")
    _, _, _, _, _, name3, _ = name2.split(".")
    pred = sitk.GetImageFromArray(pred)
    sitk.WriteImage(pred, args.result + "/VSD"+"."+ str(name1) + '.'+ str(name3)+ '.mha')
                
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
    
def test(test_loader, model, args):   
    # switch to evaluate mode
    model.eval()
    
    # initialization      
    num_ignore = 0
    num_images = int(len(test_dir)/args.num_input)
    # columns in score is (# pred, # label, pred and label)
    # lines in score is (Enhance, Core, Whole)
    dice_score = np.zeros([num_images, 3]).astype(float)
    
    for i, sample in enumerate(test_loader):
        image = sample['images'].float().cuda()
        label = sample['labels'].long().cuda()
        mask = sample['mask'].long().cuda()
        name = sample['name'][0]
        
        with torch.no_grad():      
            image = Variable(image)
            label = Variable(label)
            mask = Variable(mask)

            # The dimension of out should be in the dimension of B,C,H,W,D
            out = model(image)
            out = out.permute(0,2,3,4,1)
            out = torch.max(out, 4)[1]

            # start index corresponding to the model
            start = [14, 14, 14]
            center = out.size()[1:]
            prediction = torch.zeros(label.size())
            prediction[:, start[0]:start[0]+center[0], start[1]: start[1]+center[1], start[2]: start[2]+center[2]] = out
                          
            # make the prediction corresponding to the center part of the image
            prediction = prediction.contiguous().view(-1).cuda()            
            label = label.contiguous().view(-1)                  
            mask = mask.contiguous().view(-1)
        
        # compute the dice score
        score_per_image = accuracy(prediction.data.cpu().numpy(), mask.data.cpu().numpy(), label.data.cpu().numpy()) 

        if np.sum(score_per_image[0,:]) == 0 or np.sum(score_per_image[1,:]) == 0 or np.sum(score_per_image[2,:]) == 0:
            num_ignore += 1
            continue 
        # compute the Enhance, Core and Whole dice score
        dice_score_per = [2 * np.sum(score_per_image[k,2]) / (np.sum(score_per_image[k,0]) + np.sum(score_per_image[k,1])) for k in range(3)]   
        print('Image: %d, Enhance score: %.4f, Core score: %.4f, Whole score: %.4f' % (i, dice_score_per[0], dice_score_per[1], dice_score_per[2]))           
        
        dice_score[i, :] = dice_score_per

        if args.visualize:
            vis = out.data.cpu().numpy()[0]
            vis = np.swapaxes(vis, 0, 2).astype(dtype=np.uint8)
            visualize_result(name, vis, args)
        
    count_image = num_images - num_ignore
    dice_score = dice_score[:count_image,:]
    mean_dice = np.mean(dice_score, axis=0)
    std_dice = np.std(dice_score, axis=0)
    print('Evalution Done!')
    print('Enhance score: %.4f, Core score: %.4f, Whole score: %.4f, Mean Dice score: %.4f' % (mean_dice[0], mean_dice[1], mean_dice[2], np.mean(mean_dice)))
    print('Enhance std: %.4f, Core std: %.4f, Whole std: %.4f, Mean Std: %.4f' % (std_dice[0], std_dice[1], std_dice[2], np.mean(std_dice)))                         

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
    
    tf = ValDataset(test_dir, args)    
    test_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=False)
    test(test_loader, model, args)      
    
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
    parser.add_argument('--test_path', default='datalist/test_list.txt',
                        help='txt file of the name of test data')
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
                        help='normalizae the data before running the test')
    parser.add_argument('--shuffle', default=False, type=bool,
                        help='if shuffle the data in test')
    parser.add_argument('--mask', default=True, type=bool,
                        help='if have the mask')

    # test related arguments
    parser.add_argument('--num_gpus', default=4, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='test batch size')
    parser.add_argument('--test_epoch', default=400, type=int,
                        help='epoch to start test.')  
    parser.add_argument('--visualize', action='store_true',
                        help='save the prediction result as 3D images')
    parser.add_argument('--result', default='./result',
                        help='folder to output prediction results')
    parser.add_argument('--num_round', default=None, type=int,
                        help='restore the models from which run')

    args = parser.parse_args()
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    test_file = open(args.test_path, 'r')
    test_dir = test_file.readlines()

    if not args.num_round:
        args.ckpt = os.path.join(args.ckpt, args.id)
    else:
        args.ckpt = os.path.join(args.ckpt, args.id, str(args.num_round))

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

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

    # do the test on a series of models
    args.resume = args.ckpt + '/' + str(args.test_epoch) + '_checkpoint.pth.tar'
    main(args)

   











