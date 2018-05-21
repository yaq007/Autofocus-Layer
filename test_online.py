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
from dataset import TestDataset
import SimpleITK as sitk 

# save prediction results in the format of online submission
def visualize_result(name, pred, args):
    _, _, name1, _, name2 = name.split("/")
    _, _, _, _, _, name3, _ = name2.split(".")
    pred = sitk.GetImageFromArray(pred)
    sitk.WriteImage(pred, args.result + "/VSD"+"."+ str(name1) + '.'+ str(name3)+ '.mha')


# compute the number of segments of  the test images
def segment(image, mask, args):
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
    padding_image_shape.insert(0, args.num_input)
    image_pad = np.zeros(padding_image_shape) 
       
    # assign the original images to the padding images
    image_pad[:, start_index[0] : end_index[0], start_index[1] : end_index[1], start_index[2] : end_index[2]] = image[:, start[0]: end[0], start[1]:end[1], start[2]:end[2]]
    mask_pad[start_index[0] : end_index[0], start_index[1] : end_index[1], start_index[2] : end_index[2]] = mask[start[0]: end[0], start[1]:end[1], start[2]:end[2]]
    return image_pad, mask_pad, num_segments, (start_index, end_index), (start, end)
    
def test(test_loader, model, num_segments, args):   
    # switch to evaluate mode
    model.eval()
    h_c, w_c, d_c = args.center_size  
    pred_seg = np.zeros([num_segments[0] * h_c, num_segments[1] * w_c, d_c, args.num_classes])
    
    for i, sample in enumerate(test_loader):
        image = sample['images'].float().cuda()
        image = torch.squeeze(image, 0)

        with torch.no_grad():      
            image = Variable(image)
            # The dimension of out should be in the dimension of B,C,H,W,D
            out = model(image)
            out_size = out.size()[2:]      
            out = out.permute(0,2,3,4,1).contiguous().cuda()                    
            out_data = (out.data).cpu().numpy()          
        
        for j in range(num_segments[0]):
            pred_seg[j*h_c:(j+1)*h_c, i*d_c: (i+1)*d_c, :, :] = out_data[j, :]    

    return pred_seg

def main(args):
    # import network architecture
    builder = ModelBuilder()
    model = builder.build_net(
            arch=args.id, 
            num_input=args.num_input + 1, 
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
    num_images = int(len(test_dir)/args.num_input)
    dice_score = np.zeros([num_images, 3]).astype(float)

    for i in range(num_images):
        # load the images and mask
        im = []
        for j in range(args.num_input):
            direct, _ = test_dir[args.num_input * i + j].split("\n")
            name = direct            
            image = nib.load(args.root_path + direct).get_data()
            image = np.expand_dims(image, axis=0)
            im.append(image)
            if j == 0:
                mask = nib.load(args.root_path + direct + "mask/mask.nii").get_data()
                   
        images = np.concatenate(im, axis=0).astype(float)

        # divide the input images input small image segments
        # return the padding input images which can be divided exactly
        image_pad, mask_pad, num_segments, padding_index, index = segment(images, mask, args)

        # initialize prediction for the whole image as background
        mask_shape = list(mask.shape)
        mask_shape.append(args.num_classes)
        pred = np.zeros(mask_shape)
        pred[:,:,:,0] = 1
            
        # initialize the prediction for a small segmentation as background
        pad_shape = [int(num_segments[k] * args.center_size[k]) for k in range(3)]
        pad_shape.append(args.num_classes)
        pred_pad = np.zeros(pad_shape)  
        pred_pad[:,:,:,0] = 1 

        
        # iterate over the z dimension
        for idz in range(num_segments[2]):
            tf = TestDataset(image_pad, mask_pad, num_segments, idz, args)
            test_loader = DataLoader(tf, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=False)
            pred_seg = test(test_loader, model, num_segments, args)
            pred_pad[:, :, idz*args.center_size[2]:(idz+1)*args.center_size[2], :] = pred_seg        
           
                
        # decide the start and end point in the original image
        for k in range(3):
            if index[0][k] == 0:
                index[0][k] = int(margin[k]/2 - padding_index[0][k])
            else:
                index[0][k] = int(margin[k]/2 + index[0][k])

            index[1][k] = int(min(index[0][k] + num_segments[k] * args.center_size[k], mask.shape[k]))

        dist = [index[1][k] - index[0][k] for k in range(3)]
        pred[index[0][0]:index[1][0], index[0][1]:index[1][1], index[0][2]:index[1][2]] = pred_pad[:dist[0], :dist[1], :dist[2]]
            
        if args.visualize:
            vis = np.argmax(pred, axis=3)
            vis = np.swapaxes(vis, 0, 2).astype(dtype=np.uint8)
            visualize_result(name, vis, args)
           
    print('Evalution Done!')
   
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
    parser.add_argument('--num_input', default=4, type=int,
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
                        help='restore the models from which run'))

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

   











