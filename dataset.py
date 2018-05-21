from torch.utils.data import Dataset
import numpy as np
import random
import nibabel as nib

class TrainDataset(Dataset):
    def __init__(self, root_dir, args):   
        self.root_dir = root_dir
        self.num_input = args.num_input
        self.length = int(len(self.root_dir)/self.num_input)
        self.crop_size = args.crop_size
        self.random_flip = args.random_flip
        self.root_path = args.root_path
        assert args.mask, "Missing mask as the input"
        assert args.normalization, "You need to do the data normalization before training"
    
    def __len__(self):
        return self.length
   
    def __getitem__(self, idx):
        im = []
        for i in range(self.num_input):
            direct, _ = self.root_dir[self.num_input * idx + i].split("\n")
            if i < self.num_input - 1:
                image = nib.load(self.root_path + direct + '.gz').get_data()
                image = np.expand_dims(image, axis=0)
                im.append(image)
                if i == 0:
                    mask = nib.load(self.root_path + direct + "/mask.nii.gz").get_data()
            else:
                labels = nib.load(self.root_path + direct + '.gz').get_data()
        
        images = np.concatenate(im, axis=0).astype(float)
        
        # images shape: 4 x H x W x D 
        # labels shape: H x W x D 
        sample = {'images': images, 'mask': mask, 'labels':labels}
        transform = RandomCrop(self.crop_size, self.random_flip, self.num_input)
        sample = transform(sample)
        return sample

class ValDataset(Dataset):
    def __init__(self, image, label, mask, num_segments, idz, args):   
        self.images = image
        self.labels = label
        self.mask = mask
        self.numx = num_segments[0]
        self.numy = num_segments[1]
        self.idz = idz
        self.center_size = args.center_size
        self.crop_size = args.crop_size
        self.num_input = args.num_input - 1
        
    def __len__(self):
        return self.numy
   
    def __getitem__(self, idy):
        h, w, d = self.crop_size
        left =  np.arange(self.numx) * self.center_size[0]
        bottom =  idy * self.center_size[1]
        forward = self.idz * self.center_size[2]
             
        image = np.zeros([self.numx, self.num_input, h, w, d])
        mask = np.zeros([self. numx, h, w, d])
        label = np.zeros([self.numx, h, w, d])
        
        # dimension of label and image B x H x W x D 
        for i in range(self.numx):
            image[i,:] = self.images[:, left[i]: left[i] + h, bottom : bottom + w, forward : forward + d]
            label[i,:] = self.labels[left[i]: left[i] + h, bottom : bottom + w, forward : forward + d]
            mask[i,:] = self.mask[left[i]: left[i] + h, bottom : bottom + w, forward : forward + d]
            
        # images shape: H x W x D 
        # labels shape: H x W x D 
        sample = {'images': image, 'labels':label, 'mask':mask}
        return sample

class ValDataset_full(Dataset):
    def __init__(self, root_dir, args):   
        self.root_dir = root_dir
        self.num_input = args.num_input
        self.length = int(len(self.root_dir)/self.num_input)
        self.root_path = args.root_path
        assert args.mask, "Missing mask as the input"
        assert args.normalization, "You need to do the data normalization before training"
    
    def __len__(self):
        return self.length
   
    def __getitem__(self, idx):
        im = []
        for i in range(self.num_input):
            direct, _ = self.root_dir[self.num_input * idx + i].split("\n")
            name = direct
            if i < self.num_input - 1:
                image = nib.load(self.root_path + direct + '.gz').get_data()
                image = np.expand_dims(image, axis=0)
                im.append(image)
                if i == 0:
                    mask = nib.load(self.root_path + direct + "/mask.nii.gz").get_data()
            else:
                labels = nib.load(self.root_path + direct + '.gz').get_data()
        
        images = np.concatenate(im, axis=0).astype(float)
        
        # images shape: 4 x H x W x D 
        # labels shape: H x W x D 
        sample = {'images': images, 'mask': mask, 'labels':labels, 'name': name}
        return sample

class TestDataset(Dataset):
    def __init__(self, image, mask, num_segments, idz, args):   
        self.images = image
        self.mask = mask
        self.numx = num_segments[0]
        self.numy = num_segments[1]
        self.idz = idz
        self.center_size = args.center_size
        self.crop_size = args.crop_size
        self.num_input = args.num_input 
        
    def __len__(self):
        return self.numy
   
    def __getitem__(self, idy):
        h, w, d = self.crop_size
        left =  np.arange(self.numx) * self.center_size[0]
        bottom =  idy * self.center_size[1]
        forward = self.idz * self.center_size[2]
             
        image = np.zeros([self.numx, self.num_input, h, w, d])
        mask = np.zeros([self. numx, h, w, d])
        
        # dimension of label and image B x H x W x D 
        for i in range(self.numx):
            image[i,:] = self.images[:, left[i]: left[i] + h, bottom : bottom + w, forward : forward + d]
            mask[i,:] = self.mask[left[i]: left[i] + h, bottom : bottom + w, forward : forward + d]
            
        # images shape: H x W x D 
        # labels shape: H x W x D 
        sample = {'images': image, 'mask':mask}
        return sample   

class RandomCrop(object):
    def __init__(self, output_size, random_flip, num_input):
        assert len(output_size) == 3
        self.output_size = output_size        
        self.random_flip = random_flip
        self.num_input = num_input
    
    def __call__(self, sample):
        images, labels, mask = sample['images'], sample['labels'], sample['mask']   
        h, w, d = self.output_size
       
        # generate the training batch with equal probability for the foreground and background
        # within the mask
        labelm = labels + mask
        # choose foreground or background
        fb = np.random.choice(2)
        if fb:
            index = np.argwhere(labelm > 1)
        else:
            index = np.argwhere(labelm == 1)
        # choose the center position of the image segments
        choose = random.sample(range(0, len(index)), 1)
        center = index[choose].astype(int)
        center = center[0]
        
        # check whether the left and right index overflow
        left = []
        for i in range(3):
        	margin_left = int(self.output_size[i]/2)
        	margin_right = self.output_size[i] - margin_left
        	left_index = center[i] - margin_left
        	right_index = center[i] + margin_right
        	if left_index < 0:
        		left_index = 0
        	if right_index > labels.shape[i]:
        		left_index = left_index - (right_index - labels.shape[i])
        	left.append(left_index)
        	
        # crop the image and the label to generate image segments
        image = np.zeros([self.num_input - 1, h, w, d])
        label = np.zeros([h, w, d])
        
        image = images[:, left[0]:left[0] + h, left[1]:left[1] + w, left[2]:left[2] + d]
        label = labels[left[0]:left[0] + h, left[1]:left[1] + w, left[2]:left[2] + d]        
        
        # random flip 
        if self.random_flip:       
        	flip = np.random.choice(2)*2-1
        	image = image[:,::flip,:,:]
        	label = label[::flip,:,:]

        return {'images':image.copy(), 'labels': label.copy()}

