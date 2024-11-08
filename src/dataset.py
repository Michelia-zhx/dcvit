import os
import random
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import dataset
from torchvision import datasets, transforms

import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from .cifar import CIFAR100_withIdx
from .dataloader import create_train_loader, create_eval_loader
import socket

hostname = socket.gethostname()
if '3090' in hostname:
    cifar10_path = '/opt/Dataset/cifar10'
    cifar100_path = '/opt/Dataset/CIFAR100'
    svhn_path = '/amax/opt/Dataset/SVHN'
    # imagenet_path = '/mnt/ramdisk/ImageNet'
    imagenet_path = '/opt/Dataset/ImageNet'
    CUB_path = '/opt/Dataset/CUB'
    ADI_path = '/opt/Dataset/ADI'
    DI_path = 'gen_metric/final_images/'
    place365_path = '/opt/Dataset/place365'
else:
    cifar10_path = '/opt/Dataset/cifar10'
    cifar100_path = '/opt/Dataset/cifar100'
    svhn_path = '/amax/opt/Dataset/SVHN'
    imagenet_path = '/opt/Dataset/ImageNet'
    # imagenet_path = '~/Dataset/ImageNet'
    CUB_path = '/opt/Dataset/CUB'
    DI_path = '/mnt/data3/zhanghx/divit/final_images/'
    place365_path = '/opt/Dataset/place365'


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_mean_std(model_name):
    model = timm.create_model(model_name, pretrained=True)
    mean = model.default_cfg['mean']
    std = model.default_cfg['std']
    return mean, std

class FewShotImageFolder(torch.utils.data.Dataset):
    # set default seed=None, check the randomness
    def __init__(self, root, transform=None, N=1000, K=-1, few_samples=-1, seed=None):
        super(FewShotImageFolder, self).__init__()
        self.root = os.path.abspath(os.path.expanduser(root))
        self._transform = transform
        # load and parse from a txt file
        self.N = N
        self.K = K
        self.few_samples = few_samples
        self.seed = seed
        self.samples = self._parse_and_sample()
    
    def samples_to_file(self, save_path):
        with open(save_path, "w") as f:
            for (path, label) in self.samples:
                f.writelines("{}, {}\n".format(path.replace(self.root, "."), label))
        print("Writing train samples into {}".format(os.path.abspath(save_path)))

    def __parse(self):
        file_path = os.path.join(self.root, "train.txt")
        full_data = {}
        with open(file_path, "r") as f:
            raw_data = f.readlines()
        for rd in raw_data:
            img_path, target = rd.replace("\n", "").split()
            assert target.isalnum()
            if target not in full_data.keys():
                full_data[target] = []
            full_data[target].append(img_path)
        return full_data
    
    def _parse_and_sample(self):
        N, K, seed = self.N, self.K, self.seed
        assert 1<=N<=1000, r"N with maximum num 1000"
        assert K<=500, r"If you want to use the whole dataset, set K=-1"
        # txt default path: self.root + "/train.txt"
        full_data = self.__parse()
        all = 0
        for v in full_data.values():
            all += len(v)
        print("Full dataset has {} classes and {} images.".format(len(full_data), all))
        print("Using seed={} to sample images.".format(seed))
        sampled_data = []

        np.random.seed(seed)
        # sample classes
        if self.few_samples > 0:
            for i in range(self.few_samples):
                while True:
                    sampled_cls = np.random.choice(list(full_data.keys()), 1, replace=False)
                    cls = sampled_cls[0]
                    sampled_img = np.random.choice(full_data[cls], 1, replace=False)[0]
                    curr_sample = (os.path.join(self.root, "train", sampled_img), cls)
                    if curr_sample not in sampled_data:
                        sampled_data.append(curr_sample)
                        break
            print("Final samples: {}".format(len(sampled_data)))
        else:
            sampled_cls = np.random.choice(list(full_data.keys()), N, replace=False)
            sampled_cls.sort()
            for cls in sampled_cls:
                if K == -1:
                    # use all data
                    sampled_imgs = full_data[cls]
                else:
                    # sample images of every class
                    sampled_imgs = np.random.choice(full_data[cls], K, replace=False)
                sampled_data += [(os.path.join(self.root, "train", i), cls) for i in sorted(sampled_imgs)]
        
        self.idx_to_class = {}
        self.class_to_idx = {}
        for k, v in full_data.items():
            idx = k
            cls = v[0].split("/")[0]
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
        self.classes = list(self.idx_to_class.values())
        self._full_data = full_data
        return sampled_data
        
    def __getitem__(self, index):
        path, label = self.samples[index]
        img = pil_loader(path)
        if self._transform is not None:
            img = self._transform(img)
        return img, int(label)

    def __len__(self):
        return len(self.samples)

    def __repr__(self) -> str:
        return super().__repr__()


def imagenet(args, train, batch_size, sub_idx=None):
    mean, std = get_mean_std(args.model)
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        dataset = datasets.ImageFolder(os.path.join(imagenet_path, 'train'), transform)
        loader = create_train_loader(dataset, batch_size, args)
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        dataset = datasets.ImageFolder(os.path.join(imagenet_path, 'val'), transform)
        loader = create_eval_loader(dataset, batch_size, args)
    return loader


def imagenet_fewshot(args, img_num=1000, batch_size=64, seed=2021, train=True):
    if img_num < 1000:
        few_samples = img_num
        N = 1000
        K = -1
    else:
        few_samples = -1
        N = 1000
        K = img_num // N
    mean, std = get_mean_std(args.model)

    if train:
        # this should always dispatch to transforms_imagenet_train
        if args.data_aug:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        dataset = FewShotImageFolder(
            imagenet_path,
            transform,
            N=N, K=K, few_samples=few_samples, seed=seed)
        loader = create_train_loader(dataset, batch_size=batch_size, args=args)
    else:
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        dataset = FewShotImageFolder(
            imagenet_path,
            transform,
            N=N, K=K, few_samples=few_samples, seed=seed)
        loader = create_eval_loader(dataset, batch_size=batch_size, args=args)
    return loader


def ADI_fewshot(args, num_sample=1000, batch_size=64, seed=2021, train=False):
    if num_sample < 1000:
        few_samples = num_sample
        N = 1000
        K = -1
    else:
        few_samples = -1
        N = 1000
        K = num_sample // N

    if train:
        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        shuffle = True
    else:
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        shuffle = False

    dataset = FewShotImageFolder(
        ADI_path,
        transform,
        N=N, K=K, few_samples=few_samples, seed=seed)

    drop_last=False
    if train and len(dataset) >= batch_size:
        drop_last = True

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=4, pin_memory=False, drop_last=drop_last
    )
    return loader


def ADI_val(args, num_sample=200, batch_size=64, seed=2021):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = datasets.ImageFolder(os.path.join(ADI_path, 'train'), transform)
    np.random.seed(seed)
    if num_sample > 0 and num_sample < len(dataset):
        sub_idx = np.random.choice(list(range(len(dataset))), num_sample, replace=False)
        dataset = torch.utils.data.Subset(dataset, sub_idx)
    drop_last = len(dataset) >= batch_size
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=False, drop_last=drop_last
    )
    return train_loader

def DI_gen(args, num_sample=200, batch_size=64, seed=2021):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if 'vit_base' in args.model:
        dir_name = 'vit_base_inversion'
    elif 'vit_tiny' in args.model:
        dir_name = 'vit_tiny_inversion'
    elif 'vit_small' in args.model:
        dir_name = 'vit_small_inversion'
    elif 'vit_large' in args.model:
        dir_name = 'vit_large_inversion'
    elif 'swin' in args.model:
        dir_name = 'swin_base_inversion'
    elif 'deit' in args.model:
        dir_name = 'deit_base_inversion'
    else:
        assert(0)
    dataset = datasets.ImageFolder(os.path.join(DI_path + dir_name, str(seed)), transform)
    np.random.seed(seed)
    if num_sample > 0 and num_sample < len(dataset):
        sub_idx = np.random.choice(list(range(len(dataset))), num_sample, replace=False)
        dataset = torch.utils.data.Subset(dataset, sub_idx)
    drop_last = len(dataset) >= batch_size
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=False, drop_last=drop_last
    )
    return train_loader


def CUB_val(num_sample=200, batch_size=64, seed=2021):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = datasets.ImageFolder(os.path.join(CUB_path, 'train'), transform)
    np.random.seed(seed)
    if num_sample > 0 and num_sample < len(dataset):
        sub_idx = np.random.choice(list(range(len(dataset))), num_sample, replace=False)
        dataset = torch.utils.data.Subset(dataset, sub_idx)
    drop_last = len(dataset) >= batch_size
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=False, drop_last=drop_last)
    return loader


def place365_sub(args, num_sample, batch_size=64, seed=2021):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = datasets.Places365(place365_path, small=True, transform=transform)
    np.random.seed(seed)
    if num_sample > 0 and num_sample < len(dataset):
        sub_idx = np.random.choice(list(range(len(dataset))), num_sample, replace=False)
        dataset = torch.utils.data.Subset(dataset, sub_idx)
    drop_last = len(dataset) >= batch_size
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=False, drop_last=drop_last)
    return train_loader


def cifar100(train, batch_size=64, args=None):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
    if train:
        transformation = transforms.Compose([
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = CIFAR100_withIdx(root=cifar100_path,
                                        download=True,
                                        train=True,
                                        transform=transformation)
        loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=False)
    else:
        transformation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = CIFAR100_withIdx(root=cifar100_path,
                                    download=True,
                                    train=False,
                                    transform=transformation)
        loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=False)
        
    return loader


def fakedata(num_sample=400, batch_size=64):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = datasets.FakeData(num_sample, num_classes=100, transform=transform)
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=False)
    return train_loader

if __name__ == '__main__':
    import IPython
    IPython.embed()