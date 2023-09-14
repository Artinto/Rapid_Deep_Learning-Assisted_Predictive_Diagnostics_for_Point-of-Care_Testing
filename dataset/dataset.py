from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import cv2
import os


class KitDataset(Dataset):
    def __init__(self, root: str, args, is_train=True):
        # use_frame: tuple = (0, 20, 1), zero_th: int = 180000, img_use_hsv: str = 'not_use', use_noise: bool = True,
        self.root = root
        self.img_use_hsv = args.img_use_hsv
        self.is_train = is_train

        self.aug_transform = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., 0.001) if args.use_noise else None
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.use_frame = args.use_frame
        self.label_info = self.load_label_info(args.label_info_path)

        self.all_data = list()

        for i, (path, dir, files) in enumerate(os.walk(root)):
            if i == 0: continue

            if not files:
                label = int(os.path.split(path)[1])
                continue

            all_files = [f for f in sorted(files, key=lambda x:int(x[:-4])) if os.path.splitext(f)[1] in ['.png', '.jpg']]
            files = all_files[args.use_frame[0]:args.use_frame[1]:args.use_frame[2]]
            files = [os.path.join(path, f) for f in files]

            append_obj = {'img_files': files, 'label': label}

            del all_files; del files
            self.all_data.append(append_obj)

    def __getitem__(self, idx):
        i = idx//2 if self.img_use_hsv == 'parallel' and self.is_train else idx

        img_path_list = self.all_data[i]['img_files']
        label = self.all_data[i]['label']
        dst = self.label_info[label]['new']
        cls = self.label_info[label]['active']

        img_tensor_list = []
        for img_path in img_path_list:
            img = cv2.imread(img_path)

            if self.is_train:
                if self.img_use_hsv == 'hsv':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                elif self.img_use_hsv == 'concat':
                    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    img = np.concatenate((img, img_hsv), axis=-1)
                elif self.img_use_hsv == 'parallel':
                    if idx % 2 == 1: img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            elif (not self.is_train) or (self.img_use_hsv == 'not_use'):
                pass

            img = self.aug_transform(img) if self.is_train else self.transform(img)
            img_tensor_list.append(np.array(img))
        img_tensor_list = np.array(img_tensor_list)
        img_tensor_list = torch.from_numpy(img_tensor_list)

        # return_obj = (images, dst, cls)
        return_obj = (
            img_tensor_list,
            torch.tensor(dst, dtype=torch.float32),
            cls
        )
        return return_obj

    def __len__(self):
        if self.is_train and self.img_use_hsv == 'parallel':
            return len(self.all_data)*2
        else:
            return len(self.all_data)

    def load_label_info(self, label_info_path='label_info.csv'):
        with open(label_info_path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
        label_dict = dict()
        header = lines[0].strip().split(',')
        for line in lines[1:]:
            org, new, active = line.strip().split(',')
            label_dict[int(org)] = {header[1]: float(new), header[2]: int(active)}
        return label_dict


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RegDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        self.all_data = list()

        for i, (path, dir, files) in enumerate(os.walk(root)):
            if i == 0: continue
            density_ = int(os.path.split(path)[1])
            density = density_ if density_ != 0 else 300000     # (ex) 1/25000
            for filename in files:
                self.all_data.append((os.path.join(path, filename), density))

    def __getitem__(self, i):
        img_path = self.all_data[i][0]
        img = cv2.imread(img_path)
        density = self.all_data[i][1]

        img = self.transform(img)
        return img, density

    def __len__(self):
        return len(self.all_data)