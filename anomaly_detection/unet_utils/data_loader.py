import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image
import glob

class MVTecDRAEMTestDataset_partial(Dataset):
    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.images=[]
        self.anomaly_names = os.listdir(self.root_dir)
        for idx, anomaly_name in enumerate(self.anomaly_names):          
            img_path=os.path.join(root_dir,anomaly_name)
            img_files = os.listdir(img_path)
            img_files.sort(key=lambda x: int(x[:3]))
            l = len(img_files) // 3
            if anomaly_name=='good':
                l = 0
            self.images += [os.path.join(img_path, file_name) for file_name in img_files[l:]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = dir_path.replace("/test/", "/ground_truth/")
            mask_file_name = file_name.split(".")[0]+"_mask.png" #MVTec # asus
            #mask_file_name = file_name.split(".")[0]+".png" # visA realiad
        
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample


class MVTec_Anomaly_Detection(Dataset):
    def __init__(self, args,sample_name,length=5000,anomaly_id=None,recon=False):
        self.recon=recon
        self.good_path='%s/%s/train/good'%(args.mvtec_path,sample_name)
        self.good_files=[os.path.join(self.good_path,i) for i in os.listdir(self.good_path)]
        self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
        self.anomaly_names = [name for name in os.listdir(f"{self.root_dir}/test") if name != 'good']
        if anomaly_id!=None:
            self.anomaly_names=self.anomaly_names[anomaly_id:anomaly_id+1]
            print('training subsets',self.anomaly_names)
        l=len(self.anomaly_names)
        self.anomaly_num = l
        self.img_paths=[]
        self.mask_paths=[]
        for idx,anomaly in enumerate(self.anomaly_names):
            img_path=[]
            mask_path=[]
            image_dir = os.path.join(self.root_dir, "test", anomaly)
            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

            # for i in range(min(len(os.listdir(os.path.join(self.root_dir,"ground_truth",anomaly))),length)):
            #     img_path.append(os.path.join(self.root_dir,"test", anomaly,'%04d.png'%(i+args.id)))
            #     mask_path.append(os.path.join(self.root_dir,"ground_truth", anomaly,'%04d_mask.png'%(i+args.id)))

            for img_file in image_files[:length]:
                #print(f"img_file:{img_file}")
                img_file_path = os.path.join(image_dir, img_file)
                img_path.append(img_file_path)
                mask_path.append(img_file_path.replace("/test/", "/ground_truth/").replace(".png", "_mask.png"))
                #mask_path.append(img_file_path.replace("/test/", "/pixel_mask/").replace(".png", "_mask.png"))

            self.img_paths.append(img_path.copy())
            self.mask_paths.append(mask_path.copy())
        for i in range(l):
            print(len(self.img_paths[i]),len(self.mask_paths[i]))
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256], antialias=True)
        ])
        self.length=length
        if self.length is None:
            self.length=len(self.good_files)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if random.random()>0.5:
            image=self.loader(Image.open(self.good_files[idx%len(self.good_files)]).convert('RGB'))
            mask=torch.zeros((1,image.size(-2),image.size(-1)))
            has_anomaly = np.array([0], dtype=np.float32)
            sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': -1}
            if self.recon:
                sample['source']=image
        else:
            anomaly_id=random.randint(0,self.anomaly_num-1)
            img_path=self.img_paths[anomaly_id][idx% len(self.mask_paths[anomaly_id])]
            image = self.loader(Image.open(img_path).convert('RGB'))
            mask_path = self.mask_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
            mask = self.loader(Image.open(mask_path).convert('L'))
            mask=(mask>0.5).float()
            if mask.sum()==0:
                has_anomaly = np.array([0], dtype=np.float32)
                anomaly_id=-1
            else:
                has_anomaly = np.array([1], dtype=np.float32)
            sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'anomay_id': anomaly_id}
            if self.recon:
                img_path = self.img_paths[anomaly_id][idx % len(self.mask_paths[anomaly_id])]
                img_path=img_path.replace('image','recon')
                ori_image = self.loader(Image.open(img_path).convert('RGB'))
                sample['source']=ori_image
        return sample

class MVTec_classification_train(Dataset):
    def __init__(self, args,sample_name):
        self.root_dir = '%s/%s'%(args.generated_data_path,sample_name)
        self.anomaly_names=os.listdir(f"{self.root_dir}/test")
        self.img_paths=[]
        self.labels=[]
        for idx,anomaly in enumerate(self.anomaly_names):
            # for i in range(min(len(os.listdir(os.path.join(self.root_dir,anomaly,'mask'))),500)):
            #     self.img_paths.append(os.path.join(self.root_dir,anomaly,'image','%d.jpg'%i))
            #     self.labels.append(idx)
            anomaly_path = os.path.join(self.root_dir,"test",anomaly)
            # for filename in os.listdir(anomaly_path):
            #     if filename.endswith('.png') and '-' in filename:
            #         img_id = filename.split('-')[0] 
            #         self.img_paths.append(os.path.join(anomaly_path, f"{img_id}-{filename.split('-')[1]}"))
            #         self.labels.append(idx)
            #     else:
            #         img_id = filename.split('.')[0] 
            #         self.img_paths.append(os.path.join(anomaly_path,f'{img_id}.png'))
            #         self.labels.append(idx)
            image_files = sorted([f for f in os.listdir(anomaly_path) if f.endswith(".png")])
            for img_file in image_files:
                #print(f"img_file:{img_file}")
                img_file_path = os.path.join(anomaly_path, img_file)
                self.img_paths.append(img_file_path)
                self.labels.append(idx)

            for i in range(min(len(os.listdir(os.path.join(self.root_dir,"ground_truth",anomaly))),args.numbers)):
                self.img_paths.append(os.path.join(anomaly_path,'%04d.png'%(i+args.id)))
                self.labels.append(idx)

        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256], antialias=True)
        ])
        self.length=len(self.img_paths)
    def __len__(self):
        return self.length*5
    def class_num(self):
        return len(self.anomaly_names)
    def return_anomaly_names(self):
        return self.anomaly_names
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image,label

class MVTec_classification_test(Dataset):
    def __init__(self, args,sample_name,anomaly_names):
        root_dir = '%s/%s/test'%(args.mvtec_path,sample_name)
        self.anomaly_names=anomaly_names
        self.img_paths=[]
        self.labels=[]
        for idx, anomaly_name in enumerate(self.anomaly_names):
            img_path=os.path.join(root_dir,anomaly_name)
            img_files = os.listdir(img_path)
            img_files.sort(key=lambda x: int(x[:3]))
            l = len(img_files) // 3 
            self.img_paths += [os.path.join(img_path, file_name) for file_name in img_files[l:]]
            self.labels+=[idx for file_name in img_files[l:]]
        self.loader=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256], antialias=True)
        ])
        self.length=len(self.img_paths)
    def __len__(self):
        return self.length
    def class_num(self):
        return len(self.anomaly_names)
    def __getitem__(self, idx):
        image=self.loader(Image.open(self.img_paths[idx%len(self.img_paths)]).convert('RGB'))
        label=self.labels[idx%len(self.img_paths)]
        return image,label
    
