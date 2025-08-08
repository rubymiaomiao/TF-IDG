import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torchvision.transforms import transforms
import math 
#from feature_extractor import *
import numpy as np


def heat_map(output, target, FE):
    '''
    Compute the anomaly map
    :param output: the output of the reconstruction
    :param target: the target image
    :param FE: the feature extractor
    :param sigma: the sigma of the gaussian kernel
    :param i_d: the pixel distance
    :param f_d: the feature distance
    '''
    sigma = 3
    kernel_size = 2 * int(4 * sigma + 0.5) +1
    anomaly_map = 0

    output = torch.tensor(output).to('cuda').permute(2, 0, 1)
    target = torch.tensor(target).to('cuda').permute(2, 0, 1)
    output = output.unsqueeze(0)  # 添加 batch_size 維度，變成 [1, 3, 1024, 1024]
    target = target.unsqueeze(0)  # 同上

    i_d = pixel_distance(output, target)
    f_d = feature_distance((output),  (target), FE)
    # f_d = torch.Tensor(f_d).to(config.model.device)
    f_d = f_d.clone().detach() # modify
    
    # visualalize_distance(output, target, i_d, f_d)
    i_d = F.interpolate(i_d, size=(256, 256), mode='bilinear', align_corners=False)
    print('image_distance shape : ',i_d.shape)
    print('feature_distance shape : ',f_d.shape)
    anomaly_map += f_d + 4 *  torch.max(f_d)/ torch.max(i_d) * i_d  
    #anomaly_map += f_d + 0 * torch.max(f_d)
    anomaly_map = gaussian_blur2d(
        anomaly_map , kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma)
        )
    anomaly_map = torch.sum(anomaly_map, dim=1).unsqueeze(1)
    return anomaly_map



def pixel_distance(output, target):
    '''
    Pixel distance between image1 and image2
    '''
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    print(output.shape) 
    print(target.shape) 
    output = transform(output)
    target = transform(target)
    distance_map = torch.mean(torch.abs(output - target), dim=1).unsqueeze(1)
    return distance_map




def feature_distance(output, target, FE):
    '''
    Feature distance between output and target
    '''
    FE.to('cuda')
    target = target.to('cuda')
    output = output.to('cuda')
    FE.eval()
    transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    

    print("output.shape",output.shape) 
    print("target.shape",target.shape) 
    target = transform(target)
    output = transform(output)
    inputs_features = FE(target)
    output_features = FE(output)
    out_size = 256
    anomaly_map = torch.zeros([inputs_features[0].shape[0] ,1 ,out_size, out_size]).to('cuda')
    for i in range(len(inputs_features)): # len(inputs_features) = 3
        if i == 0:
            continue
        a_map = 1 - F.cosine_similarity(patchify(inputs_features[i]), patchify(output_features[i]))
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        anomaly_map += a_map
    return anomaly_map 


#https://github.com/amazon-science/patchcore-inspection
def patchify(features, return_spatial_info=False):
    """Convert a tensor into a tensor of respective patches.
    Args:
        x: [torch.Tensor, bs x c x w x h]
    Returns:
        x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
        patchsize]
    """
    patchsize = 3
    stride = 1
    padding = int((patchsize - 1) / 2)
    unfolder = torch.nn.Unfold(
        kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    )
    unfolded_features = unfolder(features)
    number_of_total_patches = []
    for s in features.shape[-2:]:
        n_patches = (
            s + 2 * padding - 1 * (patchsize - 1) - 1
        ) / stride + 1
        number_of_total_patches.append(int(n_patches))
    unfolded_features = unfolded_features.reshape(
        *features.shape[:2], patchsize, patchsize, -1
    )
    unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
    max_features = torch.mean(unfolded_features, dim=(3,4))
    features = max_features.reshape(features.shape[0], int(math.sqrt(max_features.shape[1])) , int(math.sqrt(max_features.shape[1])), max_features.shape[-1]).permute(0,3,1,2)
    if return_spatial_info:
        return unfolded_features, number_of_total_patches
    return features



def save_anomaly_maps(anomaly_maps, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    anomaly_maps = anomaly_maps.cpu().numpy()

    for i, anomaly_map in enumerate(anomaly_maps):

        anomaly_map = anomaly_map.squeeze()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(anomaly_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Anomaly Map , Sample {i}')
        plt.axis('off')
        
        save_path = os.path.join(save_dir, f'anomaly_map_batch_sample{i}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
