import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import os
# Writer will output to ./runs/ directory by default
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class TensorboardVisualizer():

    def __init__(self,log_dir='./logs/'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        #self.writer = SummaryWriter(log_dir=log_dir)
        self.save_dir = log_dir + "/visualize"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        #print("log_dir:",self.save_dir)

    def visualize_image_batch(self,image_batch,n_iter,image_name='Image_batch'):
        grid = torchvision.utils.make_grid(image_batch)
        self.writer.add_image(image_name,grid,n_iter)

    def plot_loss(self, loss_val, n_iter, loss_name='loss'):
        self.writer.add_scalar(loss_name, loss_val, n_iter)

    def visualize_batch_anomaly_heatmaps(self, batch_images, anomaly_heatmaps, count, dir, true_masks=None, alpha=0.5):
        """
        Visualizes anomaly heatmaps overlaid on original images for a batch.

        Parameters:
            batch_images (torch.Tensor or np.ndarray): Batch of images, shape (B, C, H, W).
            anomaly_heatmaps (np.ndarray): Corresponding anomaly heatmaps, shape (B, H, W).
            true_masks (np.ndarray, optional): Ground truth masks for comparison, shape (B, H, W).
            alpha (float): Weight for blending heatmap and original image.
        """
        batch_size = batch_images.shape[0]
        
        # Convert images to numpy and format correctly
        if isinstance(batch_images, torch.Tensor):
            batch_images = batch_images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (B, H, W, C)
        
        fig, axes = plt.subplots(batch_size, 3 if true_masks is not None else 2, figsize=(10, 5 * batch_size))
        axes = np.atleast_2d(axes)

        for i in range(batch_size):
            img = batch_images[i]
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            img = (img * 255).astype(np.uint8)  # Convert to uint8

            # Convert heatmap to color
            heatmap = (anomaly_heatmaps[i] * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Overlay heatmap on original image
            overlaid_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

            if true_masks is not None:
                gt_mask = (true_masks[i] * 255).astype(np.uint8)  # 轉爲灰度
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2RGB)  # 轉換為 3 通道
                row_images = [img, heatmap, overlaid_img, gt_mask]
            else:
                row_images = [img, heatmap, overlaid_img]

            row_combined = np.hstack(row_images)
            save_path = os.path.join(self.save_dir, dir)
            if not os.path.exists(save_path):
                os.makedirs(save_path)          
            plt.imsave(os.path.join(save_path,f"{count}_heatmap.png"), row_combined)