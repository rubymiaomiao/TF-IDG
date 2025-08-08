import numpy as np
import torch 
import cv2

def generate_mask_image_collage(masked_ref_image_compose, ref_mask_compose, cropped_target_image, tar_box_yyxx):
    if len(ref_mask_compose.shape) == 3:
        ref_mask_compose = ref_mask_compose[:,:,0]
    ref_mask_bool = ref_mask_compose > 128
    ref_image_collage = cropped_target_image.copy()
    y1, y2, x1, x2 = tar_box_yyxx
    resized_ref_image = cv2.resize(masked_ref_image_compose, (x2-x1, y2-y1))
    resized_ref_mask = cv2.resize(ref_mask_bool.astype(np.uint8), (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
    resized_ref_image = np.zeros((resized_ref_image.shape[0], resized_ref_image.shape[1], 3), dtype=np.uint8)
    ref_image_collage = np.where(np.expand_dims(resized_ref_mask, axis=2), resized_ref_image, cropped_target_image[y1:y2, x1:x2])

    return ref_image_collage

def generate_ref_image_collage(masked_ref_image_compose, ref_mask_compose, cropped_target_image, tar_box_yyxx):
 
    if len(ref_mask_compose.shape) == 3:
        ref_mask_compose = ref_mask_compose[:,:,0]
    
    ref_mask_bool = ref_mask_compose > 128

    ref_image_collage = cropped_target_image.copy()

    y1, y2, x1, x2 = tar_box_yyxx
    
    resized_ref_image = cv2.resize(masked_ref_image_compose, (x2-x1, y2-y1))
    resized_ref_mask = cv2.resize(ref_mask_bool.astype(np.uint8), (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

    ref_image_collage = np.where(np.expand_dims(resized_ref_mask, axis=2), resized_ref_image, cropped_target_image[y1:y2, x1:x2])

    return ref_image_collage

def mask_score(mask):
    '''Scoring the mask according to connectivity.'''
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    conc_score = np.max(cnt_area) / sum(cnt_area)
    return conc_score


def sobel(img, mask, thresh = 1):
    '''Calculating the high-frequency map.'''
    H,W = img.shape[0], img.shape[1]
    img = cv2.resize(img,(256,256))
    mask = (cv2.resize(mask,(256,256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)
    
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)
    sobel_X = cv2.convertScaleAbs(sobelx)
    sobel_Y = cv2.convertScaleAbs(sobely)
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1) * mask    
    
    scharr[scharr < thresh] = 0.0
    scharr = np.stack([scharr,scharr,scharr],-1)
    scharr = (scharr.astype(np.float32)/255 * img.astype(np.float32) ).astype(np.uint8)
    scharr = cv2.resize(scharr,(W,H))
    return scharr


def resize_and_pad(image, box):
    '''Fitting an image to the box region while keeping the aspect ratio.'''
    y1,y2,x1,x2 = box
    H,W = y2-y1, x2-x1
    h,w =  image.shape[0], image.shape[1]
    r_box = W / H 
    r_image = w / h
    if r_box >= r_image:
        h_target = H
        w_target = int(w * H / h) 
        image = cv2.resize(image, (w_target, h_target))

        w1 = (W - w_target) // 2
        w2 = W - w_target - w1
        pad_param = ((0,0),(w1,w2),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    else:
        w_target = W 
        h_target = int(h * W / w)
        image = cv2.resize(image, (w_target, h_target))

        h1 = (H-h_target) // 2 
        h2 = H - h_target - h1
        pad_param =((h1,h2),(0,0),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    return image



def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask


def resize_box(yyxx, H,W,h,w):
    y1,y2,x1,x2 = yyxx
    y1,y2 = int(y1/H * h), int(y2/H * h)
    x1,x2 = int(x1/W * w), int(x2/W * w)
    y1,y2 = min(y1,h), min(y2,h)
    x1,x2 = min(x1,w), min(x2,w)
    return (y1,y2,x1,x2)


def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]
    if mask.sum() < 10:
        print(f"mask is small,h:{h},w:{w}")
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)


def expand_bbox(mask,yyxx,ratio=[1.2,2.0], min_crop=0):
    y1,y2,x1,x2 = yyxx
    ratio = np.random.randint( ratio[0] * 10,  ratio[1] * 10 ) / 10
    H,W = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if H > W:
        pad_param = ((0,0),(padd_1,padd_2),(0,0))
    else:
        pad_param = ((padd_1,padd_2),(0,0),(0,0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image



def box_in_box(small_box, big_box):
    y1,y2,x1,x2 = small_box
    y1_b, _, x1_b, _ = big_box
    y1,y2,x1,x2 = y1 - y1_b ,y2 - y1_b, x1 - x1_b ,x2 - x1_b
    return (y1,y2,x1,x2 )



def shuffle_image(image, N):
    height, width = image.shape[:2]
    
    block_height = height // N
    block_width = width // N
    blocks = []
    
    for i in range(N):
        for j in range(N):
            block = image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            blocks.append(block)
    
    np.random.shuffle(blocks)
    shuffled_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(N):
        for j in range(N):
            shuffled_image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = blocks[i*N+j]
    return shuffled_image


def get_mosaic_mask(image, fg_mask, N=16, ratio = 0.5):
    ids = [i for i in range(N * N)]
    masked_number = int(N * N * ratio)
    masked_id = np.random.choice(ids, masked_number, replace=False)
    

    
    height, width = image.shape[:2]
    mask = np.ones((height, width))
    
    block_height = height // N
    block_width = width // N
    
    b_id = 0
    for i in range(N):
        for j in range(N):
            if b_id in masked_id:
                mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] * 0
            b_id += 1
    mask = mask * fg_mask
    mask3 = np.stack([mask,mask,mask],-1).copy().astype(np.uint8)
    noise = q_x(image)
    noise_mask = image * mask3 + noise * (1-mask3)
    return noise_mask

def extract_canney_noise(image, mask, dilate=True):
    h,w = image.shape[0],image.shape[1]
    mask = cv2.resize(mask.astype(np.uint8),(w,h)) > 0.5
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask =  cv2.erode(mask.astype(np.uint8), kernel, 10)

    canny = cv2.Canny(image, 50,100) * mask
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask = (cv2.dilate(canny, kernel, 5) > 128).astype(np.uint8)
    mask = np.stack([mask,mask,mask],-1)

    pure_noise = q_x(image, t=1) * 0 + 255
    canny_noise = mask * image + (1-mask) * pure_noise
    return canny_noise


def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)


def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region.astype(np.uint8)



def perturb_mask(gt, min_iou = 0.3,  max_iou = 0.99):
    iou_target = np.random.uniform(min_iou, max_iou)
    h, w = gt.shape
    gt = gt.astype(np.uint8)
    seg = gt.copy()
    
    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.1:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            # Dilate/erode
            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])
            
            seg = np.logical_or(seg, gt).astype(np.uint8)
            #seg = select_max_region(seg) 

        if compute_iou(seg, gt) < iou_target:
            break
    seg = select_max_region(seg.astype(np.uint8)) 
    return seg.astype(np.uint8)


def q_x(x_0,t=65):
    '''Adding noise for and given image.'''
    x_0 = torch.from_numpy(x_0).float() / 127.5 - 1
    num_steps = 100
    
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas,0)
    
    alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise).numpy()  * 127.5 + 127.5 


def extract_target_boundary(img, target_mask):
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)

    # sobel-x
    sobel_X = cv2.convertScaleAbs(sobelx)
    # sobel-y
    sobel_Y = cv2.convertScaleAbs(sobely)
    # sobel-xy
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1).astype(np.float32)/255
    scharr = scharr *  target_mask.astype(np.float32)
    return scharr


#===============================================================================

def preprocess_image(img, target_shape):
    h, w = target_shape

    # Resize to target shape
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))

    # If shape is (h, w) or (h, w, 1), convert to 3 channels
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    # Scale if float (like ref_processed which might be [0, 1])
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    return img
def compose_images(ref_image, ref_mask, gt_image, tar_mask, synthesis, ref_processed):
    h, w = ref_image.shape[:2]
    grid = np.zeros((h*2, w*3, 3), dtype=np.uint8)
    if ref_processed.shape[:2] != (h, w):
        ref_processed = cv2.resize(ref_processed, (w, h))
    
    images = [ref_image, ref_mask[:,:,None]*255, gt_image,
              tar_mask[:,:,None]*255, ref_processed*255, synthesis]
    positions = [(0,0), (0,1), (1,0), (1,1), (0,2), (1,2)]
    
    for img, (i, j) in zip(images, positions):
        grid[i*h:(i+1)*h, j*w:(j+1)*w] = img
    
    return grid


def get_connected_components(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for i in range(1, num_labels):  
        component_mask = (labels == i).astype(np.uint8)
        components.append(component_mask)
    return components


import numpy as np
import random
from scipy.ndimage import rotate, zoom
import re
from scipy.ndimage import rotate, zoom, binary_dilation, binary_erosion
from skimage.morphology import disk
import os
def extract_foreground_mask(gt_image):

    gray = cv2.cvtColor(gt_image, cv2.COLOR_RGB2GRAY)

    _, binary_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_mask.astype(np.uint8)



def get_next_image_index(directory, prefix=""):
    existing_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".png")]
    if not existing_files:
        return 0
    max_index = -1
    for f in existing_files:
        # '0001_sourcepath.png' -> '0001' -> 1
        match = re.match(r"(\d+)_", f)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx
    return max_index + 1

def reference_mask_augmentation(mask):
    mask = (mask > 0).astype(np.uint8)
    kernel_size = np.random.randint(5, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    return dilated_mask

def rotate_image_and_mask(image, mask):
    k = random.randint(0, 3) 
    rotated_image = np.rot90(image, 2)
    rotated_mask = np.rot90(mask, 2)
    return rotated_image, rotated_mask
def rotate_image_and_mask2(image, mask):
    angle = random.uniform(0, 360)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    return rotated_image, rotated_mask

def rotate_image_and_mask_for_transistor(image, ref_mask, tar_mask):
    '''Shift Defect'''
    angle = random.uniform(0, 360)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rotated_ref_mask = cv2.warpAffine(ref_mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    rotated_tar_mask = cv2.warpAffine(tar_mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return rotated_image, rotated_ref_mask, rotated_tar_mask
def rotate_mask(mask):
    """
    Randomly rotate the mask around the center of the image
    """
    height, width = mask.shape
    center = (width // 2, height // 2)
    angle = random.uniform(0, 360)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (width, height), 
                                    flags=cv2.INTER_NEAREST,  
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=0)  
    return rotated_mask


def seg_mask_augmentation(mask, reference_mask):
    original_size = mask.shape
    white_pixels = np.where(mask == 1)
    if len(white_pixels[0]) == 0:
        return mask

    min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
    mid_x = (min_x + max_x) // 2
    white_region = mask[min_y:max_y+1, min_x:max_x+1]
    angle = random.uniform(0, 360)
    rotated = rotate(white_region, angle, reshape=True, mode='constant', cval=0)

    scale = random.uniform(0.7, 0.8)
    scaled = zoom(white_region, scale, order=0)
    
    reference_mask = (reference_mask > 0).astype(np.uint8)

    new_mask = np.zeros(original_size, dtype=np.uint8)

    ref_white_pixels = np.column_stack(np.where(reference_mask == 1))
    if len(ref_white_pixels) == 0:
        return mask  

    center_y, center_x = random.choice(ref_white_pixels)

    start_y = max(center_y - scaled.shape[0] // 2, 0)
    start_x = max(center_x - scaled.shape[1] // 2, 0)

    end_y = min(start_y + scaled.shape[0], new_mask.shape[0])
    end_x = min(start_x + scaled.shape[1], new_mask.shape[1])
    scaled_cropped = scaled[:end_y-start_y, :end_x-start_x]

    new_mask[start_y:end_y, start_x:end_x] = scaled_cropped
    return new_mask

def seg_mask_random_placement_region(mask, object_mask):
    white_pixels = np.where(mask == 1)
    if len(white_pixels[0]) == 0:
        return mask
    
    min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
    white_region = mask[min_y:max_y+1, min_x:max_x+1]

    angle = random.uniform(0, 360)
    rotated = rotate(white_region, angle, reshape=True, mode='constant', cval=0)
    scale = random.uniform(0.9, 1.2)
    scaled = zoom(rotated, scale, order=0)

    object_coords = np.argwhere(object_mask)
    if len(object_coords) == 0:
        return mask
    
    target_point = object_coords[np.random.choice(len(object_coords))]

    new_mask = np.zeros_like(mask)
    h, w = scaled.shape
    top_left_y = target_point[0] - h // 2
    top_left_x = target_point[1] - w // 2

    for i in range(h):
        for j in range(w):
            y, x = top_left_y + i, top_left_x + j
            if (0 <= y < mask.shape[0]) and (0 <= x < mask.shape[1]) and scaled[i, j] > 0:
                if object_mask[y, x] == 1: 
                    new_mask[y, x] = 1
    return new_mask

def mask_augmentation(mask,diversity=True):
    original_size = mask.shape
    white_pixels = np.where(mask == 1)
    if len(white_pixels[0]) == 0:  
        return mask
    
    min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
    
    white_region = mask[min_y:max_y+1, min_x:max_x+1]
    
    if diversity:
        angle = random.uniform(0, 360)
        rotated = rotate(white_region, angle, reshape=True, mode='constant', cval=0)
        
        scale = random.uniform(0.7, 1.1)
        scaled = zoom(rotated, scale, order=0)
    else:
        scaled = white_region

    new_mask = np.zeros(original_size, dtype=np.uint8)
    
    max_y_pos = original_size[0] - scaled.shape[0]
    max_x_pos = original_size[1] - scaled.shape[1]

    if max_y_pos > 0 and max_x_pos > 0:
        y_pos = random.randint(0, max_y_pos)
        x_pos = random.randint(0, max_x_pos)
    else:
        scale_y = original_size[0] / scaled.shape[0]
        scale_x = original_size[1] / scaled.shape[1]
        scale = min(scale_y, scale_x) * 1 
        scaled = zoom(scaled, scale, order=0)
        y_pos = random.randint(0, original_size[0] - scaled.shape[0])
        x_pos = random.randint(0, original_size[1] - scaled.shape[1])

    new_mask[y_pos:y_pos+scaled.shape[0], x_pos:x_pos+scaled.shape[1]] = scaled
    
    return new_mask

def edge_mask_augmentation(mask, object_mask):
    white_pixels = np.where(mask == 1)
    if len(white_pixels[0]) == 0:
        return mask
    
    min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
    white_region = mask[min_y:max_y+1, min_x:max_x+1]

    angle = random.uniform(0, 360)
    rotated = rotate(white_region, angle, reshape=True, mode='constant', cval=0)
    scale = random.uniform(0.6, 0.9)
    scaled = zoom(rotated, scale, order=0)

    kernel = disk(5)
    object_edge = binary_dilation(object_mask, structure=kernel) ^ binary_erosion(object_mask, structure=kernel)

    edge_coords = np.argwhere(object_edge)
    if len(edge_coords) == 0:
        return mask
    target_point = edge_coords[np.random.choice(len(edge_coords))]

    new_mask = np.zeros_like(mask)
    h, w = scaled.shape
    top_left_y = target_point[0] - h // 2
    top_left_x = target_point[1] - w // 2

    for i in range(h):
        for j in range(w):
            y, x = top_left_y + i, top_left_x + j
            if (0 <= y < mask.shape[0]) and (0 <= x < mask.shape[1]) and scaled[i, j] > 0:
                if object_mask[y, x] == 1: 
                    new_mask[y, x] = 1

    return new_mask