import cv2
import gradio as gr
import os
import numpy as np
import torch
from torchvision import transforms
from scipy.ndimage import rotate

from sam.efficient_sam.build_efficient_sam import build_efficient_sam_vits

sam = build_efficient_sam_vits()

DESCRIPTION = """
    ## Object Moving & Resizing
    Usage:
    - Upload a target image, and then draw a box to generate the mask corresponding to the editing object.
    - Editing the object's mask."""



def get_point_move(mask, sel_pix, evt: gr.SelectData):
    if mask is None:
        return mask

    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = (mask > 127).astype(np.uint8) * 255

    target_point = [evt.index[0], evt.index[1]]

    h, w = mask.shape

    coords = cv2.findNonZero(mask)
    if coords is None:
        return mask  
    x, y, w_box, h_box = cv2.boundingRect(coords)
    mask_cx = x + w_box // 2
    mask_cy = y + h_box // 2


    dx = target_point[0] - mask_cx
    dy = target_point[1] - mask_cy

    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    moved_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    return moved_mask

def Rotate_mask(mask, d_r, padding_ratio=0.5):
    if mask is None:
        return None

    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    mask = (mask > 127).astype(np.uint8)

    white_pixels = np.where(mask == 1)
    if len(white_pixels[0]) == 0:
        return mask * 255 

    min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
    mid_y = (min_y + max_y) // 2
    mid_x = (min_x + max_x) // 2

    h = max_y - min_y + 1
    w = max_x - min_x + 1
    size = int(max(h, w) * (1 + padding_ratio))

    # padding
    square_mask = np.zeros((size, size), dtype=np.uint8)
    y_start = size // 2 - h // 2
    x_start = size // 2 - w // 2

    square_mask[y_start:y_start+h, x_start:x_start+w] = mask[min_y:max_y+1, min_x:max_x+1]
    rotated = rotate(square_mask, angle=d_r, reshape=True, order=0, mode='constant', cval=0)

    new_h, new_w = rotated.shape
    full_h, full_w = mask.shape

    top = mid_y - new_h // 2
    left = mid_x - new_w // 2

    rotated_mask = np.zeros((full_h, full_w), dtype=np.uint8)

    y1 = max(0, top)
    x1 = max(0, left)
    y2 = min(full_h, top + new_h)
    x2 = min(full_w, left + new_w)

    rot_y1 = max(0, -top)
    rot_x1 = max(0, -left)
    rot_y2 = rot_y1 + (y2 - y1)
    rot_x2 = rot_x1 + (x2 - x1)

    rotated_mask[y1:y2, x1:x2] = rotated[rot_y1:rot_y2, rot_x1:rot_x2]

    return rotated_mask * 255



def segment_with_points(
    image,
    original_image,
    global_points,
    global_point_label,
    evt: gr.SelectData
):
    if original_image is None:
        original_image = image
    else:
        image = original_image
    x, y = evt.index[0], evt.index[1]
    if len(global_points) == 0:
        global_points.append([x, y])
        global_point_label.append(2)
        image_with_point= show_point_or_box(image.copy(), global_points)
        return image_with_point, original_image, None, global_points, global_point_label, original_image
    elif len(global_points) == 1:
        global_points.append([x, y])
        global_point_label.append(3)
        x1, y1 = global_points[0]
        x2, y2 = global_points[1]
        if x1 < x2 and y1 >= y2:
            global_points[0][0] = x1
            global_points[0][1] = y2
            global_points[1][0] = x2
            global_points[1][1] = y1
        elif x1 >= x2 and y1 < y2:
            global_points[0][0] = x2
            global_points[0][1] = y1
            global_points[1][0] = x1
            global_points[1][1] = y2
        elif x1 >= x2 and y1 >= y2:
            global_points[0][0] = x2
            global_points[0][1] = y2
            global_points[1][0] = x1
            global_points[1][1] = y1
        image_with_point = show_point_or_box(image.copy(), global_points)
        # data process
        input_point = np.array(global_points)
        input_label = np.array(global_point_label)
        pts_sampled = torch.reshape(torch.tensor(input_point), [1, 1, -1, 2])
        pts_labels = torch.reshape(torch.tensor(input_label), [1, 1, -1])
        img_tensor = transforms.ToTensor()(image)
        # sam
        predicted_logits, predicted_iou = sam(
            img_tensor[None, ...],
            pts_sampled,
            pts_labels,
        )
        mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).float().cpu().detach().numpy()
        mask_image = (mask*255.).astype(np.uint8)
        return image_with_point, original_image, mask_image, global_points, global_point_label, mask_image
    else:
        global_points=[[x, y]]
        global_point_label=[2]
        image_with_point= show_point_or_box(image.copy(), global_points)
        return image_with_point, original_image, None, global_points, global_point_label, None


def show_point_or_box(image, global_points):
    # for point
    if len(global_points) == 1:
        image = cv2.circle(image, tuple(global_points[0]), 10, (0, 0, 255), -1)
    # for box
    if len(global_points) == 2:
        p1 = global_points[0]
        p2 = global_points[1]
        image = cv2.rectangle(image,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(0,0,255),2)

    return image


def horizontal_movement(mask, dx):
    if mask is None:
        return None 

    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = (mask > 127).astype(np.uint8) * 255

    h, w = mask.shape

    M = np.float32([[1, 0, dx],
                    [0, 1, 0]])
    moved_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    return moved_mask

def vertical_movement(mask, dy):
    if mask is None:
        return None  

    if len(mask.shape) == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = (mask > 127).astype(np.uint8) * 255

    h, w = mask.shape

    M = np.float32([[1, 0, 0],
                    [0, 1, dy]])
    moved_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    return moved_mask

def fun_clear(*args):
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.append([])
        else:
            result.append(None)
    return tuple(result)

def run_demo(original_image, mask):
    return mask

def mask_process_ui():
    with gr.Box(): 
        original_image = gr.State(value=None) # store original image
        mask = gr.State(value=None)
        selected_points = gr.State([])
        global_points = gr.State([])
        global_point_label = gr.State([])
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("### INPUT")
                    gr.Markdown("#### 1. Draw box to mask target object")
                    img_draw_box = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("#### 2.Rotate and Shift")
                    d_r = gr.Slider(
                            label="Rotational movement",
                            minimum=0,
                            maximum=360,
                            step=1,
                            value=0,
                            interactive=True
                        )
                    
                    gr.Markdown("#### 3. Draw point to move mask")
                    img = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")

                    with gr.Row():
                        clear_button = gr.Button("Clear")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("### OUTPUT")
                    img_mask_original = gr.Image(label="Mask of object", interactive=False, type="numpy") 
                    mask = gr.Image(source='upload', label="Target Mask", interactive=True, type="numpy")  

            img_draw_box.select(
                segment_with_points, 
                inputs=[img_draw_box, original_image, global_points, global_point_label], 
                outputs=[img_draw_box, original_image, mask, global_points, global_point_label, img_mask_original]
            )

            d_r.change(
                Rotate_mask,
                inputs = [mask, d_r],
                outputs = [mask]
            )

            img.select(
                    get_point_move,
                    inputs = [mask, selected_points],
                    outputs = [mask]
            )
        clear_button.click(fn=fun_clear, inputs=[original_image, global_points, global_point_label, selected_points, img_mask_original, mask, img_draw_box, img], outputs=[original_image, global_points, global_point_label, selected_points, mask, img_draw_box, img])
    
    print(f"img_mask_original: {type(img_mask_original)},mask:{type(mask)}")
    return img_mask_original, mask