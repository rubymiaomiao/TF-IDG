import cv2
import einops
import numpy as np
import torch
import random
import gradio as gr
import os
from PIL import Image
from datasets.data_utils import * 
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler 
from omegaconf import OmegaConf
from cldm.hack import disable_verbosity, enable_sliced_attention
from atten.shareatten import *
from pytorch_lightning import seed_everything
from atten.resnet import *
from atten.anomaly_map import *
from scipy.ndimage import rotate, zoom
from mask_process_demo import *



cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

save_memory = False  #save memory

config = OmegaConf.load('./configs/demo.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file
use_interactive_seg = config.config_file

model = create_model(model_config).cpu()
if save_memory:
    state_dict = load_state_dict(model_ckpt, location='cpu')
    state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
else:
    model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))

model = model.cuda()
ddim_sampler = DDIMSampler(model)

if use_interactive_seg:
    from iseg.coarse_mask_refine_util import BaselineModel
    model_path = './iseg/coarse_mask_refine.pth'
    iseg_model = BaselineModel().eval()
    weights = torch.load(model_path , map_location='cpu')['state_dict']
    iseg_model.load_state_dict(weights, strict= True)


disable_verbosity()
if save_memory:
    enable_sliced_attention()
    torch.cuda.empty_cache()

def process_image(image,num_samples):
    image = torch.from_numpy(image).float().cuda()
    image = torch.stack([image for _ in range(num_samples)], dim=0)
    image = einops.rearrange(image, 'b h w c -> b c h w').contiguous()
    return image

def get_next_image_index(directory, prefix=""):
    existing_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".png")]
    if not existing_files:
        return 0
    max_index = max(int(f.split(".")[0]) for f in existing_files if f.split(".")[0].isdigit())
    return max_index + 1

def mask_augmentation(image, ref_mask, mask, shift):
    original_size = mask.shape
    white_pixels = np.where(mask == 1)
    if len(white_pixels[0]) == 0:
        return mask
    
    min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
    mid_x = (min_x + max_x) // 2
    mid_y = (min_y + max_y) // 2
    white_region = mask[min_y:max_y+1, min_x:max_x+1]
    if shift == -1:
        angle = random.uniform(-180, 180)
    else:
        angle = shift

    rotated = rotate(white_region, angle, reshape=True, mode='constant', cval=0)
    image = rotate(image, angle, reshape=True, mode='constant', cval=0)
    ref_mask = rotate(ref_mask, angle, reshape=True, mode='constant', cval=0)

    new_mask = np.zeros(original_size, dtype=np.uint8)
    offset_y = random.randint(-15, 15)  
    offset_x = random.randint(-15, 15)  

    start_y = max(mid_y - rotated.shape[0] // 2  + offset_y , 0)
    start_x = max(mid_x - rotated.shape[1] // 2  + offset_x , 0)

    end_y = min(start_y + rotated.shape[0], new_mask.shape[0])
    end_x = min(start_x + rotated.shape[1], new_mask.shape[1])
    scaled_cropped = rotated[:end_y-start_y, :end_x-start_x]
    new_mask[start_y:end_y, start_x:end_x] = scaled_cropped

    return image,ref_mask,new_mask

def compose_images(ref_image, ref_mask, gt_image, tar_mask, synthesis, ref_processed):
    h, w = ref_image.shape[:2]
    
    if ref_processed.shape[:2] != (h, w):
        ref_processed = cv2.resize(ref_processed, (w, h))

    images = [
        ref_image,
        np.stack([ref_mask * 255] * 3, axis=-1),  
        gt_image,
        np.stack([tar_mask * 255] * 3, axis=-1),  
        ref_processed * 255,
        synthesis
    ]

    row_image = np.hstack(images)

    return row_image

def process_image_mask(image_np, mask_np):
    img = torch.from_numpy(image_np.transpose((2, 0, 1)))
    img = img.float().div(255).unsqueeze(0)
    mask = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    pred = iseg_model(img, mask)['instances'][0,0].detach().numpy() > 0.5 
    return pred.astype(np.uint8)

def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 3 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]
    tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return tar_image

def inference_single_image(ref_image, 
                           ref_mask, 
                           tar_image, 
                           tar_mask, 
                           strength, 
                           ddim_steps, 
                           scale, 
                           seed,
                           enable_shape_control,
                           use_inpainting,
                           ddim_inversion,
                           share_attention,
                           start_step,
                           energy_score,
                           adain_weight,
                           adaptive_mask
                           ):
    
    # ---------------------
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)
    
    raw_background = tar_image.copy()

    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, enable_shape_control)

    ref = item['ref']
    hint = item['hint']
    target = item['jpg']
    num_samples = 1

    if save_memory:
        model.low_vram_shift(is_diffusing=False)
        torch.cuda.empty_cache()

    #controlnet
    control = process_image(hint.copy(),num_samples)

    #ID Extractor--> dinov2
    clip_input = process_image(ref.copy(),num_samples)

    H,W = 512,512
    # model cldm.cldm.ControlLDM
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": [control], 
               "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    ddim_cond = {"c_concat": None, 
               "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    model.control_scales = ([strength] * 13)
    
    target_img = process_image(item['jpg'],num_samples)

    target_mask = item['mask']  
    if target_mask is None or not isinstance(target_mask, np.ndarray):
        raise ValueError("target_mask is not a valid numpy array")
    target_mask = cv2.resize(target_mask.astype(np.uint8), (64,64)).astype(np.float32)
    target_mask = torch.from_numpy(target_mask).float().cuda()
    target_mask = torch.stack([target_mask for _ in range(num_samples)], dim=0)
    target_mask = einops.rearrange(target_mask, 'b h w -> b 1 h w').clone()

    #if DDIM inversion 
    ref_background = process_image(item['ref_background'],num_samples) 
    ref_ddim_mask = item['ref_mask'] 
    if ref_ddim_mask is None or not isinstance(ref_ddim_mask, np.ndarray):
        raise ValueError("target_mask is not a valid numpy array")
    ref_ddim_mask = cv2.resize(ref_ddim_mask.astype(np.uint8), (64,64)).astype(np.float32)
    ref_ddim_mask = torch.from_numpy(ref_ddim_mask).float().cuda()
    ref_ddim_mask = torch.stack([ref_ddim_mask for _ in range(num_samples)], dim=0)
    ref_ddim_mask = einops.rearrange(ref_ddim_mask, 'b h w -> b 1 h w').clone()

    x0 = model.get_first_stage_encoding(model.encode_first_stage(target_img))
    x0_tar = model.get_first_stage_encoding(model.encode_first_stage(ref_background))

    if save_memory:
        model.low_vram_shift(is_diffusing=True)
    
    if use_inpainting:
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                        shape, cond, verbose=False, eta=0,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=un_cond,
                                        x0=x0,
                                        mask=target_mask) 
    elif ddim_inversion:
        ddim_samples, latents_list = ddim_sampler.ddim_inversion(x0, conditioning=ddim_cond, num_steps=ddim_steps)
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                        shape, verbose=False, eta=0,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=ddim_cond, 
                                        x_T=ddim_samples)
    elif share_attention: 
        editor = ShareSelfAttentionControl(start_step,10)
        regiter_attention_editor_ldm(model, editor)
        ddim_samples, latents_list = ddim_sampler.ddim_inversion(x0, conditioning=ddim_cond, num_steps=ddim_steps)
        editor.switch_to_use_mode()
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                        shape, cond, verbose=False, eta=0,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=un_cond,
                                        x0=x0,
                                        mask=target_mask,
                                        tar_latents=latents_list,
                                        adain_weight=adain_weight)
        editor.reset()
    
    elif energy_score:
        ddim_samples, latents_list = ddim_sampler.ddim_inversion(x0_tar, conditioning=ddim_cond, num_steps=ddim_steps)
        editor = ShareSelfAttentionControl(start_step,10)
        regiter_attention_editor_ldm(model, editor)
        ori_ddim_samples, ori_latents_list = ddim_sampler.ddim_inversion(x0, conditioning=ddim_cond, num_steps=ddim_steps)
        editor.switch_to_use_mode()
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                        shape, cond, verbose=False, eta=0,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=un_cond,
                                        x0=x0,
                                        mask=target_mask,
                                        energy_score=energy_score,
                                        ref_latents=latents_list,
                                        ref_mask=ref_ddim_mask,
                                        tar_latents=ori_latents_list,
                                        adain_weight=adain_weight,
                                        x0_tar=x0_tar,
                                        adaptive_mask=adaptive_mask,
                                        )
        editor.reset()

    else:
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                        shape, cond, verbose=False, eta=0,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=un_cond)

    if save_memory:
        model.low_vram_shift(is_diffusing=False)
        torch.cuda.empty_cache()

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop']

    # keep background unchanged
    tar_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    y1,y2,x1,x2 = item['tar_box_yyxx']
    raw_background[y1:y2, x1:x2, :] = tar_image[y1:y2, x1:x2, :]
    
    return raw_background, item


def process_pairs(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 0.8, enable_shape_control = False):
    # ========= Reference ===========
    #ref_mask = expand_mask(ref_mask)
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask) 
    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    cropped_ref_mask = ref_mask[y1:y2,x1:x2]
    ref_background = ref_image[y1:y2,x1:x2]

    ratio = np.random.randint(11, 15) / 10 #11,13
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize 
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # collage aug 
    masked_ref_image_compose, ref_mask_compose =  masked_ref_image, ref_mask
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)  

    # ========= Target ===========
    tar_mask_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_mask_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3
    tar_box_yyxx_full = tar_box_yyxx

    #----tar background------
    tar_box_yyxx_forground = box2squre(tar_image, tar_mask_box_yyxx)
    y1,y2,x1,x2 = tar_box_yyxx_forground
    tar_background = tar_image[y1:y2,x1:x2,:]
    #------------------------
    
    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])   
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop
    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    cropped_tar_mask = tar_mask[y1:y2,x1:x2]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx
    # collage
    ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0
    if enable_shape_control:
        print("Enable shape control")
        collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]

    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    cropped_tar_mask = np.expand_dims(cropped_tar_mask, axis=-1)
    cropped_tar_mask = pad_to_square(cropped_tar_mask, pad_value = 0, random = False).astype(np.uint8)
    cropped_ref_mask = np.expand_dims(cropped_ref_mask, axis=-1)
    cropped_ref_mask = pad_to_square(cropped_ref_mask, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = 2, random = False).astype(np.uint8)
    tar_background = pad_to_square(tar_background, pad_value = 0, random = False).astype(np.uint8)
    ref_background = pad_to_square(ref_background, pad_value = 0, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]

    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512,512)).astype(np.float32)
    tar_background = cv2.resize(tar_background.astype(np.uint8), (512,512)).astype(np.float32)
    cropped_tar_mask = cv2.resize(cropped_tar_mask.astype(np.uint8), (512,512)).astype(np.float32)
    ref_background = cv2.resize(ref_background.astype(np.uint8), (512,512)).astype(np.float32)
    cropped_ref_mask = cv2.resize(cropped_ref_mask.astype(np.uint8), (512,512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512,512)).astype(np.float32)
    collage_mask  = cv2.resize(collage_mask.astype(np.uint8), (512,512),  interpolation = cv2.INTER_NEAREST).astype(np.float32)
    collage_mask[collage_mask == 2] = -1

    masked_ref_image = masked_ref_image  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    tar_background = tar_background / 127.5 - 1.0
    ref_background = ref_background / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1] ] , -1) #adds the collage mask as an additional channel to the collage image.
    
    item = dict(ref=masked_ref_image.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                tar_background=tar_background.copy(),
                ref_background=ref_background.copy(),
                mask=cropped_tar_mask,
                ref_mask=cropped_ref_mask,
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ),
                tar_box_yyxx=np.array(tar_box_yyxx_full),
                 ) 
    return item

def mask_image(image, mask):
    blanc = np.ones_like(image) * 255
    mask = np.stack([mask,mask,mask],-1) / 255
    masked_image = mask * ( 0.5 * blanc + 0.5 * image) + (1-mask) * image
    return masked_image.astype(np.uint8)

def run_local(base,
              ref,
              base_mask,
              ref_mask,
              process_based_mask,
              num_samples,
              shift,
              save_dir,
              reference_mask_refine,
              *args):

    save_path = os.path.join("./result/asus", save_dir, "test")
    save_mask_path = os.path.join("./result/asus", save_dir, "ground_truth")
    save_source_path = os.path.join("./result/asus", save_dir, "source")
    image = base["image"].convert("RGB") 
    ref_image = ref["image"].convert("RGB")
    
    if ref_mask != None:
        ref_mask = ref_mask.convert("L")
    else:
        ref_mask = ref["mask"].convert("L")

    #upload > sam mask > painting 
    if base_mask != None:
        mask = np.asarray(base_mask.convert("L"))
    elif process_based_mask is not None:
        if isinstance(process_based_mask, np.ndarray):
            mask = process_based_mask     
            if mask.ndim == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)       
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            elif mask.max() <= 255 and mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
        else:
            mask = np.asarray(process_based_mask.convert("L"))
    else:
        mask = np.asarray(base["mask"].convert("L"))

    image = np.asarray(image)
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)

    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

    
    ref_mask = cv2.resize(ref_mask, (ref_image.shape[1], ref_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if ref_mask.sum() == 0:
        raise gr.Error('No mask for the reference image.')

    if mask.sum() == 0:
        raise gr.Error('No mask for the background image.')

    if reference_mask_refine:
        print("reference_mask_refine:",reference_mask_refine)
        ref_mask = process_image_mask(ref_image, ref_mask)

    for i in range(num_samples):
        if shift:
            ref_image1,ref_mask1,mask1 = mask_augmentation(ref_image,ref_mask,mask,shift)
        else:
            ref_image1,ref_mask1,mask1 = ref_image.copy(), ref_mask.copy(), mask.copy()
        synthesis, item = inference_single_image(ref_image1.copy(), ref_mask1.copy(), image.copy(), mask1.copy(), *args)
        synthesis = torch.from_numpy(synthesis).permute(2, 0, 1)
        synthesis = synthesis.permute(1, 2, 0).numpy()

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_mask_path, exist_ok=True)
        # os.makedirs(save_source_path, exist_ok=True)

        next_index = get_next_image_index(save_path)
        cv2.imwrite(os.path.join(save_path,f"{next_index:03d}.png"), cv2.cvtColor(synthesis.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_mask_path,f"{next_index:03d}.png"), (mask1 * 255).astype(np.uint8))
        # source_image = compose_images(ref_image1, ref_mask1, image, mask1, synthesis, ref_processed)
        # cv2.imwrite(os.path.join(save_source_path,f"{next_index:03d}.png"), cv2.cvtColor((source_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


#==================================================================================== 
    ref_processed = item['ref']
    jpg_processed = item['jpg']
    hint_processed = item['hint']
    #target_background_processed = item['ref_background']
    #mask_processed = item['mask']

    ref_img = Image.fromarray((ref_processed * 255).astype(np.uint8))
    jpg_img = Image.fromarray(((jpg_processed + 1) * 127.5).astype(np.uint8))
    hint_img = Image.fromarray(((hint_processed[:,:,:3] + 1) * 127.5).astype(np.uint8))
    target_background = Image.fromarray((mask1 * 255).astype(np.uint8))
    #target_background = Image.fromarray(((target_background_processed + 1) * 127.5).astype(np.uint8))
    #mask_img = Image.fromarray(np.squeeze(mask_processed).astype(np.uint8))
    #hint_img = Image.fromarray(np.squeeze(ref_mask).astype(np.uint8))
    
    torch.cuda.empty_cache()  # Release unused but cached memory
    torch.cuda.ipc_collect()  # Force PyTorch to release VRM

    return [synthesis],ref_img, jpg_img, hint_img, target_background



with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# Play with TF-IDG to Generate Defect") 
        with gr.Box(): 
            object_mask, process_based_mask = mask_process_ui()
        gr.Markdown("# Upload / Select Images for the Background (left) and Reference Object (right)")
        gr.Markdown("### You could draw coarse masks on the background to indicate the desired location and shape.")
        gr.Markdown("### <u>Do not forget</u> to annotate the target object on the reference image.")
        with gr.Row():
            with gr.Column():
                ref = gr.Image(label="Reference", source="upload", tool="sketch", type="pil", brush_color='#FFFFFF')
                ref_mask = gr.Image(label="Reference Mask", source="upload", type="pil", brush_color='#FFFFFF')

            with gr.Column():
                base = gr.Image(label="Background", source="upload", tool="sketch", type="pil", brush_color='#FFFFFF')
                base_mask = gr.Image(label="Target Mask", source="upload", type="pil", brush_color='#FFFFFF')
        with gr.Row():
            baseline_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=1)
            with gr.Accordion("Generated Option", open=True):
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=30.0, value=4.5, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=999999999, step=1, value=-1)
                reference_mask_refine = gr.Checkbox(label='Reference Mask Refine', value=False, interactive = True)
                enable_shape_control = gr.Checkbox(label='Enable Shape Control', value=False, interactive = True)

                use_inpainting = gr.Checkbox(label='Use_inpainting', value=False, interactive = True)
                ddim_inversion = gr.Checkbox(label='Use_ddim_inversion', value=False, interactive = True)
                share_attention = gr.Checkbox(label='Share_attention', value=False, interactive = True)
                energy_score = gr.Checkbox(label='Energy Function ', value=False, interactive = True)
                adaptive_mask = gr.Checkbox(label='Adaptive Mask ', value=False, interactive = True)
                start_step = gr.Slider(label="Share attention step", minimum=4, maximum=50, value=25, step=1)
                adain_weight = gr.Slider(label="Adain weight", minimum=0, maximum=1, value=0.3, step=0.1)

        with gr.Row():
            with gr.Accordion("Samples Option", open=True):
                num_samples = gr.Slider(label="Num samples", minimum=1, maximum=50, value=1, step=1)
                save_dir = gr.TextArea(label='Samples save', value='', placeholder='asus/object_type')
                shift = gr.Slider(label="Mask rotation", minimum=-1, maximum=360, value=-1, step=1)
                #shift = gr.Checkbox(label='Mask Shift', value=False, interactive = True)

        gr.Markdown("# Visualization of processed images.")
        with gr.Row():
            ref_output = gr.Image(label="Processed Reference", type="pil")
            jpg_output = gr.Image(label="Processed Target", type="pil")
            hint_output = gr.Image(label="Hint Image", type="pil")
            target_background_output = gr.Image(label="Mask Image", type="pil")
            #mask_output = gr.Image(label="Mask Image", type="pil")

        run_local_button = gr.Button(label="Generate", value="Run")

    run_local_button.click(fn=run_local, 
                           inputs=[base, 
                                   ref,
                                   base_mask,
                                   ref_mask,
                                   process_based_mask, # process_based_mask
                                   num_samples,
                                   shift,
                                   save_dir,
                                   reference_mask_refine,
                                   strength, 
                                   ddim_steps, 
                                   scale, 
                                   seed,
                                   enable_shape_control, 
                                   use_inpainting,
                                   ddim_inversion,
                                   share_attention,
                                   start_step,
                                   energy_score,
                                   adain_weight,
                                   adaptive_mask,
                                   ], 
                           outputs=[baseline_gallery, ref_output, jpg_output, hint_output, target_background_output]
                           #outputs=[baseline_gallery]
                        )

demo.launch(server_name="0.0.0.0")
