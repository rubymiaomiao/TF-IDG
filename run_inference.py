import cv2
import einops
import numpy as np
import torch
import random
import gc
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image
import glob
from atten.shareatten import *



save_memory = False #save memory

config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file
use_interactive_seg = config.config_file

model = create_model(model_config).cpu()

if save_memory:
    torch.cuda.set_per_process_memory_fraction(0.6, device=torch.device("cuda:0"))  # Limit usage to 60% of GPU
    state_dict = load_state_dict(model_ckpt, location='cpu')
    state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.cuda()
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
    print("save_memory")
    enable_sliced_attention()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.backends.cudnn.benchmark = True

def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask

def process_image(image,num_samples):
    image = torch.from_numpy(image).float().cuda()
    image = torch.stack([image for _ in range(num_samples)], dim=0)
    image = einops.rearrange(image, 'b h w c -> b c h w').contiguous()
    return image

def process_pairs(ref_image, ref_mask, tar_image, tar_mask, enable_shape_control = False):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]
    ref_background = ref_image[y1:y2,x1:x2]
    cropped_ref_mask = ref_mask


    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug 
    masked_ref_image_aug = masked_ref_image

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])
    tar_box_yyxx_full = tar_box_yyxx
    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    cropped_tar_mask = tar_mask[y1:y2,x1:x2]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    if enable_shape_control:
        collage_mask = np.stack([cropped_tar_mask,cropped_tar_mask,cropped_tar_mask],-1)

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    cropped_tar_mask = np.expand_dims(cropped_tar_mask, axis=-1)
    cropped_tar_mask = pad_to_square(cropped_tar_mask, pad_value = 0, random = False).astype(np.uint8)
    cropped_ref_mask = np.expand_dims(cropped_ref_mask, axis=-1)
    cropped_ref_mask = pad_to_square(cropped_ref_mask, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)
    ref_background = pad_to_square(ref_background, pad_value = 0, random = False).astype(np.uint8)
    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    cropped_tar_mask = cv2.resize(cropped_tar_mask.astype(np.uint8), (512,512)).astype(np.float32)
    ref_background = cv2.resize(ref_background.astype(np.uint8), (512,512)).astype(np.float32)
    cropped_ref_mask = cv2.resize(cropped_ref_mask.astype(np.uint8), (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    ref_background = ref_background / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), 
                jpg=cropped_target_image.copy(), 
                hint=collage.copy(), 
                ref_background=ref_background.copy(),
                mask=cropped_tar_mask,
                ref_mask=cropped_ref_mask,
                extra_sizes=np.array([H1, W1, H2, W2]), 
                tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ),
                tar_box_yyxx=np.array(tar_box_yyxx_full),
    )
    return item

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
                           tar_mask):
    
    # ==================
    num_samples = 1 
    strength = 1  
    ddim_steps = 50 
    scale = 3.5   
    seed = -1  
    eta = 0.0 
    start_step = 25
    guess_mode = False
    H,W = 512,512
    model.control_scales = ([strength] * 13)
    adain_weight = 0.5
    energy_score = True
    adaptive_mask = True
    
    # ====================
    raw_background = tar_image.copy()
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask, enable_shape_control = True )
    ref = item['ref']
    hint = item['hint']

    if save_memory:
        model.low_vram_shift(is_diffusing=False)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    if seed == -1:
        seed = random.randint(0, 65535)
    else:
        seed_everything(seed)
    
    control = process_image(hint.copy(),num_samples)

    clip_input = process_image(ref.copy(),num_samples)

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], 
               "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    ddim_cond = {"c_concat": None, 
               "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    # === Encoder ========
    target_img = process_image(item['jpg'],num_samples)
    # target_img = item['jpg']  
    # target_img = torch.from_numpy(target_img).float().cuda()
    # target_img = torch.stack([target_img for _ in range(num_samples)], dim=0)
    # target_img = einops.rearrange(target_img, 'b h w c -> b c h w').clone()

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
    # ==================================
    if save_memory:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("diffusion to gpu")
        model.low_vram_shift(is_diffusing=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):

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

    if save_memory:
        print("diffusion to cpu")
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    # result = x_samples[0][:,:,::-1]
    # result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    tar_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    y1,y2,x1,x2 = item['tar_box_yyxx']
    raw_background[y1:y2, x1:x2, :] = tar_image[y1:y2, x1:x2, :]

    if save_memory:
        del ref, hint
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return raw_background, item

def select_limited_images(ref_image_paths, fraction=1/3):

    num_images_to_select = max(1, int(len(ref_image_paths) * fraction))
    
    selected_images = ref_image_paths[:num_images_to_select]
    #selected_images = random.sample(ref_image_paths, num_images_to_select)
    
    return selected_images

if __name__ == '__main__': 
   
    from omegaconf import OmegaConf
    import os 

    dataset_path = "/your/dataset/path"
    
    categories= os.listdir(dataset_path)
    for category in categories:
        defect_classes = [cls for cls in os.listdir(f"{dataset_path}/{category}/test") if cls != "good"]
        for defect_class in defect_classes:
            print(f"category:{category}, defect class:{defect_class}")
            save_path = f"./Result/MVTec_few/{category}"
            tar_image_paths = glob.glob(os.path.join(dataset_path, category, "train", "good", "*.png")) #MVTec
            ref_image_paths = glob.glob(os.path.join(dataset_path, category, "test", defect_class, "*.png")) #MVTec
            selected_images = select_limited_images(ref_image_paths)
            for i in range(1):
                try:
                    ref_image_name = random.choice(selected_images)
                    tar_image_name = random.choice(tar_image_paths)
                    tar_mask_name = random.choice(selected_images)
                    ref_mask_path = ref_image_name.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")#mvtec
                    tar_mask_path = tar_mask_name.replace("/test/", "/ground_truth/").replace(".png", "_mask.png")#mvtec
                    print(f"Processing: [{i}] ref_image_name-{ref_image_name}, tar_image_name-{tar_image_name}, tar_mask_name-{tar_mask_name}")
                    ref_image = cv2.imread(ref_image_name)
                    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

                    # background image
                    gt_image = cv2.imread(tar_image_name)
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

                    ref_mask = Image.open(ref_mask_path).convert('L')
                    ref_mask = np.array(ref_mask)
                    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)
                    #ref_mask = reference_mask_augmentation(ref_mask)

                    # background mask
                    tar_mask = Image.open(tar_mask_path).convert('L')
                    tar_mask = np.array(tar_mask)
                    tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

                    # ===== Diversity ========
                    
                    object_mask = extract_foreground_mask(gt_image)
                    tar_mask = seg_mask_random_placement_region(tar_mask,object_mask)
                    #tar_mask = edge_mask_augmentation(tar_mask,object_mask) 
                    #tar_mask = mask_augmentation(tar_mask)
                    #ref_mask = reference_mask_augmentation(ref_mask)
                    #ref_image, ref_mask, tar_mask, = rotate_image_and_mask_for_transistor(ref_image, ref_mask, tar_mask)
                    if save_memory:
                        torch.cuda.empty_cache()  # Release unused but cached memory
                        torch.cuda.ipc_collect()  # Force PyTorch to release VRM

                    gen_image, item = inference_single_image(ref_image, ref_mask, gt_image.copy(), tar_mask)

                    synthesis = torch.from_numpy(gen_image).permute(2, 0, 1)
                    synthesis = synthesis.permute(1, 2, 0).numpy()
                    ref_processed = item['ref']

                    test_dir = os.path.join(save_path, "test", defect_class)
                    gt_dir = os.path.join(save_path, "ground_truth", defect_class)
                    source_dir = os.path.join(save_path, "source", defect_class)
                    os.makedirs(test_dir, exist_ok=True)
                    os.makedirs(gt_dir, exist_ok=True)
                    os.makedirs(source_dir, exist_ok=True)

                    next_index = get_next_image_index(test_dir)

                    # filename
                    source_path = os.path.splitext(os.path.basename(tar_image_name))[0]
                    #source_path = source_path.split('_')[1]
                    image_name = f"{next_index:04d}_{source_path}.png"
                    mask_name = f"{next_index:04d}_{source_path}_mask.png"

                    image_path = os.path.join(test_dir, image_name)
                    mask_path = os.path.join(gt_dir, mask_name)
                    source_image_path = os.path.join(source_dir, f"source_{image_name}")

                    cv2.imwrite(image_path, cv2.cvtColor(synthesis.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(mask_path, (tar_mask * 255).astype(np.uint8))

                    source_image = compose_images(ref_image, ref_mask, gt_image, tar_mask, synthesis, ref_processed)
                    cv2.imwrite(source_image_path, cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))

                    if save_memory:
                        del ref_image, ref_mask, gt_image, tar_mask, gen_image, item
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        gc.collect()
                except Exception as e:
                    print(f"[Error in iteration {i}] Skipping due to error: {e}") 
                continue
         

                



