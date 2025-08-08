"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
from atten.feature_extraction import MyControledUnetmodel

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.estimator = MyControledUnetmodel(
            use_checkpoint=True,
            image_size=32, # unused
            in_channels=4,
            out_channels=4,
            model_channels=320,
            attention_resolutions=[ 4, 2, 1 ],
            num_res_blocks=2,
            channel_mult=[ 1, 2, 4, 4 ],
            num_head_channels=64, # need to fix for flash-attn
            use_spatial_transformer=True,
            use_linear_in_transformer=True,
            transformer_depth=1,
            context_dim=1024,
            legacy=False).to(device)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float16).to(self.model.device) #fp32 --> 16

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    #@torch.no_grad()
    def  sample(self,
               S,
               batch_size,
               shape,
               conditioning=None, #input
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               attention_cond=None,
               energy_score=False,
               ref_latents=None,
               ref_mask=None,
               tar_latents=None,
               adain_weight=None,
               x0_tar=None,
               adaptive_mask=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        
        if energy_score:
            print("optimized sampling by energy score")
            samples, intermediates = self.energy_sample(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    attention_cond=attention_cond, 
                                                    ref_latents=ref_latents,
                                                    ref_mask=ref_mask,
                                                    tar_latents=tar_latents,
                                                    adain_weight=adain_weight,
                                                    x0_tar=x0_tar,
                                                    adaptive_mask=adaptive_mask)
        else:
            samples, intermediates = self.ddim_sampling(conditioning, size,
                                                        callback=callback,
                                                        img_callback=img_callback,
                                                        quantize_denoised=quantize_x0,
                                                        mask=mask, x0=x0,
                                                        ddim_use_original_steps=False,
                                                        noise_dropout=noise_dropout,
                                                        temperature=temperature,
                                                        score_corrector=score_corrector,
                                                        corrector_kwargs=corrector_kwargs,
                                                        x_T=x_T,
                                                        log_every_t=log_every_t,
                                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                                        unconditional_conditioning=unconditional_conditioning,
                                                        dynamic_threshold=dynamic_threshold,
                                                        ucg_schedule=ucg_schedule,
                                                        attention_cond=attention_cond,
                                                        tar_latents=tar_latents,
                                                        adain_weight=adain_weight
                                                        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, attention_cond=None, tar_latents=None, adain_weight=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        callback_ddim_timesteps_list = np.flip(self.ddim_timesteps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1 
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img * mask + (1. - mask) * img_orig
                if adain_weight is not None:
                    img = self.adain_latent(img, img_orig, adain_weight, mask)

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold,
                                      attention_cond=attention_cond)
            img, pred_x0 = outs
            if step in callback_ddim_timesteps_list:
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, img, step)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, attention_cond=None):
        b, *_, device = *x.shape, x.device


        if c is None:
            model_output = self.model.apply_model(x, t, unconditional_conditioning)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)# image,time,condition    
        else:
            model_t = self.model.apply_model(x, t, c)#control+dinov2
            model_uncond = self.model.apply_model(x, t, unconditional_conditioning)#control
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    
        return x_prev, pred_x0

    @torch.no_grad()
    def encode_ddim(self, img, num_steps,conditioning, unconditional_conditioning=None ,unconditional_guidance_scale=1.):
        
        print(f"Running DDIM inversion with {num_steps} timesteps")
        T = 999
        c = T// num_steps
        iterator = tqdm(range(0,T,c), desc='DDIM Inversion', total= num_steps)
        steps = list(range(0,T,c))
        latents_list = []
        pred_x0_list = []
        for i, t in enumerate(iterator):
            if i+1 >= len(steps):
                break
            img, pred = self.reverse_ddim(img, t, t_next=steps[i+1] ,c=conditioning, unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)
            latents_list.append(img)
            pred_x0_list.append(pred)
        return img, latents_list

    @torch.no_grad()
    def reverse_ddim(self, x, t,t_next, c=None, quantize_denoised=False, unconditional_guidance_scale=1.,
                     unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
        if c is None:
            e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t_tensor, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_tensor] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        alphas = self.model.alphas_cumprod #.flip(0)
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod #.flip(0)
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[t], device=device)
        a_next = torch.full((b, 1, 1, 1), alphas[t_next], device=device) #a_next = torch.full((b, 1, 1, 1), alphas[t + 1], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[t], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_next).sqrt() * e_t
        x_next = a_next.sqrt() * pred_x0 + dir_xt
        return x_next, pred_x0
    
    @torch.no_grad()
    def ddim_inversion(self, x_lat, conditioning, num_steps=50, callback=None, img_callback=None):
        """
        Use skip-step DDIM inversion to complete the inversion process with fewer steps
        Args:
            x_lat: initial latent representation [B, C, H, W]
        conditioning:
            num_steps: number of inversion steps (default 50 steps)
            callback: callback function for each step
            img_callback: image callback function
        Returns:
            x_t: noise after inversion
            intermediates: intermediate state dictionary
        """
        device = self.model.betas.device
        b = x_lat.shape[0]
        
        
        self.make_schedule(ddim_num_steps=num_steps, ddim_eta=0.0, verbose=False)
        timesteps = np.flip(self.ddim_timesteps)  
        
        x_t = x_lat
        latents_list = [x_t]
        pred_x0_list = [x_t]
        #intermediates = {'x_inter': [x_t], 'pred_x0': [x_t]}
        
        print(f"Running DDIM Inversion with {num_steps} steps")
        iterator = tqdm(enumerate(timesteps), desc='DDIM Inversion', total=len(timesteps))

        for i, step in iterator:
            index = len(timesteps) - i - 1  
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            
            # Get the current noise 
            model_output = self.model.apply_model(x_t, ts, conditioning)
            
            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x_t, ts, model_output)
            else:
                e_t = model_output
                
            # the parameters of the current time step
            a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=device)
            
            if self.model.parameterization != "v":
                pred_x0 = (x_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x_t, ts, model_output)
            
            # next step
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt
            x_t = x_prev

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, x_t, step)
            
            latents_list.append(x_t)
            pred_x0_list.append(pred_x0)
        
        return x_t, latents_list
    
    def energy_sample(self, cond, shape, repeat_noise=False,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, attention_cond=None, energy_score=None, ref_latents=None, ref_mask=None, 
                      tar_latents=None, adain_weight=None, x0_tar=None, adaptive_mask=None, energy_scale=6*1e3):
        
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        callback_ddim_timesteps_list = np.flip(self.ddim_timesteps)
        
        loss_history=[]
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            img_orig = self.model.q_sample(x0, ts)
            
            if mask is not None:
                assert x0 is not None
                
                if adaptive_mask:
                    distance_mask= self.process_mask_and_distance(mask, img, x0, i)
                else:
                    distance_mask = None                
                img = img * mask + (1. - mask) * img_orig
                if adain_weight is not None:
                    img = self.adain_latent(img, img_orig, adain_weight, mask)
                        
            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]
            
            b, *_, device = *img.shape, img.device
            
            with torch.no_grad():
                if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                    model_output = self.model.apply_model(img, ts, cond)  
                else:
                    model_t = self.model.apply_model(img, ts, cond)#control+dinov2
                    model_uncond = self.model.apply_model(img, ts, unconditional_conditioning)#control
                    model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

                if self.model.parameterization == "v":
                    e_t = self.model.predict_eps_from_z_and_v(img, ts, model_output) 
                else:
                    e_t = model_output

                if score_corrector is not None:
                    assert self.model.parameterization == "eps", 'not implemented'
                    e_t = score_corrector.modify_score(self.model, e_t, img, ts, cond, **corrector_kwargs)

            if energy_scale!=0 and i<30 and (i%2==0 or i<10):

                noise_pred_org = e_t.detach()
                img.requires_grad_(True)
                guidance, loss_edit = self.guidance_appearance(
                    latent=img, latent_noise_ref=ref_latents[-(i+2)], latent_noise_orig=tar_latents[-(i+1)],t=ts, cond=cond, 
                    energy_scale=energy_scale, mask_base_cur=mask, mask_replace_cur=ref_mask, 
                    un_cond=unconditional_conditioning, distance_mask=distance_mask)     
                #print("guidance:", guidance)
                e_t= guidance + e_t
                img.requires_grad_(False)
                loss_history.append(loss_edit.item())     
            else:
                noise_pred_org=None
        
            alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if ddim_use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if ddim_use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
            
            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0) 

            if dynamic_threshold is not None:
                raise NotImplementedError()

            # direction pointing to x_t
            if noise_pred_org is not None:
                dir_xt = (1. - a_prev - sigma_t**2).sqrt() * noise_pred_org
            else:
                dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
                
            noise = sigma_t * noise_like(img.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            img = a_prev.sqrt() * pred_x0 + dir_xt + noise

            
            if step in callback_ddim_timesteps_list:
                    if callback: callback(i)
                    if img_callback: img_callback(pred_x0, img, step)

            if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)                
        return img, intermediates

    def guidance_appearance(
        self, 
        mask_base_cur, # target mask
        mask_replace_cur, #ref mask
        latent, #target img
        latent_noise_ref, #ref img
        latent_noise_orig,
        t, 
        cond,
        un_cond,
        energy_scale, 
        distance_mask=None,
        w_edit=1, 
        w_content=2.5, 
        up_ft_index=[4,5], 
        up_scale=2, 
        
    ):
        #cos = nn.CosineSimilarity(dim=1)   
        
        with torch.no_grad():
                
            up_ft_tar_replace = self.estimator(
                        x=latent_noise_ref,
                        timesteps=t,
                        up_ft_indices=up_ft_index,
                        cond=un_cond,
                        control_model=self.model.control_model,
                        control_scales=([0] * 13))['up_ft']
            
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
        
        latent = latent.detach().requires_grad_(True)
        
        up_ft_cur = self.estimator(
                    x=latent,
                    timesteps=t,
                    up_ft_indices=up_ft_index,
                    cond=cond,
                    control_model=self.model.control_model,
                    control_scales=self.model.control_scales)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        

        loss_edit = 0
        emd_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
        for f_id in range(len(up_ft_tar_replace)):
            mask_tar = F.interpolate(mask_base_cur, size=up_ft_cur[f_id].shape[-2:], mode='nearest')
            mask_ref = F.interpolate(mask_replace_cur, size=up_ft_tar_replace[f_id].shape[-2:], mode='nearest')
            mask_tar = mask_tar.bool()
            mask_ref = mask_ref.bool()
            up_ft_cur_vec = up_ft_cur[f_id][mask_tar.repeat(1, up_ft_cur[f_id].shape[1], 1, 1)].view(up_ft_cur[f_id].shape[1], -1).permute(1, 0)
            up_ft_tar_vec = up_ft_tar_replace[f_id][mask_ref.repeat(1, up_ft_tar_replace[f_id].shape[1], 1, 1)].view(up_ft_tar_replace[f_id].shape[1], -1).permute(1, 0)
            up_ft_cur_vec = F.normalize(up_ft_cur_vec, p=2, dim=1)
            up_ft_tar_vec = F.normalize(up_ft_tar_vec, p=2, dim=1)

     
            loss = emd_loss(up_ft_cur_vec, up_ft_tar_vec)
            loss_edit += loss
        

        loss_con = 0
        if distance_mask is not None:
            for f_id in range(len(up_ft_tar_replace)):
                mask_tar_attention = F.interpolate(distance_mask, size=up_ft_cur[f_id].shape[-2:], mode='nearest')
                mask_tar_attention = mask_tar_attention.bool()
                up_ft_cur_attention_vec = up_ft_cur[f_id][mask_tar_attention.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
                up_ft_cur_attention_vec = F.normalize(up_ft_cur_vec, p=2, dim=1)
                attetnion_loss = emd_loss(up_ft_cur_attention_vec, up_ft_tar_vec)
                loss_con = loss_con + w_content*attetnion_loss
        
        if distance_mask is not None:
            total_loss = loss_con + loss_edit
            cond_grad_edit = torch.autograd.grad(total_loss*energy_scale, latent, retain_graph=True)[0]
        else:
            cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0] #cond_grad_edit: torch.Size([1, 4, 64, 64])
         
        mask = F.interpolate(mask_base_cur.float(), (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]))
        mask = (mask>0).to(dtype=latent.dtype)
        if distance_mask is not None:
            guidance = cond_grad_edit.detach()*distance_mask*8e-2
        else:
            guidance = cond_grad_edit.detach()*mask*8e-2
        self.estimator.zero_grad()
        return guidance, loss_edit
    
    
    def adain_latent(self, content_latent, style_latent, adain_weight, mask, eps=1e-5):
        content_mean = content_latent.mean(dim=(2, 3), keepdim=True)
        content_std = content_latent.std(dim=(2, 3), keepdim=True) + eps

        # Compute the mean and standard deviation of the style latent
        style_mean = style_latent.mean(dim=(2, 3), keepdim=True)
        style_std = style_latent.std(dim=(2, 3), keepdim=True) + eps

        # Apply the AdaIN formula
        modulated_latent = style_std * (content_latent * mask - content_mean) / content_std + style_mean
        latent_adain = adain_weight * modulated_latent + (1-adain_weight)*(content_latent * mask) + (1. - mask) * style_latent
        
        return latent_adain

        
    def process_mask_and_distance(self, mask, img, x0, ts):
        dis_map = ((mask * (img - x0)) ** 2).mean(dim=1)
        dis_map = torch.where(dis_map == 0, 1, dis_map)
        threshold = dis_map.mean()
        
        binary_mask = (dis_map < threshold).float()
        binary_mask = binary_mask * mask.squeeze(1)

        binary_mask = binary_mask.unsqueeze(1) 
        binary_mask = F.interpolate(binary_mask, size=mask.shape[-2:], mode='nearest')

        return binary_mask
    
    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec