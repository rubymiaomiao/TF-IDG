import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
import numpy as np

class MyControledUnetmodel(UNetModel):
    
    '''Input block 0 output shape: torch.Size([1, 320, 64, 64])
        Input block 1 output shape: torch.Size([1, 320, 64, 64])
        Input block 2 output shape: torch.Size([1, 320, 64, 64])
        Input block 3 output shape: torch.Size([1, 320, 32, 32])
        Input block 4 output shape: torch.Size([1, 640, 32, 32])
        Input block 5 output shape: torch.Size([1, 640, 32, 32])
        Input block 6 output shape: torch.Size([1, 640, 16, 16])
        Input block 7 output shape: torch.Size([1, 1280, 16, 16])
        Input block 8 output shape: torch.Size([1, 1280, 16, 16])
        Input block 9 output shape: torch.Size([1, 1280, 8, 8])
        Input block 10 output shape: torch.Size([1, 1280, 8, 8])
        Input block 11 output shape: torch.Size([1, 1280, 8, 8])
        Middle block output shape: torch.Size([1, 1280, 8, 8])
        Extracted feature at index 0 shape: torch.Size([1, 1280, 8, 8])
        Extracted feature at index 1 shape: torch.Size([1, 1280, 8, 8])
        Extracted feature at index 2 shape: torch.Size([1, 1280, 16, 16])
        Extracted feature at index 3 shape: torch.Size([1, 1280, 16, 16])
        Extracted feature at index 4 shape: torch.Size([1, 1280, 16, 16])
        Extracted feature at index 5 shape: torch.Size([1, 1280, 32, 32])
        Extracted feature at index 6 shape: torch.Size([1, 640, 32, 32])
        
        module:TimestepEmbedSequential(
        (0): ResBlock(
            (in_layers): Sequential(
            (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
            (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
                (attn1): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                )
                )
                (ff): FeedForward(
                (net): Sequential(
                    (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
                )
                (attn2): MemoryEfficientCrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1024, out_features=1280, bias=False)
                (to_v): Linear(in_features=1024, out_features=1280, bias=False)
                (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
            )
            (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
        )
        )'''
    def forward(self, x, up_ft_indices, timesteps, cond, control_model, only_mid_control=None, control_scales=0, **kwargs):
        
        device = x.device  
        context = torch.cat(cond['c_crossattn'], 1).to(device)
        if cond['c_concat'] is not None:
            control = control_model(x=x, hint=torch.cat(cond['c_concat'], 1).to(device), timesteps=timesteps, context=context)
            control = [c.to(device) * scale for c, scale in zip(control, control_scales)]
        else:
            control = None
        #print("Unet:ControledUnetmodel")

        hs = []
 
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(device)
        emb = self.time_embed(t_emb).to(device)
        h = x.type(self.dtype).to(device) 
        down_ft = []
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            if i in up_ft_indices:
                down_ft.append(h.clone())
            #print(f"Input block {i} output shape: {h.shape}")
            hs.append(h)
        h = self.middle_block(h, emb, context)
        #print(f"Middle block output shape: {h.shape}")
            
        if control is not None:
            h += control.pop()
            
        up_ft = []
        for i, module in enumerate(self.output_blocks):
            #print("layer:",i) #0,1,2,3
            if i > np.max(up_ft_indices):
                break
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1).to(device)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1).to(device)
            h = module(h, emb, context)
            
            # if isinstance(module[-1], AttentionBlock):
            #   attention_maps.append(module[-1].get_attention_map())
            if i in up_ft_indices:
                up_ft.append(h.clone())
            #print(f"Extracted feature at index {i} shape: {h.shape}")

        h = h.type(x.dtype)
        output = {}
        output['up_ft'] = up_ft
        output['down_ft'] = down_ft
        #return self.out(h)
        return output