import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
import torch.nn as nn
from torchvision.utils import save_image

from sklearn.decomposition import PCA
from math import sqrt
import torchvision.transforms as T
from PIL import Image


feature_maps_path = f"./visualization/feature_maps/qkvs"

class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        
        return out
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        #print("attention base")
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []
        self.self_k_step = []
        self.self_v_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.self_attns_step.clear()
        self.cross_attns_step.clear()
        self.self_k_step.clear()
        self.self_v_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
                self.self_k_step.append(k)
                self.self_v_step.append(v)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


class MutualSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD", attention_store=None):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        self.attention_store = attention_store
        print("MasaCtrl at denoising steps: ", self.step_idx)
        print("MasaCtrl at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or (self.cur_step-100) not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)

        # out_u = source  img q 
        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)

        return out


class ShareSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self, start_step=30, start_layer=10, layer_idx=None, step_idx=None, total_steps=50):
        super().__init__()
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, 16))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        
        self.source_store = {}
        self.is_store_mode = True  # Flag to indicate whether we're storing or using stored values

    def store_source(self, q, k, v, step, layer):
        """Store q, k, v for the source image"""
        #print("store_source")
        if step not in self.source_store:
            self.source_store[step] = {}
        self.source_store[step][layer] = (q.detach(), k.detach(), v.detach())

    def get_source(self, step, layer):
        """Retrieve stored q, k, v for the source image"""
        #print("get_source")
        return self.source_store.get(step, {}).get(layer, (None, None, None)) 

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        #print("ShareSelfAttentionControl")
        if self.cur_step == self.total_steps:
            #print(f"total step: {self.total_steps}")
            self.cur_step = 0
        if self.is_store_mode:
            #print("is_store_mode")
            # Store mode: just store the values and return original output
            if is_cross or self.cur_step  not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
                #print(f"is_cross:{is_cross},self.cur_step:{self.cur_step}, self.step_idx:{self.step_idx},self.cur_att_layer:{self.cur_att_layer},self.layer_idx:{self.layer_idx}")
                return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
            else:
                #print(f"Store:is_cross:{is_cross},self.cur_step:{self.cur_step}, self.step_idx:{self.step_idx},self.cur_att_layer:{self.cur_att_layer},self.layer_idx:{self.layer_idx}")
                self.store_source(q, k, v, self.cur_step, self.cur_att_layer)
                return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            # Use mode: apply mutual self-attention control
            if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
                #print(f"switch:is_cross:{is_cross},self.cur_step:{self.cur_step}, self.step_idx:{self.step_idx},self.cur_att_layer:{self.cur_att_layer},self.layer_idx:{self.layer_idx}")
                return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
            else:
                #print(f"switch:is_cross:{is_cross},self.cur_step:{self.cur_step}, self.step_idx:{self.step_idx},self.cur_att_layer:{self.cur_att_layer},self.layer_idx:{self.layer_idx}")
                q_source, k_source, v_source = self.get_source((self.cur_step), self.cur_att_layer)
                
                if q_source is not None:
                    
                    k_concat = torch.cat((k,k_source), dim=1)
                    v_concat = torch.cat((v,v_source), dim=1)

                    #k_concat shape: torch.Size([10, 2048, 64]), v_concat shape: torch.Size([10, 2048, 64])
                    sim_target = torch.einsum('b i d, b j d -> b i j', q, k_concat) * kwargs.get("scale", 1.0)
                    attn_target = sim_target.softmax(-1)
                    out = torch.einsum('b i j, b j d -> b i d', attn_target, v_concat)
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
                    return out
                else:
                    print("No stored values found for source image")# If not applying control or no stored values, use original attention
                    return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

    def switch_to_use_mode(self):
        """Switch from store mode to use mode"""
        print("switch_to_use_mode")
        self.is_store_mode = False

    def reset(self):
        super().reset()
        self.source_store.clear()
        self.is_store_mode = True
def regiter_attention_editor_ldm(model, editor: ShareSelfAttentionControl):
    """
    Register a attention editor to Stable Diffusion model, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                print("Have mask")
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)#bug

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            #print("net.__class__.__name__:",net.__class__.__name__)
            if net.__class__.__name__ == 'MemoryEfficientCrossAttention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.model.diffusion_model.named_children():
        #print("net name:",net_name)
        if "input" in net_name:
            cross_att_count += register_editor(net, 0, "input")
        elif "middle" in net_name:
            cross_att_count += register_editor(net, 0, "middle")
        elif "output" in net_name:
            cross_att_count += register_editor(net, 0, "output")
    editor.num_att_layers = cross_att_count


def pca_visualize(feature_map, filename):
        feature_map = rearrange(feature_map, 'h n m -> n (h m)')
        feature_maps_fit_data = feature_map.cpu().numpy() 
        pca = PCA(n_components=3)
        pca.fit(feature_maps_fit_data)
        feature_maps_pca = pca.transform(feature_map.cpu().numpy())  # N X 3
        pca_img = feature_maps_pca.reshape(-1, 3)  # (H * W) x 3
        h = w = int(sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        os.makedirs(feature_maps_path, exist_ok=True)
        pca_img.save(os.path.join(feature_maps_path, f"{filename}.png"))