import torch
from einops import rearrange
from .visctrl_utils import AttentionBase



class SelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }

    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
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
        print("VisCtrl at denoising steps: ", self.step_idx)
        print("VisCtrl at U-Net layers: ", self.layer_idx)


    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        # 8,4096,40
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        # sim = torch.einsum("h i d, h j d -> h i j", q, k) * 1.8
        attn = sim.softmax(-1)

        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        # cross_attention不做考虑
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        # qu/qc = [16,1024,80]
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        # attnu/attnc = [16,1024,1024]
        attnu, attnc = attn.chunk(2)

        # 2 1024 640
        out_u_ref_src = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], sim, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_u_ref_tar = self.attn_batch(qu[num_heads:num_heads*2], ku[num_heads:num_heads*2], vu[num_heads:num_heads*2], sim, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_u_tar_src = self.attn_batch(qu[num_heads*2:num_heads*3], ku[num_heads*2:num_heads*3], vu[num_heads*2:num_heads*3], sim, attnu, is_cross, place_in_unet, num_heads, **kwargs)
        out_u_tar_tar = self.attn_batch(qu[-num_heads:], ku[:num_heads], vu[:num_heads], sim, attnu, is_cross, place_in_unet, num_heads, **kwargs)

        out_c_ref_src = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], sim, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_ref_tar = self.attn_batch(qc[num_heads:num_heads*2], kc[num_heads:num_heads*2], vc[num_heads:num_heads*2], sim, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_tar_ref = self.attn_batch(qc[num_heads*2:num_heads*3], kc[num_heads*2:num_heads*3], vc[num_heads*2:num_heads*3], sim, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_tar_tar = self.attn_batch(qc[-num_heads:], kc[:num_heads], vc[:num_heads], sim, attnc, is_cross, place_in_unet, num_heads, **kwargs)

        out = torch.cat([out_u_ref_src, out_u_ref_tar,out_u_tar_src,out_u_tar_tar, out_c_ref_src, out_c_ref_tar, out_c_tar_ref, out_c_tar_tar], dim=0)

        # # 2 1024 640
        # out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet,
        #                         num_heads, **kwargs)
        # out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet,
        #                         num_heads, **kwargs)
        # out = torch.cat([out_u, out_c], dim=0)

        return out

