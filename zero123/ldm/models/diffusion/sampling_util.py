import torch
import numpy as np
from contextlib import nullcontext
from torch import autocast


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions.
    From https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/utils.py"""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def renorm_thresholding(x0, value):
    # renorm
    pred_max = x0.max()
    pred_min = x0.min()
    pred_x0 = (x0 - pred_min) / (pred_max - pred_min)  # 0 ... 1
    pred_x0 = 2 * pred_x0 - 1.  # -1 ... 1

    s = torch.quantile(
        rearrange(pred_x0, 'b ... -> b (...)').abs(),
        value,
        dim=-1
    )
    s.clamp_(min=1.0)
    s = s.view(-1, *((1,) * (pred_x0.ndim - 1)))

    # clip by threshold
    # pred_x0 = pred_x0.clamp(-s, s) / s  # needs newer pytorch  # TODO bring back to pure-gpu with min/max

    # temporary hack: numpy on cpu
    pred_x0 = np.clip(pred_x0.cpu().numpy(), -s.cpu().numpy(), s.cpu().numpy()) / s.cpu().numpy()
    pred_x0 = torch.tensor(pred_x0).to(self.model.device)

    # re.renorm
    pred_x0 = (pred_x0 + 1.) / 2.  # 0 ... 1
    pred_x0 = (pred_max - pred_min) * pred_x0 + pred_min  # orig range
    return pred_x0


def norm_thresholding(x0, value):
    s = append_dims(x0.pow(2).flatten(1).mean(1).sqrt().clamp(min=value), x0.ndim)
    return x0 * (value / s)


def spatial_norm_thresholding(x0, value):
    # b c h w
    s = x0.pow(2).mean(1, keepdim=True).sqrt().clamp(min=value)
    return x0 * (value / s)


@torch.no_grad()
def sample_model_zero123(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, T):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            if input_im.shape[0] != 1:
                batch_size = input_im.shape[0]
            else:
                batch_size = n_samples
            
            c_untiled = model.get_learned_conditioning(input_im)
            c = c_untiled.tile(n_samples, 1, 1)
            # T = torch.tensor([math.radians(x), math.sin(
            #     math.radians(y)), math.cos(math.radians(y)), z])
            T_old = T
            
            T = T[:, None, :].repeat(n_samples, 1, 1).to(c.device).to(torch.float32)
            
            # import pdb
            # pdb.set_trace()
            
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                # uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                # uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
                uc['c_concat'] = [torch.zeros_like(cond['c_concat'][0]).to(c.device)]  # (n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            # import pdb
            # pdb.set_trace()
                
            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)  # .cpu()