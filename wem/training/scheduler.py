# Taken from https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/schedulers/flow_match.py
import torch

class CausalSwinFlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=50,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=50, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.prev_sigmas = self.sigmas / 2
        self.curr_sigmas = self.prev_sigmas + 0.5
        self.sigmas = torch.cat([self.curr_sigmas, self.prev_sigmas], dim=0)
        self.timesteps = self.sigmas * num_inference_steps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def step(self, model_output, timestep, sample, to_final=False):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(
        self,
        original_samples,
        noise,
        curr_timestep,
        next_timestep,
        chunk_size,
        cond_size=1,
        sink_size=0,
        cond_noise_sigma=0.055,
        sink_noise_sigma=0.0,
    ):
        if not isinstance(curr_timestep, torch.Tensor):
            curr_timestep = torch.tensor(curr_timestep)
        if not isinstance(next_timestep, torch.Tensor):
            next_timestep = torch.tensor(next_timestep)
        
        if curr_timestep.dim() == 0:
            curr_timestep = curr_timestep.unsqueeze(0)
        if next_timestep.dim() == 0:
            next_timestep = next_timestep.unsqueeze(0)

        device = original_samples.device
        dtype = original_samples.dtype
        curr_timestep = curr_timestep.to(device)
        next_timestep = next_timestep.to(device)
        timesteps_ref = self.timesteps.to(device)

        B = original_samples.shape[0]
        if curr_timestep.shape[0] == 1 and B > 1:
             curr_timestep = curr_timestep.expand(B)
        if next_timestep.shape[0] == 1 and B > 1:
             next_timestep = next_timestep.expand(B)

        sample = original_samples.clone()

        dists_curr = torch.abs(curr_timestep.unsqueeze(1) - timesteps_ref.unsqueeze(0))
        curr_ids = torch.argmin(dists_curr, dim=1)

        dists_next = torch.abs(next_timestep.unsqueeze(1) - timesteps_ref.unsqueeze(0))
        next_ids = torch.argmin(dists_next, dim=1)

        curr_sigmas = self.sigmas.to(device)[curr_ids].view(B, 1, 1, 1, 1).to(dtype)
        next_sigmas = self.sigmas.to(device)[next_ids].view(B, 1, 1, 1, 1).to(dtype)

        sink_size = int(max(0, sink_size))
        cond_size = int(max(1, cond_size))
        prefix_start = sink_size
        curr_start = sink_size + cond_size
        curr_end = curr_start + chunk_size

        if sink_size > 0 and sink_noise_sigma > 0:
            sample[:, :, :sink_size, :, :] = (
                (1 - sink_noise_sigma) * original_samples[:, :, :sink_size, :, :]
                + sink_noise_sigma * noise[:, :, :sink_size, :, :]
            )

        if cond_noise_sigma > 0:
            sample[:, :, prefix_start:curr_start, :, :] = (
                (1 - cond_noise_sigma) * original_samples[:, :, prefix_start:curr_start, :, :]
                + cond_noise_sigma * noise[:, :, prefix_start:curr_start, :, :]
            )

        sample[:, :, curr_start:curr_end, :, :] = (
            (1 - curr_sigmas) * original_samples[:, :, curr_start:curr_end, :, :] + 
            curr_sigmas * noise[:, :, curr_start:curr_end, :, :]
        )

        sample[:, :, curr_end:, :, :] = (
            (1 - next_sigmas) * original_samples[:, :, curr_end:, :, :] + 
            next_sigmas * noise[:, :, curr_end:, :, :]
        )

        return sample

    def training_target(self, sample, noise):
        target = noise - sample
        return target
    
    def get_loss_mask(self, chunk_size, curr_timestep, next_timestep, device, dtype, cond_size=1, sink_size=0):
        if not isinstance(curr_timestep, torch.Tensor):
            curr_timestep = torch.tensor(curr_timestep)
        if not isinstance(next_timestep, torch.Tensor):
            next_timestep = torch.tensor(next_timestep)

        if curr_timestep.dim() == 0:
            curr_timestep = curr_timestep.unsqueeze(0)
        if next_timestep.dim() == 0:
            next_timestep = next_timestep.unsqueeze(0)
            
        curr_timestep = curr_timestep.to(device)
        next_timestep = next_timestep.to(device)
        timesteps_ref = self.timesteps.to(device)
        
        B = max(curr_timestep.shape[0], next_timestep.shape[0])
        
        if curr_timestep.shape[0] == 1 and B > 1:
             curr_timestep = curr_timestep.expand(B)
        if next_timestep.shape[0] == 1 and B > 1:
             next_timestep = next_timestep.expand(B)

        sink_size = int(max(0, sink_size))
        cond_size = int(max(1, cond_size))
        T = sink_size + cond_size + 2 * chunk_size

        dists_curr = torch.abs(curr_timestep.unsqueeze(1) - timesteps_ref.unsqueeze(0))
        curr_ids = torch.argmin(dists_curr, dim=1)

        dists_next = torch.abs(next_timestep.unsqueeze(1) - timesteps_ref.unsqueeze(0))
        next_ids = torch.argmin(dists_next, dim=1)

        curr_weights = self.linear_timesteps_weights.to(device)[curr_ids].view(B, 1, 1, 1, 1).to(dtype)
        next_weights = self.linear_timesteps_weights.to(device)[next_ids].view(B, 1, 1, 1, 1).to(dtype)
        
        curr_start = sink_size + cond_size
        curr_end = curr_start + chunk_size
        mask = torch.zeros((B, 1, T, 1, 1), device=device, dtype=dtype)
        mask[:, :, curr_start:curr_end, :, :] = curr_weights
        mask[:, :, curr_end:, :, :] = next_weights

        return mask
    
