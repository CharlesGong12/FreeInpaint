import sys
import os
sys.path.append(os.getcwd())
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import PIL.Image
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.checkpoint import checkpoint

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.models import AsymmetricAutoencoderKL
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import deprecate, is_accelerate_available, is_accelerate_version, logging

from powerpaint.utils.ptp_utils import AttendExciteAttnProcessor, AttentionStore
from powerpaint.utils.attn_utils import fn_smoothing_func
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline, prepare_mask_and_masked_image

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class StableDiffusionInpaintOptNoGuidancePipeline(StableDiffusionInpaintPipeline):
    def _encode_prompt(
        self,
        promptA,
        promptB,
        t,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_promptA=None,
        negative_promptB=None,
        t_nag=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        prompt = promptA
        negative_prompt = negative_promptA

        if promptA is not None and isinstance(promptA, str):
            batch_size = 1
        elif promptA is not None and isinstance(promptA, list):
            batch_size = len(promptA)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                promptA = self.maybe_convert_prompt(promptA, self.tokenizer)

            text_inputsA = self.tokenizer(
                promptA,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_inputsB = self.tokenizer(
                promptB,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_idsA = text_inputsA.input_ids
            text_input_idsB = text_inputsB.input_ids
            untruncated_ids = self.tokenizer(promptA, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_idsA.shape[-1] and not torch.equal(
                text_input_idsA, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputsA.attention_mask.to(device)
            else:
                attention_mask = None
            attention_mask_cond = text_inputsA.attention_mask.to(device)

            prompt_embedsA = self.text_encoder(
                text_input_idsA.to(device),
                attention_mask=attention_mask,
            )
            prompt_embedsA = prompt_embedsA[0]

            prompt_embedsB = self.text_encoder(
                text_input_idsB.to(device),
                attention_mask=attention_mask,
            )
            prompt_embedsB = prompt_embedsB[0]
            prompt_embeds = prompt_embedsA * (t) + (1 - t) * prompt_embedsB

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokensA: List[str]
            uncond_tokensB: List[str]
            if negative_prompt is None:
                uncond_tokensA = [""] * batch_size
                uncond_tokensB = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokensA = [negative_promptA]
                uncond_tokensB = [negative_promptB]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokensA = negative_promptA
                uncond_tokensB = negative_promptB

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokensA = self.maybe_convert_prompt(uncond_tokensA, self.tokenizer)
                uncond_tokensB = self.maybe_convert_prompt(uncond_tokensB, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_inputA = self.tokenizer(
                uncond_tokensA,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_inputB = self.tokenizer(
                uncond_tokensB,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_inputA.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embedsA = self.text_encoder(
                uncond_inputA.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embedsB = self.text_encoder(
                uncond_inputB.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embedsA[0] * (t_nag) + (1 - t_nag) * negative_prompt_embedsB[0]

            # negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds, attention_mask_cond
    @torch.enable_grad()
    def cond_fn(
            self,
            latent_model_input,
            latents,
            mask,
            text_embeddings,
            timestep,
            cross_attention_kwargs,
            index,
            overall_text_input,
            prompt_text_input,
            noise_pred_original,
            reward_guidance_scale,
            task_class=None,
    ):
        latents = latents.detach().requires_grad_()

        if task_class is not None:
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
                task_class=task_class,
            )[0]    # rerun unet for gradient calculation
        else:
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]    # rerun unet for gradient calculation

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler, DPMSolverMultistepScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            sample = latents - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")
        
        sample = 1 / self.vae.config.scaling_factor * sample
        image = self.vae.decode(sample, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        overall_score = self.overall_reward(overall_text_input, image) * self.overall_reward_scale
        prompt_score = self.prompt_reward(prompt_text_input, image, mask) * self.prompt_reward_scale
        harmonic_score, _ = self.harmonic_reward(image, mask)
        harmonic_score = harmonic_score * self.harmonic_reward_scale

        total_score = (overall_score + prompt_score + harmonic_score) * reward_guidance_scale
        grads = torch.autograd.grad(total_score, latents)[0]
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents.detach() + grads * (sigma**2)
            noise_pred = noise_pred_original
        else:
            # noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
            # noise_pred = noise_pred_original - grads
            noise_pred = noise_pred_original - torch.sqrt(alpha_prod_t) * grads

        return noise_pred, latents

    def fn_calc_kld_loss_func(self, log_var, mu):
        return torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp()), dim=0)
    
    def fn_compute_loss(
        self, 
        mask: torch.Tensor,
        attention_mask: torch.Tensor,
        smooth_attentions: bool = True,
        smooth_mask: bool = False,
        attention_res: Tuple[int, int] = (16, 16),) -> torch.Tensor:

        # smooth the mask
        mask = torch.nn.functional.interpolate(mask, size=attention_res, mode='bicubic').squeeze(0)   # [1, res, res]
        if smooth_mask:
            mask = fn_smoothing_func(mask[0])
            mask = mask.unsqueeze(2)   # [res, res, 1]
        else:
            mask = mask[0].unsqueeze(2)   # [res, res, 1]
        
        # -----------------------------
        # cross-attention loss
        # -----------------------------
        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"), is_cross=True)   # [res, res, num_tokens]
        
        # cross attention map preprocessing
        cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:] # already passed through softmax
        attention_mask = attention_mask[:, 1:] # [1, num_tokens]
        nonpad_len = attention_mask.sum() - 1
        cross_attention_maps = cross_attention_maps[:, :, :nonpad_len]
        if smooth_attentions: cross_attention_maps = fn_smoothing_func(cross_attention_maps)
        cross_attention_maps = (cross_attention_maps - cross_attention_maps.min()) / (cross_attention_maps.max() - cross_attention_maps.min() + 1e-8)
        masked_cross_attn = cross_attention_maps * mask
        masked_cross_attn_sum = masked_cross_attn.sum(dim=(0, 1))
        unmasked_cross_attn = cross_attention_maps * (1 - mask)
        unmasked_cross_attn_sum = unmasked_cross_attn.sum(dim=(0, 1))
        cross_attn_loss = (-masked_cross_attn_sum + unmasked_cross_attn_sum).mean()
        
        # ----------------------------
        # self-attention loss
        # ----------------------------
        self_attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"), is_cross=False)   # [res, res, res*res]
        if smooth_attentions: self_attention_maps = fn_smoothing_func(self_attention_maps)
        self_attention_maps = (self_attention_maps - self_attention_maps.min()) / (self_attention_maps.max() - self_attention_maps.min() + 1e-8)
        self_attention_maps = self_attention_maps.flatten(start_dim=0, end_dim=1)  # [res*res, res*res]

        mask_flatten = mask.flatten(start_dim=0, end_dim=1) # [res*res, 1]
        mask_attention = mask_flatten @ mask_flatten.T  # [res*res, res*res]
        mask_to_non_mask_attention = mask_flatten @ (1 - mask_flatten).T  # [res*res, res*res]

        masked_self_attn = self_attention_maps * mask_attention
        masked_self_attn_sum = masked_self_attn.sum(dim=-1)
        masked_to_non_mask_self_attn = self_attention_maps * mask_to_non_mask_attention
        masked_to_non_mask_self_attn_sum = masked_to_non_mask_self_attn.sum(dim=-1)
        self_attn_loss = -masked_self_attn_sum.mean() + masked_to_non_mask_self_attn_sum.mean()
        self_attn_loss = self_attn_loss * self.self_attn_loss_scale

        cross_attn_loss = cross_attn_loss * torch.ones(1).to(self._execution_device)
        self_attn_loss  = self_attn_loss * torch.ones(1).to(self._execution_device)

        joint_loss = cross_attn_loss * 1. +  self_attn_loss * 1.

        return joint_loss, cross_attn_loss, self_attn_loss
    
    def fn_initno(
        self,
        latents: torch.Tensor,
        mask: torch.Tensor,
        masked_image_latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        cross_attention_kwargs: Dict[str, Any],
        mask_for_loss: torch.Tensor,
        attention_mask: torch.Tensor,
        initno_lr: float = 1e-1,
        max_step: int = 50,
        attn_res: Tuple[int, int] = (16, 16),
        round: int = 0,
        tau_cross_attn: float = 0.1,
        tau_self_attn: float = 0.1,
        tau_kld: float = 0.003,
        kld_max_iter: int = 10000,
        num_inference_steps: int = 50,
        device: str = "",
        denoising_step_for_loss: int = 1,
        guidance_scale: int = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        do_classifier_free_guidance: bool = False,
        task_class: Union[torch.Tensor, float, int] = None,
    ):

        latents = latents.clone().detach()
        log_var, mu = torch.zeros_like(latents), torch.zeros_like(latents)
        log_var, mu = log_var.clone().detach().requires_grad_(True), mu.clone().detach().requires_grad_(True)
        optimizer = SGD([log_var, mu], lr=initno_lr, momentum=0.9)

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        optimization_succeed = False
        for iteration in range(max_step):# tqdm(range(max_step)):
            optimized_latents = latents * (torch.exp(0.5 * log_var)) + mu
            latent_model_input = torch.cat([optimized_latents, mask, masked_image_latents], dim=1)
            # prepare scheduler
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # loss records
            joint_loss_list, cross_attn_loss_list, self_attn_loss_list = [], [], []
            
            # denoising loop
            for i, t in enumerate(timesteps):
                if i >= denoising_step_for_loss: break

                # Forward pass of denoising with text conditioning
                if task_class is not None:
                    noise_pred_text = self.unet(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=text_embeddings[1].unsqueeze(0),
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                        task_class=task_class,
                        )[0]
                else:
                    noise_pred_text = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings[1].unsqueeze(0),
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                        )[0]

                joint_loss, cross_attn_loss, self_attn_loss = self.fn_compute_loss(
                    mask_for_loss,
                    attention_mask,
                    attention_res=attn_res,)
                joint_loss_list.append(joint_loss), cross_attn_loss_list.append(cross_attn_loss), self_attn_loss_list.append(self_attn_loss)

                if denoising_step_for_loss > 1:
                    with torch.no_grad():
                        if task_class is not None:
                            noise_pred_uncond = self.unet(
                                sample=latent_model_input,
                                timestep=t,
                                encoder_hidden_states=text_embeddings[1].unsqueeze(0),
                                cross_attention_kwargs=cross_attention_kwargs,
                                return_dict=False,
                                task_class=task_class,
                            )[0]
                        else:
                            noise_pred_uncond = self.unet(
                                sample=latent_model_input,
                                timestep=t,
                                encoder_hidden_states=text_embeddings[1].unsqueeze(0),
                                cross_attention_kwargs=cross_attention_kwargs,
                                return_dict=False,
                                task_class=task_class,
                            )[0]

                    if do_classifier_free_guidance: noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    optimized_latents = self.scheduler.step(noise_pred, t, optimized_latents, **extra_step_kwargs).prev_sample
                
            joint_loss      = sum(joint_loss_list) / denoising_step_for_loss
            cross_attn_loss = max(cross_attn_loss_list)
            self_attn_loss  = max(self_attn_loss_list)

            kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            lambda_kld = 500 + kld_loss * 1e5
            joint_loss = joint_loss + lambda_kld * kld_loss
            
            # print loss records
            joint_loss_list         = [_.item() for _ in joint_loss_list]
            cross_attn_loss_list    = [_.item() for _ in cross_attn_loss_list]
            self_attn_loss_list     = [_.item() for _ in self_attn_loss_list]

            if cross_attn_loss < tau_cross_attn and self_attn_loss < tau_self_attn:
                optimization_succeed = True
                break
  
            self.unet.zero_grad()
            optimizer.zero_grad()
            joint_loss = joint_loss.mean()
            if kld_loss > tau_kld: break
            joint_loss.backward()
            optimizer.step()


        # update kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
        while kld_loss > tau_kld and kld_max_iter > 0:
            kld_max_iter -= 1
            self.unet.zero_grad()
            optimizer.zero_grad()
            kld_loss = kld_loss.mean() * 100
            kld_loss.backward()
            optimizer.step()
            kld_loss = self.fn_calc_kld_loss_func(log_var, mu)

        optimized_latents = (latents * (torch.exp(0.5 * log_var)) + mu).clone().detach()
        return optimized_latents, optimization_succeed, joint_loss

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(attnstore=self.attention_store, place_in_unet=place_in_unet)

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    @torch.no_grad()
    def __call__(
        self,
        promptA: Union[str, List[str]] = None,
        promptB: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        tradoff: float = 1.0,
        tradoff_nag: float = 1.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_promptA: Optional[Union[str, List[str]]] = None,
        negative_promptB: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        task_class: Union[torch.Tensor, float, int] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to be inpainted (which parts of the image to be masked
                out with `mask_image` and repainted according to `prompt`).
            mask_image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to mask `image`. White pixels in the mask are repainted
                while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a single channel
                (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3, so the
                expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInpaintPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

        >>> init_image = download_image(img_url).resize((512, 512))
        >>> mask_image = download_image(mask_url).resize((512, 512))

        >>> pipe = StableDiffusionInpaintPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        >>> image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        prompt = promptA
        negative_prompt = negative_promptA
        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        overall_text_input = self.overall_reward.process_text(prompt)
        prompt_text_input = self.prompt_reward.process_text(prompt)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, attention_mask = self._encode_prompt(
            promptA,
            promptB,
            tradoff,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_promptA,
            negative_promptB,
            tradoff_nag,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        mask, masked_image, init_image = prepare_mask_and_masked_image(image, mask, height, width, return_image=True)
        mask_condition = mask.clone()

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # Optimize Noise
        attn_res = int(np.ceil(width/32)), int(np.ceil(height/32))
        self.attention_store = AttentionStore(attn_res)
        self.register_attention_control()

        if self.opt_noise_steps>0:
            max_round = 5
            with torch.enable_grad():
                optimized_latents_pool = []
                for round in range(max_round):
                    optimized_latents, optimization_succeed, loss = self.fn_initno(
                        latents,
                        mask.chunk(2)[1] if do_classifier_free_guidance else mask,
                        masked_image_latents.chunk(2)[1] if do_classifier_free_guidance else masked_image_latents,
                        prompt_embeds,
                        cross_attention_kwargs,
                        1-(mask_condition < 0.5).to(dtype=latents.dtype, device=device),
                        attention_mask,
                        initno_lr=self.initno_lr,
                        max_step=self.opt_noise_steps,
                        attn_res=attn_res,
                        round=round,
                        num_inference_steps=num_inference_steps,
                        device=device,
                        denoising_step_for_loss=1,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        eta=eta,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        task_class=task_class,
                    )
                    optimized_latents_pool.append((loss, round, optimized_latents.clone(), latents.clone(), optimization_succeed))
                    if optimization_succeed: break

                    latents_outputs = self.prepare_latents(
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents,
                        image=init_image,
                        timestep=latent_timestep,
                        is_strength_max=is_strength_max,
                        return_noise=True,
                        return_image_latents=return_image_latents,
                    )

                    if return_image_latents:
                        latents, noise, image_latents = latents_outputs
                    else:
                        latents, noise = latents_outputs
                
                optimized_latents_pool.sort()

                # nan
                if not torch.isnan(optimized_latents_pool[0][0]):
                    latents = optimized_latents_pool[0][2]

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                if task_class is not None:
                    noise_pred = self.unet(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                        task_class=task_class,
                    )[0]
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                text_embeddings_guidance = (
                    prompt_embeds.chunk(2)[1] if do_classifier_free_guidance else prompt_embeds
                )
                mask_guidance = mask_condition.clone().detach()
                mask_guidance = (mask_guidance < 0.5).to(dtype=latents.dtype, device=device)
                mask_guidance = 1 - mask_guidance
                latent_model_input_guidance = latent_model_input.chunk(2)[1] if do_classifier_free_guidance else None

                if self.reward_guidance_scale > 0:
                    noise_pred, latents = self.cond_fn(
                        latent_model_input_guidance,
                        latents,
                        mask_guidance,
                        text_embeddings_guidance,
                        t,
                        cross_attention_kwargs,
                        i,
                        overall_text_input,
                        prompt_text_input,
                        noise_pred,
                        self.reward_guidance_scale,
                        task_class,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents[:1]
                    init_mask = mask[:1]

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 1. VAE Decode
        if not output_type == "latent":
            condition_kwargs = {}
            if isinstance(self.vae, AsymmetricAutoencoderKL):
                init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
                init_image_condition = init_image.clone()
                init_image = self._encode_vae_image(init_image, generator=generator)
                mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
                condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
            image_pred = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, **condition_kwargs)[0]
            image_pred, has_nsfw_concept = self.run_safety_checker(image_pred, device, prompt_embeds.dtype)
        else:
            return StableDiffusionPipelineOutput(images=latents, nsfw_content_detected=None)

        if has_nsfw_concept is None:
            do_denormalize = [True] * image_pred.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # 2. Post-process to Numpy (0-1 float)
        image_pred_np = self.image_processor.postprocess(image_pred, output_type="numpy", do_denormalize=do_denormalize)

        # 3. Blending Logic
        mask_np = mask_condition.clone().detach()
        mask_np = (mask_np < 0.5).to(dtype=latents.dtype, device=device)
        mask_np = mask_np.cpu().numpy()
        mask_np = mask_np.astype(np.float32)
        mask_np = 1 - mask_np  # 1 indicates the area to inpaint
        # 处理通道维度 (H, W) -> (H, W, 1)
        if mask_np.ndim == 2:
            mask_np = mask_np[:, :, np.newaxis]
        elif mask_np.shape[2] == 3:
            mask_np = mask_np[:, :, 0:1] # 取单通道
        elif mask_np.ndim == 4: # (1,1,512,512)
            mask_np = mask_np[0,0,:,:,np.newaxis]

        mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255.0
        mask_blurred = mask_blurred[:, :, np.newaxis]  # 保持通道维度
        
        # 混合系数：基于原始 mask 和模糊 mask 的结合
        # logic: 1-(1-mask)*(1-blurred) 相当于并集柔化
        blend_mask = 1 - (1 - mask_np) * (1 - mask_blurred)

        # 预处理原始图片 Init Image
        if isinstance(image, PIL.Image.Image):
            init_image_np = np.array(image.resize((width, height))).astype(np.float32) / 255.0
        else:
            init_image_np = image.chunk(2)[1].cpu().numpy() if self.do_classifier_free_guidance else image.cpu().numpy()
            init_image_np = np.transpose(init_image_np, (0, 2, 3, 1))  # (B,C,H,W) -> (B,H,W,C)
            init_image_np = np.clip(init_image_np / 2 + 0.5, 0, 1)
            init_image_np = np.array([cv2.resize(img, (width, height)) for img in init_image_np])

        blended_images = []
        for i in range(len(image_pred_np)):
            # Pixel-wise blending: 原图 * (1-mask) + 生成图 * mask
            pred = image_pred_np[i]
            init = init_image_np[i]
            blended = init * (1 - blend_mask) + pred * blend_mask

            blended = np.clip(blended, 0, 1)
            blended_images.append(blended)
        
        # 4. Convert to Final Output Format (usually PIL)
        if output_type == "pil":
            final_images = self.image_processor.numpy_to_pil(np.array(blended_images))
        else:
            final_images = np.array(blended_images)


        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (final_images, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=final_images, nsfw_content_detected=has_nsfw_concept)