# Copyright 2024 Stability AI, The HuggingFace Team and The AlimamaCreative Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import PIL

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

import random
import numpy as np
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from examples.freeinpaint.utils.ptp_utils_3 import AttnProcessor, AttentionStore
from examples.freeinpaint.utils.attn_utils_3 import fn_smoothing_func, fn_get_topk, fn_clean_mask, fn_get_otsu_mask, fn_show_attention_plus


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers.utils import load_image, check_min_version
        >>> from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
        >>> from diffusers.models.controlnet_sd3 import SD3ControlNetModel

        >>> controlnet = SD3ControlNetModel.from_pretrained(
        ...     "alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True, extra_conditioning_channels=1
        ... )
        >>> pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers",
        ...     controlnet=controlnet,
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe.text_encoder.to(torch.float16)
        >>> pipe.controlnet.to(torch.float16)
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog.png"
        ... )
        >>> mask = load_image(
        ...     "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog_mask.png"
        ... )
        >>> width = 1024
        >>> height = 1024
        >>> prompt = "A cat is sitting next to a puppy."
        >>> generator = torch.Generator(device="cuda").manual_seed(24)
        >>> res_image = pipe(
        ...     negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
        ...     prompt=prompt,
        ...     height=height,
        ...     width=width,
        ...     control_image=image,
        ...     control_mask=mask,
        ...     num_inference_steps=28,
        ...     generator=generator,
        ...     controlnet_conditioning_scale=0.95,
        ...     guidance_scale=7,
        ... ).images[0]
        >>> res_image.save(f"sd3.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusion3ControlNetInpaintinOptNoGuidancePipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        controlnet ([`SD3ControlNetModel`] or `List[SD3ControlNetModel]` or [`SD3MultiControlNetModel`]):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        controlnet: Union[
            SD3ControlNetModel, List[SD3ControlNetModel], Tuple[SD3ControlNetModel], SD3MultiControlNetModel
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_resize=True, do_convert_rgb=True, do_normalize=True
        )
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_resize=True,
            do_convert_grayscale=True,
            do_normalize=False,
            do_binarize=True,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, text_inputs.attention_mask.to(device)

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds, text_inputs.attention_mask.to(device)

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
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
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed, attention_mask_clip = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed, _ = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed, attention_mask_t5 = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed, _ = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed, _ = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed, _ = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, attention_mask_clip, attention_mask_t5

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if (
            height % (self.vae_scale_factor * self.patch_size) != 0
            or width % (self.vae_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * self.patch_size} but are {height} and {width}."
                f"You can use height {height - height % (self.vae_scale_factor * self.patch_size)} and width {width - width % (self.vae_scale_factor * self.patch_size)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    def prepare_image_with_mask(
        self,
        image,
        mask,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        # Prepare image
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        # Prepare mask
        if isinstance(mask, torch.Tensor):
            pass
        else:
            mask = self.mask_processor.preprocess(mask, height=height, width=width)
        mask = mask.repeat_interleave(repeat_by, dim=0)
        mask = mask.to(device=device, dtype=dtype)

        # Get masked image
        masked_image = image.clone()
        masked_image[(mask > 0.5).repeat(1, 3, 1, 1)] = -1

        # Encode to latents
        image_latents = self.vae.encode(masked_image).latent_dist.sample()
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = image_latents.to(dtype)

        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = 1 - mask
        control_image = torch.cat([image_latents, mask], dim=1)

        if do_classifier_free_guidance and not guess_mode:
            control_image = torch.cat([control_image] * 2)

        return control_image, mask

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.transformer.attn_processors.keys():
            if cross_att_count in self.from_where:
                place_in_transformer = cross_att_count
                attn_procs[name] = AttnProcessor(attnstore=self.attention_store, place_in_transformer=place_in_transformer,from_where=self.from_where)
            else:
                attn_procs[name] = self.transformer.attn_processors[name]
            cross_att_count += 1
        self.transformer.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = len(self.from_where)*2 # cross_att_count*2  #

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents

    def fn_initno(
            self,
            latents: torch.Tensor,
            mask: torch.Tensor,
            text_embeddings: torch.Tensor,
            attention_mask: torch.Tensor,
            pooled_text_embeddings: torch.Tensor,
            controlnet_pooled_text_embeddings: torch.Tensor,
            control_image: torch.Tensor,
            cond_scale: Union[float, List[float]],
            initno_lr: float = 1e-1,
            max_step: int = 50,
            tau_cross_attn: float = 0.1,
            tau_self_attn: float = 0.1,
            tau_kld: float = 0.003,
            kld_max_iter: int = 10000,
            num_inference_steps: int = 50,
            device: str = "",
            denoising_step_for_loss: int = 1,
            guidance_scale: float = 1.0,
            do_classifier_free_guidance: bool = False,
            attention_res: Tuple[int, int] = (64, 64),
            from_where: List[int] = None
    ):

        latents = latents.clone().detach()
        log_var, mu = torch.zeros_like(latents), torch.zeros_like(latents)
        log_var, mu = log_var.clone().detach().requires_grad_(True), mu.clone().detach().requires_grad_(True)
        optimizer = SGD([log_var, mu], lr=initno_lr, momentum=0.9)

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        optimization_succeed = False
        for iteration in tqdm(range(max_step)):

            optimized_latents = latents * (torch.exp(0.5 * log_var)) + mu

            # prepare scheduler
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # loss records
            joint_loss_list, cross_attn_loss_list, self_attn_loss_list = [], [], []

            # denoising loop
            for i, t in enumerate(timesteps):
                if i >= denoising_step_for_loss: break
                timestep = t.expand(optimized_latents.shape[0])

                control_block_samples = self.controlnet(
                    hidden_states=optimized_latents,
                    timestep=timestep,
                    encoder_hidden_states=text_embeddings[1].unsqueeze(0),  # take the positive prompt
                    pooled_projections=controlnet_pooled_text_embeddings[1].unsqueeze(0),
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    return_dict=False, )[0]


                # Forward pass of denoising with text conditioning
                noise_pred_text = self.transformer(
                    hidden_states=optimized_latents,
                    timestep=timestep,
                    encoder_hidden_states=text_embeddings[1].unsqueeze(0),  # take the positive prompt
                    pooled_projections=pooled_text_embeddings[1].unsqueeze(0),  # take the positive prompt
                    block_controlnet_hidden_states=control_block_samples,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False, )[0]

                joint_loss, cross_attn_loss, self_attn_loss = self.fn_compute_loss(mask, attention_mask,
                                                                                         attention_res=attention_res, from_where=from_where)
                joint_loss_list.append(joint_loss), cross_attn_loss_list.append(cross_attn_loss), self_attn_loss_list.append(self_attn_loss)

                if denoising_step_for_loss > 1:
                    with torch.no_grad():
                        control_block_samples = self.controlnet(
                            hidden_states=optimized_latents,
                            timestep=timestep,
                            encoder_hidden_states=text_embeddings[1].unsqueeze(0),  # take the positive prompt
                            pooled_projections=controlnet_pooled_text_embeddings[1].unsqueeze(0),
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            controlnet_cond=control_image,
                            conditioning_scale=cond_scale,
                            return_dict=False, )[0]
                        # Forward pass of denoising with text conditioning
                        noise_pred_uncond = self.transformer(
                            hidden_states=optimized_latents,
                            timestep=timestep,
                            encoder_hidden_states=text_embeddings[0].unsqueeze(0),  # take the positive prompt
                            pooled_projections=pooled_text_embeddings[0].unsqueeze(0),  # take the positive prompt
                            block_controlnet_hidden_states=control_block_samples,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False, )[0]
                        
                    if do_classifier_free_guidance: noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond)
                    else: noise_pred = noise_pred_text
                    # compute the previous noisy sample x_t -> x_t-1
                    optimized_latents = self.scheduler.step(noise_pred, t, optimized_latents, return_dict=False)[0]

            joint_loss = sum(joint_loss_list) / denoising_step_for_loss
            cross_attn_loss = max(cross_attn_loss_list)
            self_attn_loss = max(self_attn_loss_list)

            kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            lambda_kld = 500 + kld_loss * 1e6
            joint_loss = joint_loss + lambda_kld * kld_loss

            if cross_attn_loss < tau_cross_attn and self_attn_loss < tau_self_attn:
                optimization_succeed = True
                break

            self.transformer.zero_grad()
            optimizer.zero_grad()
            joint_loss = joint_loss.mean()
            if kld_loss > tau_kld: break
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=10.0)
            optimizer.step()

            # update kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
        while kld_loss > tau_kld and kld_max_iter > 0:
            kld_max_iter -= 1
            self.transformer.zero_grad()
            optimizer.zero_grad()
            kld_loss = kld_loss.mean() * 100
            kld_loss.backward()
            optimizer.step()
            kld_loss = self.fn_calc_kld_loss_func(log_var, mu)

        optimized_latents = (latents * (torch.exp(0.5 * log_var)) + mu).clone().detach()
        # if self_attn_loss <= 1e-6: self_attn_loss = self_attn_loss + 1.
        return optimized_latents, optimization_succeed, joint_loss

    def fn_compute_loss(
        self,
        mask: torch.Tensor,
        attention_mask_clip: torch.Tensor,
        smooth_attentions: bool = True,
        attention_res: Tuple[int, int] = (64, 64),
        from_where: List[int]=None) -> torch.Tensor:

        # -----------------------------
        # cross-attention response loss
        # -----------------------------
        mask = torch.nn.functional.interpolate(mask, size=attention_res, mode='bicubic').squeeze(0)   # [1, res, res]
        mask = mask[0].unsqueeze(2)   # [res, res, 1]

        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
            from_where=from_where, is_cross=True)   # [res*res, num_token]
        aggregate_cross_attention_maps = aggregate_cross_attention_maps.view(
            attention_res[0], attention_res[1], -1)

        # cross attention map preprocessing
        cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:]
        attention_mask = attention_mask_clip[:, 1:]  # remove the first token
        nonpad_len = attention_mask.sum() - 1
        cross_attention_maps = cross_attention_maps[:, :, :nonpad_len]
        if smooth_attentions: cross_attention_maps = fn_smoothing_func(cross_attention_maps)
        cross_attention_maps = (cross_attention_maps - cross_attention_maps.min()) / (cross_attention_maps.max() - cross_attention_maps.min() + 1e-8)
        masked_cross_attn = cross_attention_maps * mask
        masked_cross_attn_sum = masked_cross_attn.sum(dim=(0, 1))
        unmasked_cross_attn = cross_attention_maps * (1 - mask)
        unmasked_cross_attn_sum = unmasked_cross_attn.sum(dim=(0, 1))
        cross_attn_loss = (-masked_cross_attn_sum + unmasked_cross_attn_sum).mean()


        # ---------------------------------
        # prepare aggregated self attn maps
        # ---------------------------------
        self_attention_maps = self.attention_store.aggregate_attention(from_where=from_where, is_cross=False)
        self_attention_maps = self_attention_maps.view(attention_res[0], attention_res[1], -1)  # [res, res, res*res]
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

        joint_loss = (cross_attn_loss * 1 +  self_attn_loss * 1.) / 10

        return joint_loss, cross_attn_loss, self_attn_loss

    def fn_calc_kld_loss_func(self, log_var, mu):
        return torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp()), dim=0)

    # https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py
    @torch.enable_grad()
    def cond_fn(
            self,
            latents,
            control_image,
            mask,
            pooled_prompt_embeds,
            prompt_embeds,
            controlnet_pooled_projections,
            timestep,
            cond_scale,
            index,
            overall_text_input,
            prompt_text_input,
            noise_pred_original,
            reward_guidance_scale,
    ):
        latents = latents.detach().requires_grad_()

        with torch.no_grad():
            control_block_samples = self.controlnet(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=controlnet_pooled_projections,
                joint_attention_kwargs=self.joint_attention_kwargs,
                controlnet_cond=control_image,
                conditioning_scale=cond_scale,
                return_dict=False,
            )[0]

            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                block_controlnet_hidden_states=control_block_samples,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
        
        sigma = self.scheduler.sigmas[index]
        dt = self.scheduler.sigma_min - sigma
        sample = latents + dt * noise_pred      

        sample = (sample / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(sample, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        overall_score = self.overall_reward(overall_text_input, image) * self.overall_reward_scale
        prompt_score = self.prompt_reward(prompt_text_input, image, mask.to(image.device)) * self.prompt_reward_scale
        harmonic_score, _ = self.harmonic_reward(image, mask.to(image.device))
        harmonic_score = harmonic_score * self.harmonic_reward_scale

        total_score = (overall_score + prompt_score + harmonic_score) * reward_guidance_scale
        grads = torch.autograd.grad(total_score, latents)[0]

        # \frac{1-t}{\sqrt{(1-t)^2+t^2}}
        var_maintain_coeff = (1-sigma)/torch.sqrt((1-sigma)**2+sigma**2)
        # 1. default and better
        if self.guide_no == 1:
            noise_pred = noise_pred_original - var_maintain_coeff * grads

            # latents = latents.detach() + var_maintain_coeff * grads 
            # noise_pred = noise_pred_original
        else:
            raise ValueError("No guidance method specified")
        return noise_pred, latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: PipelineImageInput = None,
        control_mask: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        controlnet_pooled_projections: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        rerun_initno_failures: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be inpainted (which parts of the image to
                be masked out with `control_mask` and repainted according to `prompt`). For both numpy array and
                pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the
                expected shape should be `(B, C, H, W)`. If it is a numpy array or a list of arrays, the expected shape
                should be `(B, H, W, C)` or `(H, W, C)`.
            control_mask (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
                color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`. And
                for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W, 1)`, or `(H, W)`.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            controlnet_pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of controlnet input conditions.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        overall_text_input = self.overall_reward.process_text(prompt)
        prompt_text_input = self.prompt_reward.process_text(prompt)

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(self.controlnet, SD3MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            attention_mask_clip,
            attention_mask_t5
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Prepare control image
        if isinstance(self.controlnet, SD3ControlNetModel):
            control_image, mask = self.prepare_image_with_mask(
                image=control_image,
                mask=control_mask,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=False,
            )
            latent_height, latent_width = control_image.shape[-2:]
            mask = (mask.sum(1)[:,None,:,:] < 0.5).to(dtype=control_image.dtype)    # [1, 1, h, w], 1 indicates masked area to inpaint

            height = latent_height * self.vae_scale_factor
            width = latent_width * self.vae_scale_factor
            mask_for_condfn = self.mask_processor.preprocess(control_mask, height=height, width=width)
            mask_for_condfn = (mask_for_condfn > 0.5).to(dtype=control_image.dtype) # 1 is hole

        elif isinstance(self.controlnet, SD3MultiControlNetModel):
            raise NotImplementedError("MultiControlNetModel is not supported for SD3ControlNetInpaintingPipeline.")
        else:
            assert False

        if controlnet_pooled_projections is None:
            controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
        else:
            controlnet_pooled_projections = controlnet_pooled_projections or pooled_prompt_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(self.controlnet, SD3ControlNetModel) else keeps)

        max_round = self.max_round_initno
        attn_res = int(np.ceil(width/16)), int(np.ceil(height/16))
        self.attention_store = AttentionStore()
        self.register_attention_control()

        if self.opt_noise_steps > 0:
            with torch.enable_grad():
                optimized_latents_pool = []
                for round in range(max_round):
                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    optimized_latents, optimization_succeed, loss = self.fn_initno(
                        latents=latents,
                        mask=mask,
                        text_embeddings=prompt_embeds,
                        attention_mask=attention_mask_clip,
                        pooled_text_embeddings=pooled_prompt_embeds,
                        controlnet_pooled_text_embeddings=controlnet_pooled_projections,
                        control_image=control_image.chunk(2)[1] if self.do_classifier_free_guidance else control_image,
                        cond_scale=cond_scale,
                        initno_lr=self.initno_lr,
                        max_step=self.opt_noise_steps,
                        num_inference_steps=num_inference_steps,
                        device=device,
                        guidance_scale=guidance_scale,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        attention_res=attn_res,
                        from_where=self.from_where
                    )

                    optimized_latents_pool.append(
                        (loss, round, optimized_latents.clone(), latents.clone(), optimization_succeed))
                    if optimization_succeed: break

                    latents = self.prepare_latents(
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents=None,
                    )

                optimized_latents_pool.sort()

                if not torch.isnan(optimized_latents_pool[0][0]):
                    if optimized_latents_pool[0][4] is True:
                        latents = optimized_latents_pool[0][2]
                    else:
                        if not rerun_initno_failures:
                            latents = optimized_latents_pool[0][2]
                        else:
                            optimized_latents, optimization_succeed, loss = self.fn_initno(
                                latents=latents,
                                mask=mask,
                                text_embeddings=prompt_embeds[1] if self.do_classifier_free_guidance else prompt_embeds,
                                attention_mask=attention_mask_clip,
                                pooled_text_embeddings=pooled_prompt_embeds[1] if self.do_classifier_free_guidance else pooled_prompt_embeds,
                                controlnet_pooled_text_embeddings=controlnet_pooled_projections,
                                control_image=control_image.chunk(2)[1] if self.do_classifier_free_guidance else control_image,
                                cond_scale=cond_scale,
                                initno_lr=self.initno_lr,
                                max_step=self.opt_noise_steps,
                                num_inference_steps=num_inference_steps,
                                device=device,
                                guidance_scale=guidance_scale,
                                do_classifier_free_guidance=self.do_classifier_free_guidance,
                                attention_res=attn_res,
                                from_where=self.from_where
                            )
                            optimized_latents_pool.append((loss, round+1, optimized_latents.clone(), latents.clone(), optimization_succeed))
                            optimized_latents_pool.sort()
                            latents = optimized_latents_pool[0][2]


        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                # controlnet(s) inference
                control_block_samples = self.controlnet(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=controlnet_pooled_projections,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    return_dict=False,
                )[0]

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_samples,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.reward_guidance_scale > 0 and (i+1) % self.guide_per_steps == 0:
                    noise_pred, latents = self.cond_fn(
                        latents=latents,
                        control_image=control_image.chunk(2)[1] if self.do_classifier_free_guidance else control_image,
                        mask=mask_for_condfn,
                        pooled_prompt_embeds=pooled_prompt_embeds.chunk(2)[1] if self.do_classifier_free_guidance else pooled_prompt_embeds,
                        prompt_embeds=prompt_embeds.chunk(2)[1] if self.do_classifier_free_guidance else prompt_embeds,
                        controlnet_pooled_projections=controlnet_pooled_projections.chunk(2)[1] if self.do_classifier_free_guidance else controlnet_pooled_projections,
                        timestep=timestep.chunk(2)[1] if self.do_classifier_free_guidance else timestep,
                        cond_scale=cond_scale,
                        index=i,
                        overall_text_input=overall_text_input,
                        prompt_text_input=prompt_text_input,
                        noise_pred_original=noise_pred,
                        reward_guidance_scale=self.reward_guidance_scale
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 1. VAE Decode
        if not output_type == "latent":
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.to(dtype=self.vae.dtype)

            image_pred = self.vae.decode(latents, return_dict=False)[0]
            image_pred = self.image_processor.postprocess(image_pred, output_type=output_type)
        else:
            return StableDiffusion3PipelineOutput(images=latents)

        # 2. Post-process to Numpy (0-1 float)
        image_pred_np = self.image_processor.postprocess(image_pred, output_type="numpy")

        # 3. Blending Logic
        mask_np = mask_for_condfn.cpu().numpy().astype(np.float32)
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
        if isinstance(control_image, PIL.Image.Image):
            init_image_np = np.array(control_image.resize((width, height))).astype(np.float32) / 255.0
        else:
            init_image_np = control_image.chunk(2)[1].cpu().numpy() if self.do_classifier_free_guidance else control_image.cpu().numpy()
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

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (final_images,)

        return StableDiffusion3PipelineOutput(images=final_images)
    
