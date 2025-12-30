import cv2
import json
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from PIL import Image
import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from diffusers import BrushNetModel, UniPCMultistepScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from typing import Tuple, Union, Optional
import safetensors
from transformers import set_seed
import time

from examples.freeinpaint.eval.metrics_calculator import rle2mask, MetricsCalculator
from examples.freeinpaint.data.eval_data import EditBench, COCOCustom
from examples.freeinpaint.utils.images import resize
from examples.freeinpaint.utils.vis_utils import get_image_grid
from examples.freeinpaint.metrics import get_mask_bbox
from examples.freeinpaint.metrics.prefpaint import InpaintReward
from examples.freeinpaint.metrics.guidance import ImageRewardScore, PromptRewardScore
from examples.freeinpaint.pipe.pipeline_brushnet_optno_guidance import StableDiffusionBrushNetOptNoGuidancePipeline


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model paths
    parser.add_argument('--brushnet_ckpt_path', 
                        type=str, 
                        default="/local/gongchao/Models/brushnet/random_mask_brushnet_ckpt")
    parser.add_argument('--base_model_path', 
                        type=str, 
                        default="/local/gongchao/Models/brushnet/realisticVisionV60B1_v51VAE")
    
    # data and output
    parser.add_argument('--data_dir',
                        type=str,
                        default="/local/gongchao/Code/inpainting/data/editbench")
    parser.add_argument('--image_save_path', 
                        type=str, 
                        default="./outputs/BrushNet-OptNoGuidance")
    parser.add_argument('--coco_image_root', type=str, default="/local/gongchao/Dataset/MSCOCO")

    # metric and reward models
    parser.add_argument('--clip_model_name_or_path', 
                        type=str, 
                        default="/local/gongchao/Models/openai-clip-vit-large-patch14")
    parser.add_argument('--image_reward_model_name_or_path',
                        type=str,
                        default='/local/gongchao/Models/ImageReward')
    parser.add_argument('--inpaint_reward_config_path',
                        type=str,
                        default='/local/gongchao/Models/prefpaintReward/configs.yaml')
    parser.add_argument('--inpaint_reward_model_path', type=str, default='/local/gongchao/Models/prefpaintReward/prefpaintReward.pt')

    # generation parameters
    parser.add_argument('--num_images_per_prompt', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--paintingnet_conditioning_scale', type=float,default=1.0)

    # parameters
    parser.add_argument('--reward_guidance_scale', type=float, default=25)
    parser.add_argument('--overall_reward_scale', type=float, default=0.1)
    parser.add_argument('--prompt_reward_scale', type=float, default=4)
    parser.add_argument('--harmonic_reward_scale', type=float, default=1.0)
    parser.add_argument('--self_attn_loss_scale', type=float, default=5.0)

    parser.add_argument('--max_round_initno', type=int, default=5)
    parser.add_argument('--guide_per_steps', type=int, default=1)
    parser.add_argument('--opt_noise_steps', type=int, default=40)

    # dataset flags
    parser.add_argument('--use_prompt_mask', action='store_true', help='use prompt_mask in EditBench')

    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 1
    set_seed(seed)

    exp_suffix = (
        f"-rw_{args.reward_guidance_scale}-ov_{args.overall_reward_scale}"
        f"-pr_{args.prompt_reward_scale}-hm_{args.harmonic_reward_scale}"
        f"-sattn_{args.self_attn_loss_scale}-steps_{args.opt_noise_steps}"
        f"-gps_{args.guide_per_steps}-maxrin_{args.max_round_initno}"
    )
    args.image_save_path = f'{args.image_save_path}{exp_suffix}'
    os.makedirs(os.path.join(args.image_save_path, 'images'), exist_ok=True)

    with open(os.path.join(args.image_save_path, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    overall_reward = ImageRewardScore(args.image_reward_model_name_or_path, device, dtype=torch.float16)
    prompt_reward = PromptRewardScore(args.clip_model_name_or_path, device, dtype=torch.float16)
    harmonic_reward = InpaintReward(args.inpaint_reward_config_path, device, dtype=torch.float16)
    harmonic_reward = harmonic_reward.load_model(harmonic_reward, args.inpaint_reward_model_path)

    base_model_path = args.base_model_path
    brushnet_path = args.brushnet_ckpt_path
    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionBrushNetOptNoGuidancePipeline.from_pretrained(
        base_model_path, 
        brushnet=brushnet, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    )
    for param in pipe.brushnet.parameters():
        param.requires_grad = False
    for param in pipe.unet.parameters():
        param.requires_grad = False
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False

    pipe.max_round_initno = args.max_round_initno
    pipe.self_attn_loss_scale = args.self_attn_loss_scale
    pipe.opt_noise_steps = args.opt_noise_steps
    pipe.guide_per_steps = args.guide_per_steps

    pipe.overall_reward=overall_reward
    pipe.prompt_reward=prompt_reward
    pipe.harmonic_reward=harmonic_reward
    pipe.reward_guidance_scale=args.reward_guidance_scale
    pipe.overall_reward_scale=args.overall_reward_scale
    pipe.prompt_reward_scale=args.prompt_reward_scale
    pipe.harmonic_reward_scale=args.harmonic_reward_scale

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed or when using Torch 2.0.
    pipe.enable_xformers_memory_efficient_attention()
    # memory optimization.
    # pipe.enable_model_cpu_offload()
    pipe.to(device)

    if 'editbench' in args.data_dir.lower():
        test_ds = EditBench(root_folder=args.data_dir, verbose=True, use_prompt_mask=args.use_prompt_mask)
    elif 'coco' in args.data_dir.lower() and 'llava' in args.data_dir.lower():
        test_ds = COCOCustom(
            jsonl_path=args.data_dir,
            image_root=args.coco_image_root,
            verbose=True,
            use_bbox=args.use_bbox
        )
    else:
        raise NotImplementedError("Only EditBench and COCO-Llava dataset are supported in this script.")


    for key, info in list(enumerate(test_ds)):  # [:1000] for coco
        # slice with start_idx and end_idx
        if args.start_idx is not None and key < args.start_idx:
            continue
        if args.end_idx is not None and key >= args.end_idx:
            break

        image_name, init_image, mask_rgb, caption, caption_mask = info
        save_path= os.path.join(args.image_save_path,'images',image_name+".jpg") 
        masked_image_save_path=save_path.replace(".jpg","_masked.jpg")

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if os.path.exists(save_path) and os.path.exists(masked_image_save_path):
            print(f"image {key} exitst! skip...")
            continue

        print(f"generating image {key} ...")

        mask_array = np.array(mask_rgb)[:,:,0]/255.
        init_image = resize(init_image, (512, 512))
        if mask_array.shape != init_image.size:
            mask_array = cv2.resize(mask_array, init_image.size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        mask_array = mask_array[:,:,np.newaxis]
        
        init_image_np=np.array(init_image)
        init_image = init_image * (1-mask_array)
        init_image = Image.fromarray(init_image).convert("RGB")
        mask_image = Image.fromarray(mask_array.repeat(3,-1)*255).convert("RGB")

        generator = torch.Generator(device).manual_seed(seed)

        images = pipe(
            caption if not args.use_prompt_mask else caption_mask,
            init_image, 
            mask_image, 
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            brushnet_conditioning_scale=args.paintingnet_conditioning_scale,
            num_images_per_prompt=args.num_images_per_prompt,
        ).images

        grid_image = get_image_grid(images)
        grid_image.save(save_path)
        init_image.save(masked_image_save_path)

    del pipe, brushnet, overall_reward, prompt_reward, harmonic_reward
    torch.cuda.empty_cache()

    eval_csv_name = "evaluation_result.csv" if args.start_idx is None and args.end_idx is None else f"evaluation_result_{args.start_idx}_{args.end_idx}.csv"
    # only evaluate when the evaluation result not exists
    if not os.path.exists(os.path.join(args.image_save_path, eval_csv_name)):
        print("Start evaluation ...")
        evaluation_df = pd.DataFrame(columns=['Image ID','Image Reward', 'HPS V2.1', 'CLIP Similarity', 'Inpaint Reward', 'local CLIP', 'Global LPIPS', 'prompt', 'prompt_mask'])
        
        metrics_calculator=MetricsCalculator(
            device,
            clip_metric_model_name_or_path=args.clip_model_name_or_path,
            image_reward_model_name_or_path=args.image_reward_model_name_or_path,
            inpaint_reward_config_path=args.inpaint_reward_config_path,
            inpaint_reward_model_path=args.inpaint_reward_model_path
            )

        for key, info in list(enumerate(test_ds)):
            if args.start_idx is not None and key < args.start_idx:
                continue
            if args.end_idx is not None and key >= args.end_idx:
                break

            image_name, init_image, mask_rgb, caption, caption_mask = info
            mask_array = np.array(mask_rgb)[:,:,0]/255.
            init_image = resize(init_image, 512)
            if mask_array.shape != init_image.size:
                mask_array = cv2.resize(mask_array, init_image.size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            prompt=caption
            src_image = init_image

            tgt_image_path = os.path.join(args.image_save_path,'images',image_name+".jpg") 
            tgt_image = Image.open(tgt_image_path).resize(src_image.size)

            evaluation_result=[key]
            mask = 1 - mask_array[:,:,np.newaxis]

            for metric in evaluation_df.columns.values.tolist()[1:]:
                if metric == 'Image Reward':
                    metric_result = metrics_calculator.calculate_image_reward(tgt_image,prompt)
                    
                if metric == 'HPS V2.1':
                    metric_result = metrics_calculator.calculate_hpsv21_score(tgt_image,prompt)
                
                if metric == 'CLIP Similarity':
                    metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)

                if metric == 'Inpaint Reward':
                    metric_result = metrics_calculator.calculate_inpaint_reward(tgt_image, 1-mask)

                if metric == 'local CLIP':
                    metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, caption_mask, 1-mask)

                if metric == 'Global LPIPS':
                    metric_result = metrics_calculator.calculate_lpips(src_image, tgt_image)

                if metric == 'prompt':
                    metric_result = prompt

                if metric == 'prompt_mask':
                    metric_result = caption_mask

                evaluation_result.append(metric_result)
            
            evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

        print("The averaged evaluation result:")
        averaged_results=evaluation_df.mean(numeric_only=True)
        print(averaged_results)

        if args.start_idx is None and args.end_idx is None:
            averaged_results.to_csv(os.path.join(args.image_save_path,"evaluation_result_sum.csv"))
            evaluation_df.to_csv(os.path.join(args.image_save_path,"evaluation_result.csv"))
        else:
            averaged_results.to_csv(os.path.join(args.image_save_path,f"evaluation_result_sum_{args.start_idx}_{args.end_idx}.csv"))
            evaluation_df.to_csv(os.path.join(args.image_save_path,f"evaluation_result_{args.start_idx}_{args.end_idx}.csv"))
        print(f"The generated images and evaluation results is saved in {args.image_save_path}")
    else:
        print("Evaluation result exists, skip evaluation!")