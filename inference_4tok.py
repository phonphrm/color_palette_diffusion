import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict


from extract_palette import get_palette_on_img
from plot_utils import get_default_palette_to_plot, get_default_prompt_to_plot, plot_prompt_palette, compose_all_images
from ip_adapter import IPAdapterPalette
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)

def main(args):
    # Model used in training
    model_name = "runwayml/stable-diffusion-v1-5"
    palette_checkpoint_path = "sd-ip_adapter_4_tokens"
    weight_dtype = torch.float32
    
    # Load the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Compose the entire pipeline
    pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(weight_dtype)

    
    # Get IP Adapter checkpoint
    checkpoints = os.listdir(palette_checkpoint_path)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    checkpoints_step = [int(d.split("-")[1]) for d in checkpoints if d.startswith("checkpoint")]
    checkpoints_step = sorted(checkpoints_step)

    # load specfic checkpoint
    checkpoint_step_to_use = args.checkpoint_step
    idx_checkpoint = checkpoints_step.index(checkpoint_step_to_use)
    checkpoint_to_use = checkpoints[idx_checkpoint]
    checkpoint_specific = os.path.join(palette_checkpoint_path, checkpoint_to_use)
    assert os.path.exists(checkpoint_specific), f"Checkpoint {checkpoint_specific} does not exist"

    palette_ckpt = os.path.join(checkpoint_specific, "palette_proj/model.safetensors")
    adapter_ckpt = os.path.join(checkpoint_specific, "adapter/model.safetensors")
    assert os.path.exists(palette_ckpt), f"Palette checkpoint {palette_ckpt} does not exist"
    assert os.path.exists(adapter_ckpt), f"Adapter checkpoint {adapter_ckpt} does not exist"

    # Load the IP Adapter
    ip_model = IPAdapterPalette(
        sd_pipe=pipe,
        ip_ckpt=adapter_ckpt,
        pl_ckpt=palette_ckpt,
        device="cuda",
        n_tokens=4,
        weight_dtype=weight_dtype,
        used_stable=False,
        is_set_adapter=True, # For setting the adapter to the unet of the pipeline
    )
    
    
    # Prompt and palette to evalulate
    prompts = get_default_prompt_to_plot()
    palettes, palettes_ori = get_default_palette_to_plot()
    
    # Config for inference
    image_logs = defaultdict(list)
    guidance_scale = args.cfg
    img_size = 512 # By default
    num_samples = 4 # By default
    num_inference_steps = 50
    seed = args.seed if args.seed else 42

    # Plotting config
    row = 2
    col = num_samples // 2
    
    for prompt in prompts:
        for i, palette in enumerate(palettes):
            palette = palette.to(weight_dtype)
            palette = palette.unsqueeze(0)
            
            image = ip_model.generate(
                palette=palette,
                prompt=prompt,
                num_samples=num_samples,
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=guidance_scale,
            )
            image_logs[f"{prompt}"].append(image)
            print(f"Prompt: {prompt}, Palette: {i+1} done")
    
    # Plot the images
    output_pil_list = []
    for i in range(len(prompts)):
        cur_pil_list = plot_prompt_palette(image_logs,
                                           palettes_ori,
                                           prompt=prompts[i],
                                           row=row,
                                           col=col)
        output_pil_list.append(cur_pil_list)
    
    # Compose all images
    compose_all_images(
        output_pil_list,
        n_palette=len(palettes_ori),
        prompt=prompts,
        step=checkpoint_step_to_use,
        cfg=guidance_scale,
        dir="inference_images",
    )
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_step", type=int, required=True, default=50000, help="Checkpoint step to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed for inference")
    parser.add_argument("--cfg", type=float, default=7.5, help="Guidance scale for inference")
    args = parser.parse_args()
    main(args)