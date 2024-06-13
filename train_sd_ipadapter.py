import argparse
import contextlib
import gc
import logging
import math
import os
import random
import itertools
import shutil
from pathlib import Path
from copy import deepcopy

# Torch, Accelerate, Transformers, etc.
import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

# Diffusers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# IP Adapter modules
from ip_adapter import IPAttnProcessor, AttnProcessor, PaletteProjModel, IPAdapterPalette, PaletteProjModelStable

# Palette plot utils
from extract_palette import get_bigger_palette_to_show, get_palette_on_img



# Check if wanb is available
if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Training SD-IP Adapter model for Color Palette")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--used_stable_proj_model",
        type=int,
        default=1,
        help="If use the stable projection model for palette, then the value is 1 ,else 0. Default is 1.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images",
    )
    parser.add_argument(
        "--palette_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of color palette",
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of text",
    )
    parser.add_argument(
        "--palette_sample_dir",
        type=str,
        default=None,
        required=True,
        help="A folder to save palette samples for validation",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_sd_ipadapter",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def log_validation(
    vae, text_encoder, tokenizer, unet, palettes, args, accelerator, weight_dtype, step,
):
    logger.info("Running validation")
    
    # From ip_adapter_t2i-adapter_demo.ipynb
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # Load SD Pipeline
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
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True) # Dont show progress when inferencing
    
    # Load IP Adapter
    if args.pretrained_ip_adapter_path is None:
        # Load the lastest checkpoint
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        last_checkpoint = checkpoints[-1]
        pretrained_ip_adapter_path = os.path.join(args.output_dir, last_checkpoint)
        
        palette_ckpt = os.path.join(pretrained_ip_adapter_path, "palette_proj/model.safetensors")
        adapter_ckpt = os.path.join(pretrained_ip_adapter_path, "adapter/model.safetensors")
    
    if args.used_stable_proj_model == 1:
        n_tokens = 10
        used_stable = True
    else:
        n_tokens = 4
        used_stable = False
    
    # NOTE: In this initailization, the IPAdapterPalette will load the IP Adapter to the Pipe which is our unet and it might be transfer to another dtype
    ip_model = IPAdapterPalette(pipe, adapter_ckpt, palette_ckpt, device=accelerator.device, n_tokens=n_tokens, weight_dtype=weight_dtype, used_stable=used_stable) # Path to safetensor
    
    
    if palettes.size(0) == len(args.validation_prompt):
        palettes = palettes
        validation_prompts = args.validation_prompt
    elif palettes.size(0) == 1:
        palettes = palettes.repeat(len(args.validation_prompt), 1, 1)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        palettes = palettes
        validation_prompts = args.validation_prompt * palettes.size(0)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )
    
    image_logs = []
    inference_ctx = torch.autocast("cuda") # Not final validation, for final validation use nullcontext()
    
    for validation_prompt, palette in zip(validation_prompts, palettes):
        palette_inference = palette.to(accelerator.device)
        
        if args.used_stable_proj_model == 1:
            palette_inference = palette_inference.view(-1, 10, 3)
        else:
            palette_inference = palette_inference.view(1, -1)
        
        # for _ in range(args.num_validation_images):
        with inference_ctx:
            image = ip_model.generate(
                palette=palette_inference,
                prompt=validation_prompt,
                num_samples=1, # One image per prompt and palette
                num_inference_steps=50,
                seed=args.seed,
            )[0] # Index zero because we only have one image
        
        # Get the final image with color palette at the bottom
        palette_ori_scale = (palette.cpu().numpy() * 255).astype(np.uint8)
        palette_img = get_palette_on_img(image, palette_ori_scale, size=(500,500), final_size=(512,512), output='')
        image_logs.append({"image": palette_img, "prompt": validation_prompt})
    
    tracker_key = "validation" # If final validation use "test"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []
            
            for log in image_logs:
                im = log["image"]
                prompt = log["prompt"]
                formatted_images.append(wandb.Image(im, caption=prompt))

            tracker.log({tracker_key: formatted_images})
        else:
            raise ValueError(f"Tracker {tracker.name} is not supported for my validation")

    del pipe, ip_model, unet # Need to remove the model to free up memory
    gc.collect()
    torch.cuda.empty_cache()
    return image_logs
    

class PaletteDataset(Dataset):
    def __init__(self,
                 data_dir,
                 palette_dir,
                 text_dir,
                 tokenizer,
                 size=512,
                 t_drop_rate=0.05,
                 i_drop_rate=0.05,
                 ti_drop_rate=0.05,
                 args=None):
        super().__init__()
        
        self.args = args
        self.data_dir = data_dir
        self.palette_dir = palette_dir
        self.text_dir = text_dir
        
        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.data_name = os.listdir(data_dir)
        self.data_name = [name[:-4] for name in self.data_name]
        
        # Check if every datapoints really exists
        for name in self.data_name:
            assert os.path.exists(os.path.join(data_dir, name + ".jpg"))
            assert os.path.exists(os.path.join(palette_dir, name + ".npy"))
            assert os.path.exists(os.path.join(text_dir, name + ".txt"))
    
    def __len__(self):
        return len(self.data_name)
    
    def __getitem__(self, idx):
        data_name = self.data_name[idx]
        image_file = os.path.join(self.data_dir, data_name + ".jpg")
        palette_file = os.path.join(self.palette_dir, data_name + ".npy")
        text_file = os.path.join(self.text_dir, data_name + ".txt")
        
        # Load image, Palette, and Text
        image = Image.open(image_file).convert("RGB")
        palette = np.load(palette_file)
        text = open(text_file, "r").read()
        
        # Randomly drop the text and palette (CFG)
        drop_palette = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            # Drop palette
            drop_palette = 1
        elif rand_num < self.i_drop_rate + self.t_drop_rate:
            # Drop text
            text = ""
        elif rand_num < self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate:
            # Drop both text and palette
            drop_palette = 1
            text = ""

        # Tokenize text
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        
        # Transform image and normalize palette
        image = self.transform(image)
        palette = torch.tensor(palette).float() / 255.0

        if self.args.used_stable_proj_model == 1:
            palette = palette.view(1, 10, 3)
        else:
            palette = palette.view(1, -1)
        
        return {
            "pixel_values": image,
            "text_input_ids": text_input_ids,
            "palette": palette,
            "drop_palette": drop_palette,
        }


def collate_fn(batch):
    images = torch.stack([example["pixel_values"] for example in batch])
    text_input_ids = torch.cat([example["text_input_ids"] for example in batch], dim=0)
    palette = torch.cat([example["palette"] for example in batch], dim=0)
    drop_palette = [example["drop_palette"] for example in batch]
    
    return {
        "pixel_values": images,
        "text_input_ids": text_input_ids,
        "palette": palette,
        "drop_palette": drop_palette,
    }


def generate_palette_from_num_validation_images(args):
    # Default palette (User-specified) for 5 first images
    assert args.num_validation_images <= 5, "Only 5 palettes are available"
    
    # 1: https://coolors.co/palette/590d22-800f2f-a4133c-c9184a-ff4d6d-ff758f-ff8fa3-ffb3c1-ffccd5-fff0f3
    palette_1 = [
        [89, 13, 34],
        [128, 15, 47],
        [164, 19, 60],
        [201, 24, 74],
        [255, 77, 109],
        [255, 117, 143],
        [255, 143, 163],
        [255, 179, 193],
        [255, 204, 213],
        [255, 240, 243],
    ]
    
    # 2: https://coolors.co/palette/03071e-370617-6a040f-9d0208-d00000-dc2f02-e85d04-f48c06-faa307-ffba08
    palette_2 = [
        [3, 7, 30],
        [55, 6, 23],
        [106, 4, 15],
        [157, 2, 8],
        [208, 0, 0],
        [220, 47, 2],
        [232, 93, 4],
        [244, 140, 6],
        [250, 163, 7],
        [255, 186, 8],
    ]
    
    # 3: https://coolors.co/palette/d9ed92-b5e48c-99d98c-76c893-52b69a-34a0a4-168aad-1a759f-1e6091-184e77
    palette_3 = [
        [217, 237, 146],
        [181, 228, 140],
        [153, 217, 140],
        [118, 200, 147],
        [82, 182, 154],
        [52, 160, 164],
        [22, 138, 173],
        [26, 117, 159],
        [30, 96, 145],
        [24, 78, 119],
    ]
    
    # 4: https://coolors.co/palette/001219-005f73-0a9396-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-ae2012-9b2226
    palette_4 = [
        [0, 18, 25],
        [0, 95, 115],
        [10, 147, 150],
        [148, 210, 189],
        [233, 216, 166],
        [238, 155, 0],
        [202, 103, 2],
        [187, 62, 3],
        [174, 32, 18],
        [155, 34, 38],
    ]
    
    # 5: https://coolors.co/palette/ff0000-ff8700-ffd300-deff0a-a1ff0a-0aff99-0aefff-147df5-580aff-be0affRandom color palette
    palette_5 = [
        [255, 0, 0],
        [255, 135, 0],
        [255, 211, 0],
        [222, 255, 10],
        [161, 255, 10],
        [10, 255, 153],
        [10, 239, 255],
        [20, 125, 245],
        [88, 10, 255],
        [190, 10, 255],
    ]
    
    # Plot the color palette and save to the palette_sample_dir
    os.makedirs(args.palette_sample_dir, exist_ok=True)
    for i, palette in enumerate([palette_1, palette_2, palette_3, palette_4, palette_5]):
        np_palette = np.array(palette)
        palette_img = get_bigger_palette_to_show(np_palette, c=50)
        palette_img = Image.fromarray(palette_img.astype(np.uint8))
        palette_img.save(os.path.join(args.palette_sample_dir, f"palette_{i+1}.png"))
    
    # Turn all into tensor and stack
    palettes = [palette_1, palette_2, palette_3, palette_4, palette_5]
    palettes = [torch.tensor(palette).float() / 255.0 for palette in palettes]
    palettes = torch.stack(palettes)
    
    return palettes[:args.num_validation_images]


class IPAdapter(nn.Module):
    """IP-Adapter"""
    
    def __init__(self, unet, palette_proj, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.palette_proj = palette_proj
        self.adapter_modules = adapter_modules
  
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
    
    def load_from_checkpoint(self, ckpt_path):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.palette_proj.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.palette_proj.load_state_dict(state_dict["palette_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.palette_proj.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    
    def forward(self, noisy_latents, timesteps, encoder_hidden_states, palette):
        # Get the palette embeddings and concatenate with encoder_hidden_states(text)
        # NOTE: This will be passed to IPAttnProcessor (which both will be added)
        p_tokens = self.palette_proj(palette)
        encoder_hidden_states = torch.cat([encoder_hidden_states, p_tokens], dim=1)
        
        # Predict the noise
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred
        

def main():
    args = parse_args()
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot the color palette and save to the palette_sample_dir
    palette_for_validation = generate_palette_from_num_validation_images(args)
    
    # Load the scheduler, tokenizer, and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Freeze parameters of the models
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    
    # Initialize Palette Projection Model
    if args.used_stable_proj_model == 1:
        logger.info("Using 10-tokens Projection Model for Palette")
        palette_proj = PaletteProjModelStable(
            palette_dim=3,
            cross_attention_dim=unet.config.cross_attention_dim,
            n_tokens=10,
        )
        n_tokens = 10
    else:
        logger.info("Using 4-tokens Projection Model for Palette")
        # Use the same number tokens as in IPAdapter paper
        palette_proj = PaletteProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            n_tokens=4, # Default number of tokens like in the paper
        )
        n_tokens = 4
    
    # Initialize Adapter modules in Stable Diffusion
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor() # Self-Attention
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=n_tokens)
            attn_procs[name].load_state_dict(weights) # Initialize weights from UNet
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    # Initialize IP Adapter
    ip_adapter = IPAdapter(unet, palette_proj, adapter_modules, args.pretrained_ip_adapter_path)
    
    # Optimizer
    params_to_opt = itertools.chain(ip_adapter.palette_proj.parameters(), ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Make Dataset and DataLoader
    train_dataset = PaletteDataset(
        data_dir=args.data_dir,
        palette_dir=args.palette_dir,
        text_dir=args.text_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        args=args,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(
        ip_adapter, optimizer, train_dataloader
    )
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # NOTE: Trainable model should be in full precision, therefore we will not push ip_adapter to fp16
    # unet.to(accelerator.device, dtype=weight_dtype) # This is like in the original repo, they comment this
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get palette ready
                palette_ = []
                for palette_i, drop_palette in zip(batch["palette"], batch["drop_palette"]):
                    if drop_palette == 1:
                        palette_.append(torch.zeros_like(palette_i))
                    else:
                        palette_.append(palette_i)
                palette = torch.stack(palette_).to(accelerator.device)
                
                with torch.no_grad():
                    # Encode text token
                    text_input_ids = batch["text_input_ids"].to(accelerator.device)
                    encoder_hidden_states = text_encoder(text_input_ids)[0]
                
                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, palette)
                
                # Loss computation
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # Update the model one step
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    # Save the checkpoint with first step and every `checkpointing_steps`
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        
                        unwrap = accelerator.unwrap_model(deepcopy(ip_adapter))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_path_adapter = os.path.join(save_path, "adapter")
                        save_path_palette_proj = os.path.join(save_path, "palette_proj")
                        accelerator.save_state(save_path)
                        
                        # Remove model.safetensors(whole model)
                        os.remove(os.path.join(save_path, "model.safetensors"))
                        
                        accelerator.save_model(unwrap.adapter_modules, save_path_adapter)
                        accelerator.save_model(unwrap.palette_proj, save_path_palette_proj)
                        logger.info(f"Saved state to {save_path}")
                        
                        del unwrap
                    
                    # Plotting the results
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            palette_for_validation,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )
            
            logs = {"loss": loss.detach().item(), "lr": optimizer.param_groups[0]["lr"]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
                
    
if __name__ == "__main__":
    main()