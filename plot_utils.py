import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from extract_palette import get_palette_on_img


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def get_default_prompt_to_plot():
    prompts = [
        "A photography of a cat",
        "A scene of a city",
        "A picture of a human in realist style",
    ]
    
    # Load additional prompt from samples directory
    samples_dir = "samples"
    if os.path.exists(samples_dir):
        sample_files = os.listdir(samples_dir)
        sample_files = sorted(sample_files)
        for sample in sample_files:
            if ".txt" in sample:
                with open(f"{samples_dir}/{sample}", "r") as f:
                    cur_prompt = f.read().strip()
                    prompts.append(cur_prompt)
    else:
        print(f"Samples directory {samples_dir} does not exist, use default prompt instead")
    
    print(f"Used {len(prompts)} prompts")
    
    return prompts


def get_default_palette_to_plot():
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


    # Turn all into tensor and stack
    palettes_ori = [palette_1, palette_2, palette_3, palette_4, palette_5]
    
    # Load additional palettes from samples directory
    samples_dir = "samples"
    if os.path.exists(samples_dir):
        sample_files = os.listdir(samples_dir)
        sample_files = sorted(sample_files)
        for sample in sample_files:
            if ".npy" in sample:
                cur_palette = np.load(f"{samples_dir}/{sample}")
                palettes_ori.append(cur_palette.tolist())
    else:
        print(f"Samples directory {samples_dir} does not exist, use default palettes instead")
    
    palettes = [torch.tensor(palette).float() / 255.0 for palette in palettes_ori]
    palettes = torch.stack(palettes)
    
    n_palette = len(palettes)
    print(f"Used {n_palette} palettes")

    # Resize for 4 tokens
    palettes = palettes.view(n_palette, -1)
    
    return palettes, palettes_ori


def plot_prompt_palette(image_logs, palette, prompt="", row=2, col=2):
    """Plot the images with the same prompt and same color palette

    Args:
        image_logs (_type_): image_logs from model
        palette (_type_): palette with 10 colors and range 0-255
        prompt (str, optional): current prompt to look at. Defaults to "".
        row (int, optional): row for subplot. Defaults to 2.
        col (int, optional): col for subplot. Defaults to 2.
    """
    images = image_logs[prompt]
    output_pil_list = []
    for i, img_list in enumerate(images):
        cur_palette = np.array(palette[i])
        cur_image_grid = image_grid(img_list, row, col)
        img_w_palette = get_palette_on_img(cur_image_grid, cur_palette, size=(1000,1000), final_size=(1000,1000), output="")
        output_pil_list.append(img_w_palette)
    
    return output_pil_list

def compose_all_images(output_pil_list_all, n_palette=5, prompt=None, step=0, cfg=0, dir="inference_images"):
    
    os.makedirs(dir, exist_ok=True)
    
    fig, ax = plt.subplots(len(output_pil_list_all), n_palette, figsize=(30, 30))
    for i, img_list in enumerate(output_pil_list_all):
        for j, img in enumerate(img_list):
            ax[i, j].imshow(img)
            ax[i, j].axis("off")
            
            if j == (n_palette//2) and prompt is not None:
                ax[i, j].set_title(f"Prompt: {prompt[i]}")
    
    plt.tight_layout()
    fig.savefig(f"{dir}/4toks_step={step}_cfg={cfg}.png")


def get_sample_from_dataset(dataset_path, n_sample=3, dest_path="samples"):
    os.makedirs(dest_path, exist_ok=True)
    
    image_path = dataset_path
    palette_path = dataset_path + "_palette"
    prompt_path = dataset_path + "_text"

    assert os.path.exists(image_path), f"Image path {image_path} does not exist"
    assert os.path.exists(palette_path), f"Palette path {palette_path} does not exist"
    assert os.path.exists(prompt_path), f"Prompt path {prompt_path} does not exist"
    
    # Get all images scene
    image_files = os.listdir(image_path)
    
    # Randomly pick n samples
    sample_files = np.random.choice(image_files, n_sample, replace=False)
    
    for sample in sample_files:
        sample_name = sample[:-4]
        palette_name = sample_name + ".npy"
        prompt_name = sample_name + ".txt"
        
        # Copy image, palette, and prompt
        shutil.copyfile(f"{image_path}/{sample}", f"{dest_path}/{sample}")
        shutil.copyfile(f"{palette_path}/{palette_name}", f"{dest_path}/{palette_name}")
        shutil.copyfile(f"{prompt_path}/{prompt_name}", f"{dest_path}/{prompt_name}")
        

if __name__ == "__main__":
    get_sample_from_dataset("../PaletteDiffusion/movie_identification_dataset/movie_identification_dataset_filtered")