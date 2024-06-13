import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from extract_palette import Pylette_color_palette
from config import Dataset

def generate_palette_from_one_image(image_path, n_palette=10):
    '''
    Accept the image with PIL format
    '''
    
    palette = Pylette_color_palette(image_path, 
                                    palette_size=n_palette, 
                                    resize=False,  # Result will make it faster but with lower quality
                                    sort_mode='luminance')
    
    return palette

def generate_palette_from_directory(image_path, n_palette=10):
    '''
    Accept the image path: e.g. "./data"
    '''
    output_path = image_path + "_palette"
    
    if os.path.exists(output_path):
        print("The palette already exists. Skip the generation.")
        return 
    else:
        os.makedirs(output_path)
    
    all_images = os.listdir(image_path)
    pbar = tqdm(all_images, desc="Generating palettes")
    for image_name in pbar:
        
        # Pbar add image name
        pbar.set_description("Generating palette for %s" % image_name.split(".")[0])
        
        # Load image
        path = os.path.join(image_path, image_name)
        
        # Try to generate palette if possible, if not then delete the image and move on (the convex hull in this code can't work with grayscale images)
        try:
            palette = generate_palette_from_one_image(path, n_palette)
        except:
            os.remove(path)
            continue
        
        # Check if size of palette is equal to n_palette (some palette may have less than n_palette colors)
        if palette.shape[0] != n_palette:
            print(palette.shape[0], "colors found in the palette. Skip the image.")
            os.remove(path)
            continue
        
        # Save palette
        palette_path = os.path.join(output_path, image_name[:-4] + ".npy")
        np.save(palette_path, palette)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="movie_identification_dataset")
    parser.add_argument("--n_palette", type=int, required=False, default=10)
    
    args = parser.parse_args()
    image_path = Dataset[args.dataset][1]
    generate_palette_from_directory(image_path, args.n_palette)