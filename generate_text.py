import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from config import ImageCaptioning, ImageCaptioningProcessor, Dataset
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipForConditionalGeneration

def generate_text_from_one_image_blip2(image, model, processor, device="cuda"):
    '''
    Accept the image with PIL format
    '''
    
    input = processor(images=image, return_tensors="pt")
    if device == "cuda":
        input = input.to(device)
    
    with torch.no_grad():
        out = model.generate(**input)
        text = processor.decode(out[0], skip_special_tokens=True)
    
    
    return text

def generate_text_from_one_image_blip(image, model, processor, device="cuda"):
    '''
    Accept the image with PIL format
    '''
    
    input_text = ["a photography of", "a scene of", "a picture of", ""] # List of prefix text
    input_text = np.random.choice(input_text, 1)[0]
    
    input = processor(images=image, text=input_text, return_tensors="pt")
    if device == "cuda":
        input = input.to(device)
    
    with torch.no_grad():
        out = model.generate(**input, max_length=20)
        text = processor.decode(out[0], skip_special_tokens=True)
    
    
    return text

def generate_text_from_directory(image_path, model, processor, device="cuda", model_name="blip2"):
    
    '''
    Accept the image path: e.g. "./data"
    '''
    output_path = image_path + "_text"
    
    if os.path.exists(output_path):
        print("The text already exists. Skip the generation.")
        return
    else:
        os.makedirs(output_path)
    
    # Push model to cuda if available
    if device == "cuda":
        model.to(device)
    
    all_images = os.listdir(image_path)
    pbar = tqdm(all_images, desc="Generating text")
    for image_name in pbar:
        # Pbar add image name
        pbar.set_description("Generating text for %s" % image_name.split(".")[0])
        
        # Load image
        path = os.path.join(image_path, image_name)
        image = Image.open(path).convert('RGB')
        
        # Generate text
        if model_name == "blip":
            text = generate_text_from_one_image_blip(image, model, processor, device)
        elif model_name == "blip2":
            text = generate_text_from_one_image_blip2(image, model, processor, device)

        # Save text in a txt file
        text_file_path = os.path.join(output_path, image_name[:-4] + ".txt")
        with open(text_file_path, "w") as f:
            f.write(text)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="movie_identification_dataset")
    parser.add_argument("--model_name", type=str, required=True, default="blip2")
    parser.add_argument("--device", type=str, required=False, default="cuda")
    arg = parser.parse_args()
    
    # Get dataset name path
    image_path = Dataset[arg.dataset][1]
    
    # Ge the model and processor
    processor_pretrained = ImageCaptioningProcessor[arg.model_name]
    model_pretrained = ImageCaptioning[arg.model_name]
    
    if arg.model_name == "blip":
        processor = AutoProcessor.from_pretrained(processor_pretrained)
        model = BlipForConditionalGeneration.from_pretrained(model_pretrained)
    elif arg.model_name == "blip2":
        processor = AutoProcessor.from_pretrained(processor_pretrained)
        model = Blip2ForConditionalGeneration.from_pretrained(model_pretrained)
    
    print(f"Use {arg.model_name} model to generate text from {arg.dataset} dataset")
    
    # Generate text
    generate_text_from_directory(image_path,
                                 model,
                                 processor,
                                 arg.device,
                                 arg.model_name)
    
    torch.cuda.empty_cache() # Empty the cache