import os

# Filtering the dataset
os.system("python data_filtering.py \
    --dataset movie_identification_dataset")

# Generate palette 
os.system("python generate_palette.py \
    --dataset movie_identification_dataset")

# # Generate text
os.system("python generate_text.py \
    --dataset movie_identification_dataset \
    --model_name blip \
    --device cuda")