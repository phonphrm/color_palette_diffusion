import os

model_name = "runwayml/stable-diffusion-v1-5"
data_dir = "movie_identification_dataset/movie_identification_dataset_filtered"
palette_dir = data_dir + "_palette"
text_dir = data_dir + "_text"

max_train_steps = 50_000
gradient_accumulation_steps = 4
num_validation_images = 5
validation_steps = 1000
checkpointing_steps = 1000
checkpoints_total_limit = 50
seed = 42
train_batch_size = 4

output_dir = "sd-ip_adapter"
report_to = "wandb"
palette_sample_dir = "palette_samples_for_validation"

os.system(f"accelerate launch --num_processes 4 --multi_gpu train_sd_ipadapter.py \
        --pretrained_model_name_or_path={model_name} \
        --used_stable_proj_model=0\
        --data_dir={data_dir} \
        --palette_dir={palette_dir} \
        --text_dir={text_dir} \
        --validation_prompt 'A scene of high-quality close-up dslr photo of man wearing a hat with trees in the background, realistic, 4k, ultra'\
        --num_validation_images={num_validation_images} \
        --validation_steps={validation_steps} \
        --output_dir={output_dir} \
        --report_to={report_to} \
        --gradient_accumulation_steps={gradient_accumulation_steps} \
        --checkpointing_steps={checkpointing_steps} \
        --palette_sample_dir={palette_sample_dir} \
        --seed={seed} \
        --max_train_steps={max_train_steps} \
        --train_batch_size={train_batch_size}")