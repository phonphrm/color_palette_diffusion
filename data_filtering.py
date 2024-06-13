import os
import shutil
import argparse
from config import Dataset
from tqdm import tqdm

def data_filtering_movie_identification_dataset(data_path, output_path='movie_identification_dataset_filtered'):
    '''
    Dataset URL: https://www.kaggle.com/datasets/asaniczka/movie-identification-dataset-800-movies/data
    The folder structure is as follows:
    
    - resized_frames
        - movie01
            - frame_0000.jpg
            - frame_0001.jpg
            - ...
            - frame_0999.jpg
        - movie02
            - frame_0000.jpg
            - frame_0001.jpg
            - ...
            - frame_0999.jpg
        - ...
    
    '''
    
    # Create the output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print("The output folder already exists. Skip the filtering.")
        return
    
    # Get all movie folders
    all_movies = os.listdir(data_path)
    all_movies.sort() # Sort the movies
    
    for movie in tqdm(all_movies):
        movie_path = os.path.join(data_path, movie)
        all_frames = os.listdir(movie_path)
        all_frames.sort() # Sort the frames
        filtered_frames = all_frames[300:450] # 150 frames
        
        # Save all filtered frames in the same output folder with following name structure: e.g. movie01_frame_0300.jpg
        for frame in filtered_frames:
            frame_number = frame.split("_")[-1].split(".")[0]
            frame_path = os.path.join(movie_path, frame)
            output_frame_path = os.path.join(output_path, f"{movie}_{frame_number}.jpg")
            shutil.copy(frame_path, output_frame_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="movie_identification_dataset")
    
    data_path, output_path = Dataset[parser.parse_args().dataset]
    data_filtering_movie_identification_dataset(data_path, output_path)
        