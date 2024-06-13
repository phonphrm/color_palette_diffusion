import sys
import os
import numpy as np
from PIL import Image
from Pylette import extract_colors
sys.path.append("fastLayerDecomposition/") # Add path to the fastLayerDecomposition directory

'''
CODE FROM: https://github.com/CraGL/fastLayerDecomposition
'''

def get_bigger_palette_to_show(palette, c=50):
    ##### palette shape is M*3
    palette2=np.ones((1*c, len(palette)*c, 3))
    for i in range(len(palette)):
        palette2[:,i*c:i*c+c,:]=palette[i,:].reshape((1,1,-1))
    return palette2


def Pylette_color_palette(image_path, palette_size=10, resize=True, sort_mode='luminance', mode="MC"):
    '''
    This function extracts the most common colors from an image using the Pylette library.
    '''
    palette = extract_colors(image=image_path, palette_size=palette_size, resize=resize, sort_mode=sort_mode, mode=mode) # MC mode will be deterministic
    most_common_color = palette[:palette_size]
    
    # Create a numpy to store the most common colors
    final_palette = np.zeros((palette_size, 3), dtype=np.uint8)
    
    for i, c in enumerate(most_common_color):
        final_palette[i] = c.rgb
    
    return final_palette

def get_palette_on_img(img, palette, size=(500,500), final_size=(256, 256), output='np'):
    '''
    For visualization purposes, this function adds the palette to the bottom of the image.
    '''
    assert size[1] % palette.shape[0] == 0, "The palette height must be a divisor of the image height"
    # Get the palette width
    palette_width = size[1] // palette.shape[0]
    
    # Turn palette from (Nx3) to (c, c*N, 3)
    palette = get_bigger_palette_to_show(palette, c=palette_width)
    palette = palette / 255
    
    # Resize the image in numpy format to the given size
    img = img.resize(size)
    img = np.asfarray(img) / 255
    
    # Extend the bottom of the image to fit the palette
    palette_height = palette.shape[0]
    img = np.concatenate((img, np.zeros((palette_height, size[0], 3))), axis=0)
    
    # Add the palette to the image
    img[-palette_height:] = palette
    
    # Resize the image to the final size
    img = Image.fromarray((img*255).round().astype(np.uint8))
    img = img.resize(final_size)
    
    # Turn the image back to numpy format
    if output == 'np':
        img = np.asfarray(img) / 255
    
    return img