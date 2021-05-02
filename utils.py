import string
import matplotlib.pyplot as plt
import numpy as np


def visualize_attention(orig_image, words, atten_weights):
    """Plots attention in the image sample.
    
    Arguments:
    ----------
    - orig_image - image of original size
    - words - list of tokens
    - atten_weights - list of attention weights at each time step 
    """
    fig = plt.figure(figsize=(14,12)) 
    len_tokens = len(words)
    
    for i in range(len(words)):
        atten_current = atten_weights[i].detach().numpy()
        atten_current = atten_current.reshape(7,7)       
        ax = fig.add_subplot(len_tokens//2, len_tokens//2, i+1)
        ax.set_title(words[i])
        img = ax.imshow(np.squeeze(orig_image))
        ax.imshow(atten_current, cmap='gray', alpha=0.8, extent=img.get_extent(), interpolation = 'bicubic')
    plt.tight_layout()
    plt.show()
