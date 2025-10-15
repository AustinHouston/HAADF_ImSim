import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom

path = '/Users/austin/Desktop/Projects/BCC_Iron_Yao/completed_sims/'

files = os.listdir(path)
files = [f for f in files if '.npz' in f]
files.sort()

for file_name in files:
    ### --- Load --- ###
    recip_zoom_factor = 2
    data = np.load(path + file_name)['arr_0']
    data = zoom(data, (1, 1, recip_zoom_factor, recip_zoom_factor), order=3)
    data.shape
    max_diff = np.max(data, axis=(0,1))
    avg_im = np.mean(data, axis=(2,3))
    max_diff -= max_diff.min()
    max_diff /= max_diff.max()

    plt.figure()
    plt.imshow(max_diff, cmap='gray', vmax=0.3)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(f'sim_ims/{file_name}_max_diff.png', dpi=300, bbox_inches='tight')
    plt.close()

    ### --- Virtual Detectors --- ###
    g = 8.75 * recip_zoom_factor * 2 # pixels
    center = [data.shape[-1]//2, data.shape[-2]//2]

    if '0' in file_name.split('_')[2]:
        g_vecs = np.array([[0,0],[0,2*g],[0,-2*g], [-2*g,0], [2*g,0]])
    elif '0' in file_name.split('_')[3]:
        g_vecs = np.array([[0,0],[0,2*g],[0,-2*g], [-2*g,0], [2*g,0]])
    else:
        g_vecs = np.array([[0,0],[2*g,2*g],[2*g,-2*g], [-2*g, 2*g], [-2*g,-2*g]])

    g_vecs += center
    detector_radius = 10 * recip_zoom_factor
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    plt.figure()
    plt.imshow(max_diff, cmap='gray', vmax=0.5)
    plt.axis('off')
    plt.colorbar()
    for color, (cx, cy) in zip(colors, g_vecs):
        circle = plt.Circle((cx, cy), detector_radius, color=color, fill=False, lw=1.5)
        plt.gca().add_artist(circle)
    plt.savefig(f'sim_ims/{file_name}_virtual_detectors.png', dpi=300, bbox_inches='tight')
    plt.close()

    diff_shape = data.shape[-2:]  # diffraction plane shape
    Y, X = np.ogrid[:diff_shape[0], :diff_shape[1]]

    masks = {}
    for color, (cx, cy) in zip(colors, g_vecs):
        dist_sq = (X - cx)**2 + (Y - cy)**2
        masks[color] = dist_sq <= detector_radius**2  # Boolean mask

    ### --- Generate virtual images --- ###
    virtual_images = {}
    for color, mask in masks.items():
        # Apply mask over diffraction dimensions (last two dims of data)
        masked_data = data[..., mask]  
        virtual_image = masked_data.sum(axis=-1)  # Sum over masked diffraction pixels
        virtual_images[color] = virtual_image

    edge_cutoff = 3
    for key in virtual_images:
        img = virtual_images[key]
        virtual_images[key] = img[edge_cutoff:-edge_cutoff, edge_cutoff:-edge_cutoff]

    fig, axes = plt.subplots(1, len(virtual_images), figsize=(15, 3), dpi=300)
    images = []
    for ax, (color, img) in zip(axes, virtual_images.items()):
        img = zoom(img, zoom=3, order=2)
        images.append(img)
        img -= img.min()
        
        ax.imshow(img, cmap='gray', vmin=0, vmax=0.8*img.max())
        ax.set_title(f"{color} detector")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'sim_ims/{file_name}_virtual_images.png', dpi=300, bbox_inches='tight')
    plt.close()

    images = np.array(images)
    np.savez(f'sim_ims/{file_name}_virtual_images.npz', images)





