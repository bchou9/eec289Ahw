import numpy as np
from scipy.ndimage import binary_dilation
import random
import matplotlib.pyplot as plt
from PIL import Image
import os


def generate_gaussian_window(window_size):
    x = np.arange(window_size) - window_size // 2
    y = np.arange(window_size) - window_size // 2
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-0.5 * (X**2 + Y**2) / (window_size/6.4)**2)
    Z = Z / np.sum(Z)
    return np.stack((Z, Z, Z), axis=-1)

def get_pixel_list(mask):
    struct = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(mask == 1, structure=struct)
    edge_mask = dilated_mask & (mask == 0)
    valid_pixel_locations = np.where(edge_mask)
    valid_pixels = list(zip(valid_pixel_locations[0], valid_pixel_locations[1]))
    random.shuffle(valid_pixels)
    valid_pixels = sorted(valid_pixels, key=lambda x: np.sum(mask[x[0]-1:x[0]+2, x[1]-1:x[1]+2]), reverse=True)
    return valid_pixels

def get_neighborhood_window(image, pixel, window_size):
    x, y = pixel
    half_size = window_size // 2
    return image[x-half_size:x+half_size+1, y-half_size:y+half_size+1]

def get_neighborhood_window_mask(mask, pixel, window_size):
    x, y = pixel
    half_size = window_size // 2
    return mask[x-half_size:x+half_size+1, y-half_size:y+half_size+1]

def get_all_square_gen(sample, window_size):
    h, w, _ = sample.shape
    half_window_size = window_size // 2
    coords = [(i, j) for i in range(half_window_size, h-half_window_size) for j in range(half_window_size, w-half_window_size)]
    random.shuffle(coords)
    for i, j in coords:
        yield sample[i-half_window_size:i+half_window_size+1, j-half_window_size:j+half_window_size+1], (i, j)

def find_matches(template, template_mask, sample, max_err_threshold):
    window_size = template.shape[0]
    gauss_mask = generate_gaussian_window(window_size)
    template_mask = np.expand_dims(template_mask, axis=-1)
    template_mask = np.repeat(template_mask, 3, axis=2)
    tot_weight = np.sum(gauss_mask * template_mask)
    distances = []
    patches = get_all_square_gen(sample, window_size)
    for patch, (i, j) in patches:
        dist = (template - patch) ** 2
        ssd = np.sum(dist * template_mask * gauss_mask, axis=(0, 1, 2)) / tot_weight
        distances.append((ssd, (i, j)))
    min_ssd = min(distances, key=lambda x: x[0])[0]
    return [x for x in distances if x[0] <= min_ssd * (1 + max_err_threshold)]

def grow_image(sample_image, image, mask, window_size, max_err_threshold, output_dir='output10_16'):
    half_window_size = window_size // 2
    fillable_region = mask[half_window_size:-half_window_size, half_window_size:-half_window_size]
    filled_pixels = 0
    while not np.all(fillable_region):
        progress = False
        pixel_list = get_pixel_list(mask)
        for pixel in pixel_list:
            if (
                pixel[0] < half_window_size
                or pixel[0] >= mask.shape[0] - half_window_size
                or pixel[1] < half_window_size
                or pixel[1] >= mask.shape[1] - half_window_size
            ):
                continue
            if np.all(mask[pixel]):
                continue
            template = get_neighborhood_window(image, pixel, window_size)
            template_mask = get_neighborhood_window_mask(mask, pixel, window_size)
            matches = find_matches(template, template_mask, sample_image, max_err_threshold)
            if matches:
                best_match = random.choice(matches)
                new_pixel_val = sample_image[best_match[1]]
                x, y = pixel
                image[x, y] = new_pixel_val
                mask[x, y] = 1
                progress = True
                if filled_pixels % 500 == 0:
                    plt.imsave(f'{output_dir}/snapshot_c_{filled_pixels}.png', image)
                    print(f'Filled pixel: ({x}, {y}) with value: {new_pixel_val}')
                filled_pixels += 1
        if not progress:
            max_err_threshold *= 1.1
    return image

seed_image_path = '6_small.png'
seed_image = np.array(Image.open(seed_image_path).convert('RGB')) / 255.0

border_size = 16*10
border_size_h = border_size // 2
output_image_size = (seed_image.shape[0] + border_size, seed_image.shape[1] + border_size, 3)
output_image = np.zeros(output_image_size)
mask = np.zeros(output_image_size[:2])

start_row = border_size_h
end_row = start_row + seed_image.shape[0]
start_col = border_size_h
end_col = start_col + seed_image.shape[1]

output_image[start_row:end_row, start_col:end_col] = seed_image
mask[start_row:end_row, start_col:end_col] = 1

max_err_threshold = 0.075
window_size = 16

image_name = os.path.splitext(seed_image_path)[0]
# Create the output directory
output_dir = f"{image_name}_{window_size}"
os.makedirs(output_dir, exist_ok=True)

synthesized_image = grow_image(seed_image, output_image, mask, window_size, max_err_threshold, output_dir)

plt.imshow(synthesized_image)
synthesized_image = (synthesized_image * 255).astype(np.uint8)
Image.fromarray(synthesized_image).save(f"synthesized_{image_name}_{window_size}.png")
plt.savefig(f"synthesized_{image_name}_{window_size}_plot.png")