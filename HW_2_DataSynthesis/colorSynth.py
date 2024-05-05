import numpy as np
from scipy.ndimage import binary_dilation
import random
import matplotlib.pyplot as plt

def generate_gaussian_window(window_size):
    x = np.arange(window_size) - window_size // 2
    y = np.arange(window_size) - window_size // 2
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-0.5 * (X**2 + Y**2) / (window_size/6.4)**2)
    Z = Z / np.sum(Z)
    return np.stack((Z, Z, Z), axis=-1)  # Repeat the Gaussian window for all channels

def getUnfilledNeighbors(image):
    neighbors = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.any(image[i, j] == 0):  # Check if any channel is unfilled
                neighbors.append((i, j))
    return neighbors


def get_pixel_list(image, mask):
    struct = np.ones((3, 3), dtype=bool)
    dilated_image = binary_dilation(np.any(image == 0, axis=-1), structure=struct)
    dilated_mask = binary_dilation(mask, structure=struct)
    valid_pixel_locations = np.where(np.any(image == 0, axis=-1) & dilated_mask)
    valid_pixels = list(zip(valid_pixel_locations[0], valid_pixel_locations[1]))
    valid_pixels = sorted(valid_pixels, key=lambda x: np.sum(dilated_mask[x[0] - 1:x[0] + 2, x[1] - 1:x[1] + 2]), reverse=True)
    return valid_pixels
def GetNeighborhoodWindow(image, pixel, window_size):
    x, y = pixel
    half_size = window_size // 2
    return image[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1]

def getNeighborhoodWindowMask(mask, pixel, window_size):
    x, y = pixel
    half_size = window_size // 2
    return mask[x - half_size:x + half_size + 1, y - half_size:y + half_size + 1]

def get_all_square_gen(Sample, window_size):
    h, w, _ = Sample.shape
    for i in range(0, h - window_size + 1, 6):
        for j in range(0, w - window_size + 1, 6):
            yield Sample[i:i+window_size, j:j+window_size], (i, j)

def findMatches(Template, Template_mask, Sample, MaxErrThreshold):
    window_size = Template.shape[0]
    gaussMask = generate_gaussian_window(window_size)
    Template_mask = np.expand_dims(Template_mask, axis=-1)
    Template_mask = np.repeat(Template_mask, 3, axis=2)

    TotWeight = np.sum(gaussMask * Template_mask)
    distances = []
    patches = get_all_square_gen(Sample, window_size)
    for patch, (i, j) in patches:
        dist = (Template - patch) ** 2
        SSD = np.sum(dist * Template_mask * gaussMask, axis=(0, 1, 2)) / TotWeight
        distances.append((SSD, (i, j)))
    min_SSD = min(distances, key=lambda x: x[0])[0]
    return [x for x in distances if x[0] <= min_SSD * (1 + MaxErrThreshold)]

def GrowImage(SampleImage, Image, Mask, WindowSize, MaxErrThreshold, output_dir='output2'):
    half_window_size = (WindowSize // 2)
    fillable_region = Mask[half_window_size:-half_window_size, half_window_size:-half_window_size]
    filled_pixels = 0
    while not np.all(fillable_region):
        progress = False
        PixelList = get_pixel_list(Image, Mask)
        for pixel in PixelList:
            if (
                    pixel[0] < half_window_size
                    or pixel[0] >= Mask.shape[0] - half_window_size
                    or pixel[1] < half_window_size
                    or pixel[1] >= Mask.shape[1] - half_window_size
            ):
                continue
            if np.all(Mask[pixel]):  # Skip already filled pixels
                continue
            Template = GetNeighborhoodWindow(Image, pixel, WindowSize)
            Template_mask = getNeighborhoodWindowMask(Mask, pixel, WindowSize)
            matches = findMatches(Template, Template_mask, SampleImage, MaxErrThreshold)
            if matches:
                best_match = random.choice(matches)
                new_pixel_val = SampleImage[best_match[1]]
                x, y = pixel
                Image[x, y] = new_pixel_val
                Mask[x, y] = 1
                progress = True
                filled_pixels += 1

                # Save an image snapshot every 10 filled pixels
                if filled_pixels % 1000 == 0:
                    plt.imsave(f'{output_dir}/snapshot_{filled_pixels}.png', Image)

        if not progress:
            MaxErrThreshold *= 1.1
    return Image

# Example usage
from PIL import Image
seed_image_path = '7.png'
seed_image = np.array(Image.open(seed_image_path).convert('RGB')) / 255.0  # Convert to RGB and normalize to 0-1 range
WindowSize = 35

# Initialize the mask and output image
border_size = 200
border_size_h = border_size // 2
mask = np.zeros((seed_image.shape[0] + border_size, seed_image.shape[1] + border_size))
mask[border_size_h:-border_size_h, border_size_h:-border_size_h] = 1  # Set the center region where the seed image will grow

# Initialize output_image with three channels, filled with zeros
output_image = np.zeros((seed_image.shape[0] + border_size, seed_image.shape[1] + border_size, 3))

# Place the seed image in the center of the output image
start_row = border_size_h
start_col = border_size_h
output_image[start_row:start_row + seed_image.shape[0], start_col:start_col + seed_image.shape[1]] = seed_image

MaxErrThreshold = 0.1

synthesized_image = GrowImage(seed_image, output_image, mask, WindowSize, MaxErrThreshold)
plt.imshow(synthesized_image)
#convert from 0-1 to 0-255
synthesized_image = (synthesized_image * 255).astype(np.uint8)
#save iamge not using plt
from PIL import Image
Image.fromarray(synthesized_image).save('synthesized_imageb.png')
plt.savefig('synthesized_imagea.png')

plt.imshow(synthesized_image)
plt.savefig('synthesized_image_rgb.png')
plt.imshow(mask, cmap='gray')
plt.savefig('mask_rgb.png')