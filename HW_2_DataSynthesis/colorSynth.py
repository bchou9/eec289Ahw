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


def get_pixel_list( mask):
    struct = np.ones((3, 3), dtype=bool)
    # Use the mask directly to identify filled and unfilled areas:
    # Filled pixels in the mask are represented by a value of 1

    # Dilate the filled areas in the mask to identify potential growth boundaries
    dilated_mask = binary_dilation(mask == 1, structure=struct)

    # Edge mask: identifies the unfilled pixels adjacent to filled pixels
    # This is computed by finding where the dilated mask is true but the original mask is not
    edge_mask = dilated_mask & (mask == 0)

    # Get coordinates of potential pixels to fill
    valid_pixel_locations = np.where(edge_mask)
    valid_pixels = list(zip(valid_pixel_locations[0], valid_pixel_locations[1]))

    # Optionally, sort these pixels by the number of filled neighbors to prioritize them
    # This prioritizes pixels surrounded by more filled pixels, potentially for more stability in patterns
    valid_pixels = sorted(valid_pixels, key=lambda x: np.sum(mask[x[0] - 1:x[0] + 2, x[1] - 1:x[1] + 2]), reverse=True)

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
    half_window_size = window_size // 2

    # Find all the possible x, y coordinates for the center of the window
    coords = [(i, j) for i in range(half_window_size, h - half_window_size) for j in
              range(half_window_size, w - half_window_size)]

    # Randomly permute the list
    random.shuffle(coords)

    # Yield the first 8000 samples
    for i, j in coords:
        yield Sample[i - half_window_size:i + half_window_size + 1, j - half_window_size:j + half_window_size + 1], (
        i, j)
    # for i in range(0, h - window_size + 1,6):
    #     for j in range(0, w - window_size + 1,6):
    #         yield Sample[i:i+window_size, j:j+window_size], (i, j)

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
    return [x for x in distances if x[0] <= min_SSD]

def GrowImage(SampleImage, Image, Mask, WindowSize, MaxErrThreshold, output_dir='output6'):
    half_window_size = (WindowSize // 2)
    fillable_region = Mask[half_window_size:-half_window_size, half_window_size:-half_window_size]
    filled_pixels = 0
    while not np.all(fillable_region):
        progress = False
        PixelList = get_pixel_list(Mask)
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
            print(matches)
            if matches:
                best_match = random.choice(matches)
                new_pixel_val = SampleImage[best_match[1]]
                x, y = pixel
                Image[x, y] = new_pixel_val
                Mask[x, y] = 1
                progress = True
                filled_pixels += 1

                # Save an image snapshot every 10 filled pixels
                if filled_pixels % 500 == 0:
                    plt.imsave(f'{output_dir}/snapshot_c_{filled_pixels}.png', Image)
                    print(f'Filled pixel: ({x}, {y}) with value: {new_pixel_val}')

        if not progress:
            MaxErrThreshold *= 1.1
    return Image

# Example usage
from PIL import Image
seed_image_path = '6_small.png'
seed_image = np.array(Image.open(seed_image_path).convert('RGB')) / 255.0  # Convert to RGB and normalize to 0-1 range
WindowSize = 45

# Initialize the mask and output image
border_size = 200
border_size_h = border_size // 2
output_image_size = (seed_image.shape[0] + border_size, seed_image.shape[1] + border_size, 3)
output_image = np.zeros(output_image_size)
mask = np.zeros(output_image_size[:2])  # Create a 2D mask with the same size as the output image

# Calculate the center coordinates of the output image
center_row = output_image.shape[0] // 2
center_col = output_image.shape[1] // 2

# Calculate the starting and ending indices for the square region
start_row = center_row - WindowSize // 2
end_row = start_row + WindowSize
start_col = center_col - WindowSize // 2
end_col = start_col + WindowSize

# Extract the square region from the center of the seed image
seed_image_square = seed_image[seed_image.shape[0]//2 - WindowSize//2 : seed_image.shape[0]//2 + WindowSize//2 + 1,
                                seed_image.shape[1]//2 - WindowSize//2 : seed_image.shape[1]//2 + WindowSize//2 + 1]

# Place the square region in the center of the output image
output_image[start_row:end_row, start_col:end_col] = seed_image_square

# Set the corresponding region in the mask to 1
mask[start_row:end_row, start_col:end_col] = 1

MaxErrThreshold = 0.1

synthesized_image = GrowImage(seed_image, output_image, mask, WindowSize, MaxErrThreshold)
plt.imshow(synthesized_image)
#convert from 0-1 to 0-255
synthesized_image = (synthesized_image * 255).astype(np.uint8)
#save iamge not using plt
from PIL import Image
Image.fromarray(synthesized_image).save('synthesized_image_c.png')
plt.savefig('synthesized_image_c.png')
plt.imshow(mask, cmap='gray')
plt.savefig('mask_rgb.png')