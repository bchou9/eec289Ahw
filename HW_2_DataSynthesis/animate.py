from PIL import Image
import os

# Define the directory containing your PNG files
input_directory = '.'  # Current directory
output_directory = '.'  # Current directory
output_gif_path = os.path.join(output_directory, 'animated3.gif')

# Helper function to extract numbers from filenames for sorting
def extract_number(filename):
    # Extracts numbers after the last underscore in the filename
    num_part = filename.split('_')[-1]
    return int(num_part.split('.')[0])  # Remove the file extension and convert to int

# List all PNG files in the directory and sort them by numeric value
png_files_all = [f for f in os.listdir(input_directory) if f.endswith('.png')]
png_files = [f for f in png_files_all if 'snapshot_c' in f]
png_files.sort(key=extract_number)  # Use extract_number function to sort files

# Load images and crop 50 pixels from each edge
cropped_images = []

crop_dim = 96//2  # Crop 80 pixels from each edge
for filename in png_files:
    path = os.path.join(input_directory, filename)
    with Image.open(path) as img:
        width, height = img.size
        # Define the box to crop (left, upper, right, lower)
        crop_box = (crop_dim, crop_dim, width - crop_dim, height - crop_dim)
        cropped_image = img.crop(crop_box)
        cropped_images.append(cropped_image)

# Save the sequence of cropped images as an animated GIF
cropped_images[0].save(output_gif_path, save_all=True, append_images=cropped_images[1:], optimize=False, duration=100, loop=0)
