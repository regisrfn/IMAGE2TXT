import pytesseract
from PIL import Image
import os
import re

# Path to the Tesseract executable (change this if needed)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def transcribe_image_to_text(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(img)
    return text

def save_text_to_unique_file(text, output_dir, index):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Generate a unique file name using the index
    unique_filename = f'{index:04d}.txt'
    output_path = os.path.join(output_dir, unique_filename)
    # Write the text to the file
    with open(output_path, 'w') as text_file:
        text_file.write(text)
    return output_path

def natural_sort_key(s):
    # Function to sort strings with numbers naturally
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def process_images_in_folder(folder_path, output_dir="output_texts"):
    # Get all image file paths in the folder
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    image_files.sort(key=natural_sort_key)  # Sort the image files to maintain natural order
    
    for index, filename in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, filename)
        text = transcribe_image_to_text(image_path)
        output_path = save_text_to_unique_file(text, output_dir, index)
        print(f"Transcribed text saved to: {output_path}")

# Example usage with a folder path
folder_path = '/home/regis/Documents/Git/PDF2IMG/The_Name_of_the_Wind/batch_101_to_200'
process_images_in_folder(folder_path)
