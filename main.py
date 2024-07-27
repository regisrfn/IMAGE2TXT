import pytesseract
from PIL import Image
import os

def extract_text_from_images(image_folder, output_file):
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    # Open the output file for writing
    with open(output_file, 'w') as outfile:
        for image_file in image_files:
            # Full path to the image file
            image_path = os.path.join(image_folder, image_file)

            # Open the image file
            with Image.open(image_path) as img:
                # Extract text from the image using pytesseract
                text = pytesseract.image_to_string(img)

                # Write the text to the output file
                outfile.write(f"Content of file {image_file}:\n")
                outfile.write(text)
                outfile.write("\n\n")

    print(f"Text extraction completed. Check the output file: {output_file}")

if __name__ == "__main__":
    # Define the path to the folder containing images and the output file
    image_folder_path = "/home/regis/Documents/Git/IMAGE2TXT"
    output_file_path = "The_Name_of_the_Wind/output.txt"

    # Perform the text extraction
    extract_text_from_images(image_folder_path, output_file_path)

