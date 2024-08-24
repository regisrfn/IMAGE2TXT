# OCR Image to Text Transcriber

This Python application uses Tesseract OCR to transcribe text from image files and save the transcribed text to `.txt` files. The app processes all image files in a given folder and outputs the transcriptions to a specified directory.

## Features
- Automatically scans all image files in a folder.
- Supports multiple image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`.
- Saves transcriptions as unique `.txt` files, named based on the image file name (without the extension).
- Natural sorting for files with numerical names (e.g., `file1`, `file10`, `file2` will be sorted properly as `file1`, `file2`, `file10`).

## Requirements

- Python 3.x
- Tesseract OCR installed on your system
- The following Python libraries:
  - `pytesseract`
  - `Pillow`
  - `os`
  - `re`

## Installation

1. **Install Python Dependencies**  
First, clone or download this repository, then install the necessary Python libraries using `pip`:

```bash
pip install pytesseract pillow
```

2. **Install Tesseract OCR**  
Make sure Tesseract OCR is installed and available on your system. You can download it from the Tesseract OCR GitHub or use your system's package manager:

- For Ubuntu/Debian:
```bash
sudo apt-get install tesseract-ocr
```

- For MacOS (using Homebrew):
```bash
brew install tesseract
```

- For Windows:
Download and install from [this link](https://github.com/tesseract-ocr/tesseract/wiki).

3. **Set the Path to Tesseract Executable**  
Make sure to set the correct path to the Tesseract executable in the Python script. For example:

```python
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Example for Linux
```

On Windows, it might look like this:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Usage

1. **Prepare Your Folder of Images**  
Place all the image files you want to transcribe into a folder. Supported formats include `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`.

2. **Run the Script**  
Run the script and specify the folder containing the images and the output directory where the transcriptions will be saved. For example:

```bash
python your_script.py
```

The script will:
- Transcribe all image files in the specified folder.
- Save the transcriptions in the `output_texts` folder (or a custom output folder you provide).

Example:

If you have images in `/home/user/images` and want to save the transcriptions to `transcriptions/`:

```bash
python your_script.py /home/user/images transcriptions/
```

3. **Accessing the Transcribed Text**  
The transcriptions will be saved as `.txt` files in the output directory. Each `.txt` file will be named based on the original image file name (without the extension).

## Configuration

- **Input Folder**: This is where your images are stored. Update `folder_path` in the script or pass it as an argument when running the script.
- **Output Folder**: By default, the transcriptions are saved in the `output_texts` folder, but you can modify this by changing the `output_dir` variable or passing it as an argument.

## Notes

- This script processes each image file separately and saves a corresponding `.txt` file with the transcribed text.
- If you are working with large batches of images, the process might take some time depending on the size and complexity of the images.
- For best results with OCR, ensure the images are of high quality with clear, readable text.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Tesseract OCR for providing open-source OCR capabilities.
- Pillow for image processing in Python.

Let me know if you need further changes or additions to this!