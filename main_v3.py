import os
import re
import cv2
import pytesseract
import layoutparser as lp
from PIL import Image

# --- MONKEY PATCH (if needed) to avoid AttributeError: PIL.Image has no attribute 'LINEAR' ---
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.BILINEAR
# ---------------------------------------------------------------------------------------------

from layoutparser.models import Detectron2LayoutModel
from transformers import BlipProcessor, BlipForConditionalGeneration

def fix_cache_filenames(cache_root):
    """Renames any file that includes '?dl=' in its name to remove the query string."""
    for root, _, files in os.walk(cache_root):
        for file_name in files:
            if "?dl=" in file_name:
                old_path = os.path.join(root, file_name)
                new_name = file_name.split("?")[0]
                new_path = os.path.join(root, new_name)
                print(f"Renaming {old_path} to {new_path}")
                os.rename(old_path, new_path)

def extract_numeric_suffix(filename):
    """
    Extracts the numeric part from the filename for sorting.
    E.g. 'page_10.jpg' -> 10, 'page_2.jpg' -> 2
    If no number is found, returns 0 or some default.
    """
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def process_image(
    image_path,
    layout_model,
    caption_processor,
    caption_model,
    tesseract_lang="eng"
):
    """Extract text regions and generate captions for figure regions."""
    # Load with OpenCV (BGR)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return "", []
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect layout
    layout = layout_model.detect(image_rgb)
    
    # Separate text vs. figure regions
    text_blocks = [b for b in layout if b.type in ["Text", "Title", "List", "Table"]]
    figure_blocks = [b for b in layout if b.type == "Figure"]
    
    # --- Extract text (Tesseract) ---
    extracted_text = ""
    for block in text_blocks:
        x1, y1, x2, y2 = map(int, block.coordinates)
        roi = image_rgb[y1:y2, x1:x2]
        # Add the desired language for OCR
        text = pytesseract.image_to_string(roi, lang=tesseract_lang)
        extracted_text += text + "\n"
    
    # --- Generate captions (BLIP) ---
    explanations = []
    for block in figure_blocks:
        x1, y1, x2, y2 = map(int, block.coordinates)
        roi = image_rgb[y1:y2, x1:x2]
        pil_roi = Image.fromarray(roi)
        
        inputs = caption_processor(pil_roi, return_tensors="pt")
        output_ids = caption_model.generate(**inputs)
        caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
        explanations.append(caption)
    
    return extracted_text.strip(), explanations

def process_folder(
    folder_path,
    output_file="output.txt",
    tesseract_lang="eng"
):
    """
    Processes all images in a folder:
    1) Numeric sort for correct ordering (page_1 before page_2).
    2) Extract text (via Tesseract) in chosen language.
    3) Generate figure explanations (BLIP).
    4) Save combined results to a single file.
    """
    
    # Fix any potential iopath filenames with '?dl='
    cache_root = os.path.expanduser("~/.torch/iopath_cache")
    if os.path.exists(cache_root):
        fix_cache_filenames(cache_root)
    
    # Initialize the LayoutParser model (PubLayNet)
    layout_model = Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
    )
    
    # Initialize BLIP
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Gather image files (PNG, JPG, JPEG) and sort them numerically
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
           and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    # Sort by numeric suffix to ensure page_1 < page_2 < page_10, etc.
    image_files.sort(key=extract_numeric_suffix)
    
    all_text = []
    all_explanations = []
    
    for fname in image_files:
        full_path = os.path.join(folder_path, fname)
        print(f"Processing: {full_path}")
        
        text, figs = process_image(
            full_path,
            layout_model,
            caption_processor,
            caption_model,
            tesseract_lang=tesseract_lang  # either "eng", "por", or "por+eng", etc.
        )
        
        # Accumulate results
        text_block = f"--- {fname} ---\n{text}\n"
        all_text.append(text_block)
        
        if figs:
            fig_block = f"--- {fname} (Figures) ---\n"
            for i, caption in enumerate(figs, 1):
                fig_block += f"Figure {i}: {caption}\n"
            all_explanations.append(fig_block)
    
    # Write all results to output file
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("Extracted Text:\n\n")
        out.write("\n".join(all_text))
        out.write("\nExplanations of Figures:\n\n")
        out.write("\n".join(all_explanations))
    
    print(f"Done! Results saved in {output_file}")

if __name__ == "__main__":
    # Example usage:
    folder = "/home/regis/Documents/Git/IMAGE2TXT/calculo"         # folder with page_1.jpg, page_2.jpg, etc.
    out_file = "result.txt"  
    # Choose language: 'eng' or 'por' or 'por+eng' (if you have both languages installed)
    lang = "eng+por"
    
    process_folder(folder, out_file, tesseract_lang=lang)
