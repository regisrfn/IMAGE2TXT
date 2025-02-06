import cv2
import pytesseract
import numpy as np
import logging
from PIL import Image
from layoutparser import models
from transformers import BlipProcessor, BlipForConditionalGeneration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookConverter:
    def __init__(self):
        # Configure Tesseract
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
        # Initialize layout detection model (CPU-only)
        self.layout_model = models.Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            model_path='https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth',
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
        )
        
        # Initialize BLIP model for image captioning
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        # OCR configuration
        self.ocr_config = r'--oem 3 --psm 6 -l eng+por'
        
        # Character replacement mapping
        self.char_replacements = {
            '◁': 'Δ', '▷': 'Δ', '▶': 'Δ',
            '⇒': '→', '▯': '□', '≅': '≈'
        }

    def _preprocess_image(self, image_path):
        """Load and validate input image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Image loading error: {str(e)}")
            raise

    def _enhance_region(self, image):
        """Enhance image region for better OCR results"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    def _process_text_block(self, image_block):
        """Process a single text block with OCR"""
        try:
            enhanced = self._enhance_region(image_block)
            enhanced = cv2.resize(enhanced, None, fx=2, fy=2, 
                                interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(
                enhanced, config=self.ocr_config
            )
            return self._correct_ocr_errors(text)
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            return ""

    def _generate_caption(self, image):
        """Generate caption for figures"""
        try:
            inputs = self.processor(image, return_tensors="pt")
            out = self.caption_model.generate(**inputs)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Caption generation error: {str(e)}")
            return "Could not generate caption"

    def _correct_ocr_errors(self, text):
        """Correct common OCR mistakes"""
        for wrong, correct in self.char_replacements.items():
            text = text.replace(wrong, correct)
        return text.strip()

    def convert(self, image_path):
        """Main conversion method"""
        try:
            image = self._preprocess_image(image_path)
            layout = self.layout_model.detect(image)
            
            # Sort elements by position
            elements = sorted(layout, key=lambda x: (x.block.y_1, x.block.x_1))
            
            output_text = []
            figures = []
            
            for element in elements:
                x1, y1, x2, y2 = map(int, element.block.coordinates)
                region = image[y1:y2, x1:x2]
                
                if element.type == 'Figure':
                    pil_image = Image.fromarray(region)
                    caption = self._generate_caption(pil_image)
                    figures.append(f"[FIGURE {len(figures)+1}]: {caption}")
                else:
                    text = self._process_text_block(region)
                    output_text.append(text)
            
            # Combine results
            return '\n\n'.join(output_text) + '\n\nFIGURES:\n' + '\n'.join(figures)
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            raise

if __name__ == "__main__":
    
    try:
        converter = BookConverter()
        result = converter.convert("/home/regis/Documents/Git/IMAGE2TXT/calculo/page_94.jpg")
        
        with open('output.txt', 'w', encoding='utf-8') as f:
            f.write(result)
            
        print("Conversion successful. Output saved to output.txt")
        
    except Exception as e:
        print(f"Critical error: {str(e)}")