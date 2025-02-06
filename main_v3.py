import os
import re
import cv2
import pytesseract
import layoutparser as lp
from PIL import Image

##########################################
# 1) MONKEY PATCH: PIL.Image.LINEAR -> PIL.Image.BILINEAR
##########################################
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.BILINEAR

##########################################
# 2) CONTINUE IMPORTS (após o patch)
##########################################
from layoutparser.models import Detectron2LayoutModel
from transformers import BlipProcessor, BlipForConditionalGeneration

##########################################
# Função para corrigir nomes na cache
##########################################
def fix_cache_filenames(cache_root):
    for root, _, files in os.walk(cache_root):
        for file_name in files:
            if "?dl=" in file_name:
                old_path = os.path.join(root, file_name)
                new_name = file_name.split("?")[0]
                new_path = os.path.join(root, new_name)
                print(f"Renomeando {old_path} para {new_path}")
                try:
                    os.rename(old_path, new_path)
                except Exception as e:
                    print(f"Erro ao renomear {old_path}: {e}")

##########################################
# Função para extrair número para ordenação de arquivos
##########################################
def extract_numeric_suffix(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

##########################################
# Função para ordenar blocos de layout pela posição (primeiro por y1, depois por x1)
##########################################
def sort_blocks_by_position(blocks):
    # Cada bloco possui coordenadas no formato [x1, y1, x2, y2]
    # Ordenamos pelo y1 (linha superior) e, em caso de empate, pelo x1 (coluna esquerda)
    return sorted(blocks, key=lambda b: (b.coordinates[1], b.coordinates[0]))

##########################################
# Função para limpar o texto removendo caracteres especiais indesejados
##########################################
def limpar_texto(texto):
    try:
        # Remove tudo que não seja alfanumérico, espaço ou pontuação básica
        texto_limpo = re.sub(r"[^\w\s.,!?;:()\-]", "", texto)
        return texto_limpo
    except Exception as e:
        print(f"Erro ao limpar texto: {e}")
        return texto

##########################################
# Função para processar uma única imagem (página do livro)
##########################################
def process_image(image_path, layout_model, caption_processor, caption_model, tesseract_lang="eng"):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Não foi possível carregar a imagem: {image_path}")
            return "", []
    except Exception as e:
        print(f"Erro ao carregar a imagem {image_path}: {e}")
        return "", []
    
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Erro na conversão da imagem {image_path}: {e}")
        return "", []
    
    try:
        layout = layout_model.detect(image_rgb)
    except Exception as e:
        print(f"Erro na detecção de layout da imagem {image_path}: {e}")
        return "", []
    
    # Separa os blocos de texto e de figura e os ordena pela posição na página
    text_blocks = sort_blocks_by_position([b for b in layout if b.type in ["Text", "Title", "List", "Table"]])
    figure_blocks = sort_blocks_by_position([b for b in layout if b.type == "Figure"])
    
    extracted_text = ""
    for block in text_blocks:
        try:
            x1, y1, x2, y2 = map(int, block.coordinates)
            roi = image_rgb[y1:y2, x1:x2]
            config = "--psm 6"  # Pode ajustar o modo de segmentação de página conforme necessário
            text = pytesseract.image_to_string(roi, lang=tesseract_lang, config=config)
            text = limpar_texto(text)
            extracted_text += text + "\n"
        except Exception as e:
            print(f"Erro ao processar região de texto em {image_path}: {e}")
    
    explanations = []
    for block in figure_blocks:
        try:
            x1, y1, x2, y2 = map(int, block.coordinates)
            roi = image_rgb[y1:y2, x1:x2]
            pil_img = Image.fromarray(roi)
            inputs = caption_processor(pil_img, return_tensors="pt")
            output_ids = caption_model.generate(**inputs)
            caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
            explanations.append(caption)
        except Exception as e:
            print(f"Erro ao gerar legenda para figura em {image_path}: {e}")
    
    return extracted_text.strip(), explanations

##########################################
# Função para processar todas as imagens de uma pasta
# Gravando os resultados de cada página imediatamente no arquivo
##########################################
def process_folder(folder_path, output_file="resultado.txt", tesseract_lang="eng"):
    cache_root = os.path.expanduser("~/.torch/iopath_cache")
    if os.path.exists(cache_root):
        fix_cache_filenames(cache_root)
    
    try:
        layout_model = Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
        )
    except Exception as e:
        print(f"Erro ao inicializar o modelo de layout: {e}")
        return
    
    try:
        caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    except Exception as e:
        print(f"Erro ao inicializar o modelo BLIP: {e}")
        return
    
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_files.sort(key=extract_numeric_suffix)
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        for fname in image_files:
            full_path = os.path.join(folder_path, fname)
            print(f"Processando: {full_path}")
            
            text, figs = process_image(full_path, layout_model, caption_processor, caption_model, tesseract_lang)
            
            # Cria um bloco com delimitadores claros para cada página
            block = []
            block.append("=" * 80)
            block.append(f"Página: {fname}")
            block.append("=" * 80)
            block.append("")
            block.append(text)
            block.append("")
            if figs:
                block.append(f"--- Legendas para figuras em {fname} ---")
                for i, caption in enumerate(figs, 1):
                    block.append(f"Figura {i}: {caption}")
                block.append("")
            
            block_str = "\n".join(block) + "\n\n"
            out_f.write(block_str)
            out_f.flush()
    
    print(f"\nProcessamento concluído. Resultados salvos em {output_file}")

##########################################
# Execução Principal
##########################################
if __name__ == "__main__":
    # Example usage:
    folder = "/home/regis/Documents/Git/IMAGE2TXT/calculo"         # folder with page_1.jpg, page_2.jpg, etc.
    out_file = "result.txt"  
    # Choose language: 'eng' or 'por' or 'por+eng' (if you have both languages installed)
    lang = "eng+por"
    
    process_folder(folder, out_file, tesseract_lang=lang)
