import os
import cv2
import pytesseract
from layoutparser.models import Detectron2LayoutModel
from PIL import Image
import numpy as np

# Exemplo de mapeamento simples de correção de símbolos
SPECIAL_CHAR_MAP = {
    "∆": "Δ",   # se Tesseract confundir, podemos padronizar
    "ˆ": "^",
    "≈": "~",
    # Exemplo: se OCR reconhecer "A" mas contexto seria "Δ"
    # etc. Você pode expandir esse dicionário.
}

def corrige_caracteres_especiais(texto, char_map=SPECIAL_CHAR_MAP):
    """Corrige caracteres especiais no texto de acordo com mapeamento."""
    for errado, correto in char_map.items():
        texto = texto.replace(errado, correto)
    return texto

def processar_imagem_e_extrair_texto(caminho_imagem, detector, idx_pagina=1):
    """
    - Usa layoutparser para detectar blocos.
    - Identifica blocos de texto e de figura.
    - Faz OCR nos blocos de texto, e "marca" blocos de figura.
    - Retorna texto formatado + indicações de figura ao final de cada bloco de texto.
    """
    
    # Carregar a imagem
    image = cv2.imread(caminho_imagem)

    # Converter para RGB (caso seja BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detectar layout (retornos: lista de LayoutElement)
    layout = detector.detect(image_rgb)

    # Podemos filtrar os blocos por tipo (text, title, figure, table, etc.)
    # O modelo pré-treinado do layoutparser, dependendo da configuração,
    # costuma retornar categories como "Text", "Title", "List", "Table", "Figure", etc.

    # Ordenar blocos de acordo com posição vertical (top)
    # layout[i].block.coordinates -> (x_1, y_1, x_2, y_2)
    layout_sorted = sorted(layout, key=lambda l: l.block.y_1)

    # Vamos agrupar a saída de texto e figuras
    texto_final = []
    figura_count = 1

    for element in layout_sorted:
        label = element.type  # ex: 'Text', 'Figure', etc.
        # Coordinates
        x_1, y_1, x_2, y_2 = map(int, element.block.coordinates)
        
        # Recortar a região
        recorte = image_rgb[y_1:y_2, x_1:x_2]

        if label.lower() in ["text", "title", "list"]:  
            # Zoom se necessário (exemplo, se recorte for pequeno):
            h, w, _ = recorte.shape
            if h < 50 or w < 50:
                # Exemplo de upscaling 2x:
                recorte = cv2.resize(recorte, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Converter recorte para PIL para pytesseract
            pil_recorte = Image.fromarray(recorte)
            
            # OCR
            ocr_result = pytesseract.image_to_string(pil_recorte, lang='por+eng')  
            # Ajustar se usar português/inglês ou outra língua. 
            
            # Correção de caracteres especiais
            ocr_result = corrige_caracteres_especiais(ocr_result)
            
            # Limpar texto (remover quebras em excesso, etc.)
            ocr_result = ocr_result.strip()
            
            if ocr_result:
                texto_final.append(ocr_result + "\n")
        
        elif label.lower() in ["figure", "table", "figure caption"]:
            # Para simplificar, só vamos "marcar" que existe figura.
            # Caso deseje extrair descrições de figura (por exemplo, OCR dentro da figura ou
            # uso de modelos de descrição de imagem), você pode acoplar aqui.
            texto_final.append(f"\n[Figura {figura_count}: bloco detectado como '{label}']\n")
            figura_count += 1
        
        else:
            # Caso o modelo retorne outras classes não previstas
            texto_final.append(f"\n[Bloco detectado como '{label}' não tratado especificamente]\n")

    # Unir tudo
    texto_organizado = "\n".join(texto_final)
    return texto_organizado


def process_files(diretorio_imagens):
    # Inicializar detector de layout
    # Exemplo: usar um modelo pré-treinado 'prp' do LayoutParser.
    # Você pode trocar por 'detectron2_resnet50' ou outro.
    model = Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config", # Exemplo
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        model_path="/home/regis/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth",
        device="cpu"  # ou "cuda" se tiver GPU compatível
    )
    
    # Percorrer arquivos do diretório
    for arquivo in sorted(os.listdir(diretorio_imagens)):
        if arquivo.lower().endswith((".jpg", ".png", ".jpeg")) and arquivo.startswith("page_"):
            caminho = os.path.join(diretorio_imagens, arquivo)
            # Extrair número da página para referência
            # ex: "pagina_1.jpg" -> idx_pagina=1
            try:
                idx_pagina = int(arquivo.split("_")[1].split(".")[0])
            except:
                idx_pagina = 0
            
            texto_extraido = processar_imagem_e_extrair_texto(caminho, model, idx_pagina=idx_pagina)
            
            # Salvar em um arquivo txt
            nome_arquivo_txt = f"pagina_{idx_pagina}.txt"
            with open(os.path.join(diretorio_imagens, nome_arquivo_txt), "w", encoding="utf-8") as f:
                f.write(texto_extraido)
            
            print(f"Processado {arquivo} -> {nome_arquivo_txt}")


if __name__ == "__main__":
    # Exemplo de uso:
    # python extrair_texto.py
    
    diretorio = "/home/regis/Documents/Git/IMAGE2TXT/calculo"
    process_files(diretorio)
