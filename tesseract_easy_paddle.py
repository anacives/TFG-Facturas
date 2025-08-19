import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
from paddleocr import PaddleOCR
import re
import json
from tqdm import tqdm

# Ruta a Tesseract (ajusta si es necesario)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\anaci\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Carpetas
carpeta_facturas = "facturas recibidas vento"
carpeta_salida = "facturas texto paddle"
os.makedirs(carpeta_salida, exist_ok=True)

# Inicializar OCRs (una sola vez)
reader_easy = easyocr.Reader(['es'], gpu=False)
reader_paddle = PaddleOCR(lang='es', use_angle_cls=True)

def preprocesar_imagen(pil_img):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    return img

def combinar_resultados(tess_texts, easy_results, paddle_results):
    lineas = set()

    # Tesseract
    for t in tess_texts:
        for linea in t.splitlines():
            if linea.strip():
                lineas.add(linea.strip())

    # EasyOCR
    for pagina in easy_results:
        for _, texto, conf in pagina:
            if conf >= 0.85:
                lineas.add(texto.strip())

    # PaddleOCR
    for pagina in paddle_results:
        for _, (texto, conf) in pagina:
            if conf >= 0.85:
                lineas.add(texto.strip())

    return "\n".join(sorted(lineas))

def extraer_campos(texto):
    campos = {}
    emisor = re.search(r'(?i)(?:(?!Banco)\b[A-ZÁÉÍÓÚÑ\s&,.]{3,})', texto)
    campos["emisor"] = emisor.group(0).strip() if emisor else ""
    fecha = re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', texto)
    campos["fecha_emision"] = fecha.group(0) if fecha else ""
    iban = re.search(r'\bES\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,4}\b', texto)
    campos["iban"] = iban.group(0).replace(" ", "") if iban else ""
    importe = re.search(r'\b\d{1,3}(?:\.\d{3})*,\d{2}\b', texto)
    campos["importe"] = importe.group(0) if importe else ""
    iva = re.search(r'\b\d{1,3},\d{2}\b(?=.*IVA)', texto, re.IGNORECASE)
    campos["iva"] = iva.group(0) if iva else ""
    num_factura = re.search(r'F\d{6,}', texto)
    campos["numero_factura"] = num_factura.group(0) if num_factura else ""
    return campos

# Procesar PDFs
for archivo in tqdm(os.listdir(carpeta_facturas), desc="Procesando facturas"):
    if archivo.lower().endswith(".pdf"):
        ruta_pdf = os.path.join(carpeta_facturas, archivo)
        doc = fitz.open(ruta_pdf)

        # 1️⃣ Renderizar todas las páginas y preprocesar
        imgs_pil = []
        imgs_cv = []
        for pagina in doc:
            pix = pagina.get_pixmap(dpi=300)
            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            imgs_pil.append(img_pil)
            imgs_cv.append(preprocesar_imagen(img_pil))

        # 2️⃣ Pasar por los OCRs en bloque
        tess_texts = [pytesseract.image_to_string(img, lang='spa') for img in imgs_cv]
        easy_results = reader_easy.readtext_batched(imgs_cv)
        paddle_results = [reader_paddle.ocr(np.array(img)) for img in imgs_pil]

        # 3️⃣ Combinar
        texto_factura = combinar_resultados(tess_texts, easy_results, paddle_results)

        # 4️⃣ Guardar TXT
        nombre_base = os.path.splitext(archivo)[0]
        ruta_txt = os.path.join(carpeta_salida, nombre_base + "_final.txt")
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(texto_factura)

        # 5️⃣ Extraer y guardar JSON
        campos = extraer_campos(texto_factura)
        ruta_json = os.path.join(carpeta_salida, nombre_base + "_campos.json")
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(campos, f, ensure_ascii=False, indent=4)
