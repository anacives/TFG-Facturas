import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import sys 

# 📌 Ruta a Tesseract en Windows
TESSERACT_PATH = r"C:\Users\anaci\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# 🔍 Verificar que Tesseract existe
if not os.path.exists(TESSERACT_PATH):
    print(f"❌ ERROR: No se encontró Tesseract en:\n{TESSERACT_PATH}")
    print("➡ Revisa la ruta o instala Tesseract desde:")
    print("   https://github.com/UB-Mannheim/tesseract/wiki")
    sys.exit(1)
else:
    print(f"✅ Tesseract encontrado en: {TESSERACT_PATH}")
    print(f"Versión detectada: {pytesseract.get_tesseract_version()}")

# Carpetas
carpeta_facturas = "facturas recibidas vento"
carpeta_salida = "facturas_texto_final"
os.makedirs(carpeta_salida, exist_ok=True)

# EasyOCR lector en español
reader = easyocr.Reader(['es'])

# Función de preprocesado
def preprocesar_imagen(pil_img):
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    return img

# Combinar resultados
def combinar_textos(texto_tess, resultados_easy):
    # texto_tess como backup, EasyOCR por líneas con confianza alta
    lineas_tess = texto_tess.splitlines()
    texto_final = []

    # Convertimos resultados_easy a dict {posicion: texto}
    for resultado in resultados_easy:
        caja, texto, confianza = resultado
        if confianza >= 0.85:
            texto_final.append(texto)
    
    # Si EasyOCR no dio suficiente, usamos Tesseract
    if len(texto_final) < len(lineas_tess) * 0.5:
        texto_final.extend(lineas_tess)

    return "\n".join(texto_final)

# Procesar PDFs
for archivo in os.listdir(carpeta_facturas):
    if archivo.lower().endswith(".pdf"):
        ruta_pdf = os.path.join(carpeta_facturas, archivo)
        print(f"📄 Procesando: {archivo}")

        doc = fitz.open(ruta_pdf)
        texto_factura = ""

        for i, pagina in enumerate(doc):
            pix = pagina.get_pixmap(dpi=300)
            img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_cv = preprocesar_imagen(img_pil)

            # OCR Tesseract
            texto_tess = pytesseract.image_to_string(img_cv, lang='spa')

            # OCR EasyOCR
            resultados_easy = reader.readtext(img_cv)

            # Combinar
            texto_final_pagina = combinar_textos(texto_tess, resultados_easy)

            texto_factura += f"\n\n--- Página {i+1} ---\n\n{texto_final_pagina}"

        # Guardar
        ruta_salida = os.path.join(carpeta_salida, os.path.splitext(archivo)[0] + "_final.txt")
        with open(ruta_salida, "w", encoding="utf-8") as f:
            f.write(texto_factura)

        print(f"✅ Guardado en: {ruta_salida}")
