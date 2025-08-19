import os
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import numpy as np

# Inicializamos el lector de EasyOCR (con español y/o inglés)
reader = easyocr.Reader(['es', 'en'])

# Carpeta de entrada y salida
carpeta_facturas = 'facturas recibidas vento'
carpeta_textos = 'facturas_texto2'
os.makedirs(carpeta_textos, exist_ok=True)

# Recorremos todos los PDFs
for archivo in os.listdir(carpeta_facturas):
    if archivo.lower().endswith('.pdf'):
        ruta_pdf = os.path.join(carpeta_facturas, archivo)
        print(f'Procesando: {archivo}')

        doc = fitz.open(ruta_pdf)
        texto_final = ''

        for i, pagina in enumerate(doc):
            # Renderiza la página como imagen (pixmap) a 300 DPI aprox.
            pix = pagina.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convertir la imagen a formato compatible con EasyOCR (como array)
            img_array = np.array(img)

            # OCR con EasyOCR
            resultado = reader.readtext(img_array, detail=0, paragraph=True)
            texto = "\n".join(resultado)

            texto_final += f'\n\n--- Página {i+1} ---\n\n' + texto

        # Guardamos el texto extraído
        nombre_salida = os.path.splitext(archivo)[0] + '.txt'
        ruta_salida = os.path.join(carpeta_textos, nombre_salida)
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            f.write(texto_final)

        print(f'Texto guardado en: {ruta_salida}')
