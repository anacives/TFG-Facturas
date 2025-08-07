import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# Ruta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# IMPORTANTE: Ruta a la carpeta que contiene "tessdata"
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR'

# Carpeta de entrada y salida
carpeta_facturas = 'facturas recibidas vento'
carpeta_textos = 'facturas_texto'
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

            # OCR
            texto = pytesseract.image_to_string(img, lang='spa')
            texto_final += f'\n\n--- Página {i+1} ---\n\n' + texto

        # Guardamos el texto extraído
        nombre_salida = os.path.splitext(archivo)[0] + '.txt'
        ruta_salida = os.path.join(carpeta_textos, nombre_salida)
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            f.write(texto_final)

        print(f'Texto guardado en: {ruta_salida}')
