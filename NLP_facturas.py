from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import os
import json
import re

# Cargar modelo NER en español
model_name = "PlanTL-GOB-ES/roberta-base-bne-capitel-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

carpeta = "facturas_texto"

def limpiar_texto(texto):
    texto = texto.replace('\n', ' ').replace('\t', ' ')
    texto = texto.replace('€', ' EUR')
    texto = texto.replace('O', '0')
    texto = texto.replace('lVA', 'IVA')
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def extraer_datos(texto):
    texto_limpio = limpiar_texto(texto)
    ner_result = ner_pipeline(texto_limpio)

    resultado = {
        "fecha_emision": None,
        "proveedor": None,
        "importe_total": None,
        "moneda": None,
        "cif_nif": None,
        "numero_factura": None
    }

    # 1. Extracción NER
    for entidad in ner_result:
        et = entidad['entity_group']
        valor = entidad['word']

        if et == "DATE" and resultado["fecha_emision"] is None:
            resultado["fecha_emision"] = valor
        elif et == "ORG" and resultado["proveedor"] is None:
            resultado["proveedor"] = valor
        elif et == "MONEY" and resultado["importe_total"] is None:
            resultado["importe_total"] = valor
            if "EUR" in valor or "€" in valor:
                resultado["moneda"] = "EUR"

    # 2. Regex de respaldo

    # Fecha si NER no detectó
    if resultado["fecha_emision"] is None:
        match_fecha = re.search(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', texto_limpio)
        if match_fecha:
            resultado["fecha_emision"] = match_fecha.group(1)

    # CIF/NIF español
    cif = re.search(r'\b[A-Z]\d{7}[A-Z0-9]\b', texto_limpio)
    if cif:
        resultado["cif_nif"] = cif.group()

    # Número de factura flexible
    num_factura = re.search(
        r'(Factura(?:\s*N[ºo.]?)?|N[ºo.]?\s*Factura)?[\s:]*([A-Z]?[0-9\-\/]{4,})',
        texto_limpio,
        re.IGNORECASE
    )
    if num_factura:
        resultado["numero_factura"] = num_factura.group(2)

    # Importe total por regex si no lo detectó NER
    if resultado["importe_total"] is None:
        patrones_importe = [
            r'total\s+factura',
            r'importe\s+total',
            r'total\s+recibo',
            r'total\s+euros',
            r'total\s*\(s\.?e\.?u\.?o\.?\)',
            r'total\s+factura\s+de\s+cargo',
            r'total\s+adeudado\s*\(s\.?e\.?u\.?o\.?\)',
            r'importe\s+total\s*[€eur]*',
            r'total\s+a\s+pagar\s*\(s\.?e\.?u\.?o\.?\)?',
            r'total\s+a\s+pagar',
            r'total\s*\(\*\)',
            r'total\s*eur',
            r'^total\b',
            r'^importe\b'
        ]

        for patron in patrones_importe:
            regex = rf'{patron}[\s:]*([\d\.,]+)\s*(EUR|€)?'
            match = re.search(regex, texto_limpio, re.IGNORECASE)
            if match:
                resultado["importe_total"] = match.group(1) + (" EUR" if match.group(2) else "")
                if resultado["moneda"] is None and match.group(2):
                    resultado["moneda"] = "EUR"
                break


        # Detectar EUR si está suelto
        if resultado["moneda"] is None and ("EUR" in texto_limpio or "€" in texto):
            resultado["moneda"] = "EUR"

        return resultado

os.makedirs("resultados_json", exist_ok=True)

for archivo in os.listdir(carpeta):
    if archivo.endswith(".txt"):
        ruta = os.path.join(carpeta, archivo)
        with open(ruta, 'r', encoding='utf-8') as f:
            texto = f.read()

        datos = extraer_datos(texto)

        print(f"\n--- {archivo} ---")
        print(json.dumps(datos, indent=2, ensure_ascii=False))

        nombre_json = archivo.replace(".txt", ".json")
        with open(os.path.join("resultados_json", nombre_json), 'w', encoding='utf-8') as fout:
            json.dump(datos, fout, ensure_ascii=False, indent=2)
