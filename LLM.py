import os
from transformers import pipeline

# Cargar el modelo NER preentrenado
nlp_ner = pipeline("ner", model="mrm8488/bert-spanish-cased-finetuned-ner", tokenizer="mrm8488/bert-spanish-cased-finetuned-ner", grouped_entities=True)

# Carpeta con los txts extraídos del OCR
carpeta_textos = "facturas_texto"

# Procesar cada archivo
for archivo in os.listdir(carpeta_textos):
    if archivo.endswith(".txt"):
        ruta = os.path.join(carpeta_textos, archivo)
        print(f"\n📄 Procesando: {archivo}")

        with open(ruta, "r", encoding="utf-8") as f:
            texto = f.read()

        # Ejecutar NER
        entidades = nlp_ner(texto)

        # Mostrar entidades encontradas
        for ent in entidades:
            print(f"{ent['entity_group']}: {ent['word']} (score={ent['score']:.2f})")
