import os
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# === CONFIGURACIÓN ===
modelo_id = "mistralai/Mistral-7B-Instruct-v0.2"
carpeta_textos = "facturas_texto"
carpeta_json = "facturas_json"
os.makedirs(carpeta_json, exist_ok=True)

# === CARGAR MODELO (en CPU) ===
print("Cargando modelo, esto puede tardar unos minutos...")
tokenizer = AutoTokenizer.from_pretrained(modelo_id)
model = AutoModelForCausalLM.from_pretrained(modelo_id, device_map="auto")  # Usa CPU automáticamente

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if model.device.type == "cuda" else -1)

# === CAMPOS A EXTRAER ===
campos = [
    "fecha_emision", 
    "proveedor", 
    "importe_total", 
    "moneda", 
    "cif_nif", 
    "numero_factura"
]

def construir_prompt(texto):
    return f"""
Extrae del siguiente texto los siguientes campos:
{', '.join(campos)}

Texto de la factura:
\"\"\"
{texto}
\"\"\"

Devuelve los campos en formato JSON.
"""

# === PROCESAR FACTURAS ===
for archivo in os.listdir(carpeta_textos):
    if archivo.endswith(".txt"):
        ruta_txt = os.path.join(carpeta_textos, archivo)
        print(f"Procesando: {archivo}")

        with open(ruta_txt, "r", encoding="utf-8") as f:
            texto = f.read()

        prompt = construir_prompt(texto)

        respuesta = pipe(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]

        # Intenta localizar el primer bloque JSON dentro del texto generado
        inicio = respuesta.find("{")
        fin = respuesta.rfind("}")
        if inicio != -1 and fin != -1 and fin > inicio:
            json_text = respuesta[inicio:fin+1]
            try:
                datos = json.loads(json_text)
            except json.JSONDecodeError:
                print(f"⚠️ Error al decodificar JSON en {archivo}")
                continue
        else:
            print(f"⚠️ JSON no encontrado en {archivo}")
            continue

        # Guardamos como .json
        nombre_json = os.path.splitext(archivo)[0] + ".json"
        ruta_json = os.path.join(carpeta_json, nombre_json)
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)

        print(f"✅ Guardado en: {ruta_json}")
