import os
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# === CONFIGURACIÓN ===
modelo_id = "tiiuae/falcon-rw-1b"  # Ligero, público y sin login
carpeta_textos = "facturas_texto"
carpeta_json = "facturas_json"
os.makedirs(carpeta_json, exist_ok=True)

# === CARGA DE MODELO EN CPU ===
print("Cargando modelo ligero (esto puede tardar un poco)...")
tokenizer = AutoTokenizer.from_pretrained(modelo_id)
model = AutoModelForCausalLM.from_pretrained(modelo_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

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
\"\"\"{texto}\"\"\"

Devuelve solo un objeto JSON con los campos extraídos.
"""

for archivo in os.listdir(carpeta_textos):
    if archivo.endswith(".txt"):
        ruta_txt = os.path.join(carpeta_textos, archivo)
        print(f"Procesando: {archivo}")

        with open(ruta_txt, "r", encoding="utf-8") as f:
            texto = f.read()

        prompt = construir_prompt(texto)
        resultado = pipe(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]

        inicio = resultado.find("{")
        fin = resultado.rfind("}")
        if inicio != -1 and fin != -1:
            try:
                datos = json.loads(resultado[inicio:fin+1])
            except Exception as e:
                print(f"⚠️ Error al parsear JSON: {e}")
                continue
        else:
            print("⚠️ No se encontró JSON en la salida.")
            continue

        ruta_salida = os.path.join(carpeta_json, archivo.replace(".txt", ".json"))
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)

        print(f"✅ Guardado en: {ruta_salida}")
