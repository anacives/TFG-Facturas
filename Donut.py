import os
import fitz
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import json
import torch

# Carpetas
carpeta_facturas = "facturas_recibidas_vento"
carpeta_salida = "facturas_json_donut_llm"
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar Donut
print("⏳ Cargando Donut...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-invoices")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-invoices")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Cargar modelo LLM pequeño y gratuito para fallback
print("⏳ Cargando LLM fallback...")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

# Función para extraer con LLM
def extraer_campos_llm(texto, campos):
    prompt = f"Extrae los siguientes campos de esta factura: {campos}. Responde en JSON.\nFactura:\n{texto}"
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = llm_model.generate(**inputs, max_length=256)
    salida = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        return json.loads(salida)
    except:
        return {}

# Campos clave
campos_clave = ["emisor", "fecha", "IBAN", "importe", "IVA", "numero_factura"]

# Procesar PDFs
for archivo in os.listdir(carpeta_facturas):
    if archivo.lower().endswith(".pdf"):
        ruta_pdf = os.path.join(carpeta_facturas, archivo)
        print(f"📄 Procesando: {archivo}")

        doc = fitz.open(ruta_pdf)
        datos_factura = []

        for pagina in doc:
            # Convertir página a imagen
            pix = pagina.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Donut
            pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
            task_prompt = "<s_invoices>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.config.decoder.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )

            sequence = processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            sequence = sequence.lstrip(task_prompt)

            try:
                datos_json = json.loads(sequence)
            except:
                datos_json = {}

            # Fallback con LLM para campos faltantes
            faltantes = [c for c in campos_clave if c not in datos_json or not datos_json[c]]
            if faltantes:
                texto_pagina = pagina.get_text("text")
                llm_result = extraer_campos_llm(texto_pagina, faltantes)
                datos_json.update({k: llm_result.get(k, "") for k in faltantes})

            # Mantener solo campos clave
            datos_factura.append({k: datos_json.get(k, "") for k in campos_clave})

        # Guardar resultado
        if datos_factura:
            ruta_salida = os.path.join(carpeta_salida, os.path.splitext(archivo)[0] + ".json")
            with open(ruta_salida, "w", encoding="utf-8") as f:
                json.dump(datos_factura, f, ensure_ascii=False, indent=4)
            print(f"✅ JSON guardado en: {ruta_salida}")
