from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from dotenv import load_dotenv
import torch
import os

load_dotenv()  # Carga variables de entorno desde .env
hf_token = os.getenv("HUGGINGFACE_TOKEN")

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch_dtype
)

print("Cargando modelo LLaMA 3 8B Instruct...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch_dtype,
    use_auth_token=hf_token
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

PROMPT_BASE = """
Eres NetAgent, un asistente técnico experto en Fortinet. Tu tarea es interpretar estados y datos de dispositivos FortiGate desde su API y ayudar a operadores humanos con instrucciones claras y específicas.

Cuando recibas datos de estado o configuración, analiza la situación y proporciona respuestas claras, breves y precisas. Usa lenguaje técnico cuando sea necesario, pero siempre buscando claridad.
Responde siempre de forma breve y precisa, sin repetir frases ni preguntar si quieres añadir algo más.

Pregunta:
{pregunta}

Respuesta:

Formato de respuesta:
----------------------
PREGUNTA:
{pregunta}
----------------------
RESPUESTA:
"""

MAX_PROMPT_LENGTH = 1000  # Ajustar según memoria y límites del modelo (en caracteres)

def recortar_contexto(contexto: str) -> str:
    """Recorta el contexto para mantener solo la parte más reciente si es muy largo."""
    if len(contexto) <= MAX_PROMPT_LENGTH:
        return contexto
    return contexto[-MAX_PROMPT_LENGTH:]

def generar_respuesta(contexto: str) -> str:
    # Construye prompt con plantilla y recorta si es necesario
    prompt = PROMPT_BASE.replace("{pregunta}", contexto)
    prompt = recortar_contexto(prompt)

    print("\n--- PROMPT ENVIADO AL MODELO ---\n")
    print(prompt)

    res = pipe(prompt, max_new_tokens=400, do_sample=True, temperature=0.7)
    respuesta = res[0]['generated_text'].replace(prompt, "").strip()

    print("\n--- RESPUESTA GENERADA POR EL MODELO ---\n")
    print(respuesta)

    return respuesta

def consultar_netagent(pregunta: str):
    # Función auxiliar para uso directo (sin contexto largo)
    prompt = PROMPT_BASE.replace("{pregunta}", pregunta)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    seccion = respuesta.split("----------------------\nRESPUESTA:", 1)
    if len(seccion) > 1:
        print("----------------------\nPREGUNTA:")
        print(pregunta)
        print("----------------------\nRESPUESTA:")
        print(seccion[1].strip())
    else:
        print("Respuesta completa:\n", respuesta)

if __name__ == "__main__":
    pregunta = "Imagina que te he conectado a la API de un FortiGate. ¿Cómo puedo obtener el estado de las interfaces y las políticas de seguridad?"
    consultar_netagent(pregunta)
