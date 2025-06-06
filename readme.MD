NetAgent - Agente LLM para redes Fortinet

Este proyecto monta un agente local basado en un modelo LLM (como LLaMA 2) adaptado para interactuar con APIs FortiGate y responder en lenguaje natural a operadores de red.

Pasos para preparar el entorno
	1.	Actualizar e instalar dependencias básicas del sistema:
sudo apt update
sudo apt install python3.12-venv python3-pip git
	2.	Crear un entorno virtual:
python3 -m venv ~/consultor_venv
	3.	Activar el entorno virtual:
source ~/consultor_venv/bin/activate
	4.	Actualizar pip:
pip install –upgrade pip
	5.	Instalar PyTorch con soporte CUDA 12.6:
pip install torch torchvision torchaudio –index-url https://download.pytorch.org/whl/cu126
	6.	Instalar librerías necesarias para modelos LLM:
pip install transformers accelerate bitsandbytes
	7.	Crear un archivo de script, por ejemplo llm.py, con el siguiente contenido básico:
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
model_name = “meta-llama/Llama-2-7b-chat-hf”
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=“auto”, load_in_8bit=True)
generator = pipeline(“text-generation”, model=model, tokenizer=tokenizer, device=0)
prompt = “Explícame la diferencia entre una policy y una static route en FortiGate.”
output = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
print(output[0][“generated_text”])
	8.	Crear cuenta en Hugging Face y solicitar acceso al modelo:
	•	Ir a https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
	•	Aceptar los términos del modelo
	9.	Una vez aprobado, iniciar sesión desde terminal:
huggingface-cli login
(Pegar el token cuando lo pida)
	10.	Ejecutar el script:
python llm.py

Estructura del proyecto
	•	netagent/
	•	llm.py               # Script base para pruebas con LLM
	•	consultor.py         # Script que conectará con FortiGate API
	•	.gitignore           # Ignorar carpetas del entorno virtual y cachés
	•	README.md            # Documentación del proyecto

Notas
	•	El entorno virtual puede usar la GPU si tienes una NVIDIA RTX y CUDA instalado correctamente.
	•	Los modelos grandes como LLaMA 2 7B pueden tardar bastante en descargarse (~13 GB).
	•	bitsandbytes permite cargar el modelo en 8-bit para ahorrar memoria.

Próximos pasos
	•	Añadir conexión real a FortiGate API desde consultor.py
	•	Integrar parsing JSON de respuestas y preprocesamiento
	•	Afinar el prompt y pipeline para respuestas contextualizadas