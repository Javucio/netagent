Nnetagent
Agente inteligente para operar dispositivos FortiGate mediante lenguaje natural.
Combina llamadas API a FortiGate con un LLM tuneado para entender y responder en “lenguaje Forti”, facilitando la gestión y análisis en tiempo real.

Requisitos
Ubuntu 20.04+ (se recomienda WSL2 en Windows)

Python 3.10+

GPU NVIDIA con CUDA 12.6 (opcional pero recomendado)

64 GB RAM (para modelos grandes)

Acceso a modelo LLM (p.ej. Llama 7B o superior) desde HuggingFace (requiere autorización)

Setup rápido
bash
Copy
Edit
# Instalar paquete para entornos virtuales si no está
sudo apt install python3-venv -y

# Crear y activar entorno virtual
python3 -m venv ./venv
source ./venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias base con soporte CUDA 12.6 (ajustar versión CUDA si es necesario)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Instalar HuggingFace Transformers y otras librerías
pip install transformers accelerate bitsandbytes requests

# (Opcional) Instalar cliente SSH o herramientas adicionales según necesidad
Uso básico
bash
Copy
Edit
# Ejecutar script consultor para llamadas FortiGate API
python consultor.py

# Ejecutar script para carga y prueba del LLM
python llm.py
Nota: Actualmente el acceso a modelos Llama 2 está restringido y requiere login y autorización en HuggingFace.

Estructura recomendada
graphql
Copy
Edit
netagent/
│
├── consultor.py           # Script base para llamadas API FortiGate
├── llm.py                 # Script para cargar y probar modelo LLM
├── README.md              # Este archivo de documentación
├── requirements.txt       # Dependencias para pip (generar con pip freeze > requirements.txt)
├── configs/               # Configs para modelos, tokens, y parámetros
│   └── fortigate_api.json # Configuración API FortiGate (endpoints, credenciales)
│
├── data/                  # Datos para entrenamiento/fine-tuning futuros
│   └── fortigate_logs/    # Logs y ejemplos de datos FortiGate
│
├── models/                # Modelos LLM descargados o checkpoints locales
│
├── utils/                 # Funciones auxiliares y librerías propias
│   └── api_helpers.py     # Funciones para manejo de API FortiGate
│   └── llm_helpers.py     # Funciones para interacción con LLM
│
└── tests/                 # Tests unitarios y de integración
    └── test_consultor.py