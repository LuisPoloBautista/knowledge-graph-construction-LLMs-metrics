from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast  # pip install ai2-olmo
import torch
import pandas as pd
import gc

torch.random.manual_seed(0)
model = OLMoForCausalLM.from_pretrained(
    "allenai/OLMo-7B",
    #revision="step1000-tokens4B"
    device_map="cuda",
    torch_dtype="auto",
)

tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B")

# Cargar el archivo original en modo lectura
data = pd.read_csv('', encoding="latin9")
data.info()

# Seleccionar un rango específico para procesar
df = data.iloc[0:50]

# Inicializar el pipeline de generación de texto
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Parámetros para la generación
generation_args = {
    "max_new_tokens": 1400,
    "return_full_text": False,
    "temperature": 0.7,  # Aumentar la temperatura para variabilidad
    "do_sample": True,    # Permitir muestreo
}

# Set batch size
batch_size = 5  # Reducido para usar menos memoria

# Iterar sobre el DataFrame en lotes
for i in range(0, len(df), batch_size):
    batch = df[i: i + batch_size]
    total_rows = len(batch)
    
    for idx, row in batch.iterrows():
        text = row['texto_completo']  # Obtener el texto de la columna 'texto_completo'

        # Comprobar si el texto excede el límite del modelo
        input_tokens = tokenizer.encode(text)
        if len(input_tokens) > 4096:
            print(f"Texto en fila {idx} excede la longitud máxima de tokens. Se truncará.")
            text = tokenizer.decode(input_tokens[:4096])  # Truncar a 4096 tokens

        # Crear el mensaje para el modelo
        messages = [
            {"role": "system",
             "content": f"""A partir del siguiente texto relacionado con desastres naturales: {text},
                    realiza las siguientes tareas:

                    1. **Extraer entidades nombradas**: Identifica todas las entidades relevantes en el texto.
                    2. **Identificar relaciones**: Establece las relaciones entre las entidades extraídas.

                    **Requisitos de Salida**:
                    - Formato: Debe devolver un objeto JSON.
                    - Estructura del objeto: Cada entidad y su relación deben estar representadas con las siguientes claves:
                        - **head**: Entidad principal.
                        - **head_type**: Tipo de la entidad principal (e.g., 'persona', 'lugar', 'fecha').
                        - **relation**: Relación entre la entidad principal y la secundaria (e.g., 'ubicado en', 'causado por').
                        - **tail**: Entidad secundaria.
                        - **tail_type**: Tipo de la entidad secundaria (e.g., 'organización', 'infraestructura').

                    **Ejemplo de formato JSON (es solo es un ejemplo, no lo copies ni uses en producción)**:

                    [
                        {{
                            "head": "Entidad principal",
                            "head_type": "Tipo de la entidad principal",
                            "relation": "Relación entre la entidad principal y la secundaria",
                            "tail": "Entidad secundaria",
                            "tail_type": "Tipo de la entidad secundaria"
                        }},
                        {{
                             "head": "Entidad principal",
                            "head_type": "Tipo de la entidad principal",
                            "relation": "Relación entre la entidad principal y la secundaria",
                            "tail": "Entidad secundaria",
                            "tail_type": "Tipo de la entidad secundaria"
                        }},
                        {{
                             "head": "Entidad principal",
                            "head_type": "Tipo de la entidad principal",
                            "relation": "Relación entre la entidad principal y la secundaria",
                            "tail": "Entidad secundaria",
                            "tail_type": "Tipo de la entidad secundaria"
                        }}
                    ]
                    """}
        ]

        # Obtener la salida del modelo
        output = pipe(messages, **generation_args)

        # Agregar el resultado en una nueva columna 'tripletas_Phi'
        df.at[idx, 'tripletas_noe'] = output[0]["generated_text"]

        # Liberar memoria GPU
        del messages, output
        torch.cuda.empty_cache()  # Limpiar la caché de CUDA
        gc.collect()  # Liberar memoria no utilizada

        # Imprimir el progreso
        processed_rows = idx - (i - 1)
        print(f"Procesando fila {idx + 1}/{len(df)}: {processed_rows}/{total_rows} completadas.")

# Guardar el DataFrame procesado en un archivo
df.to_csv('', mode='w', header=True, index=False)