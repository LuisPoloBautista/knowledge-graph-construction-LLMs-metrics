import json
import nltk

# Lee el archivo JSON
with open("", 'r', encoding="utf-8") as f:
    data = json.load(f)


for entry in data:
    if 'head' in entry and isinstance(entry['head'], str):
        entry['head'] = entry['head'].lower()
    if 'head_type' in entry and isinstance(entry['head_type'], str):
        entry['head_type'] = entry['head_type'].lower()
    if 'relation' in entry and isinstance(entry['relation'], str):
        entry['relation'] = entry['relation'].lower()
    if 'tail' in entry and isinstance(entry['tail'], str):
        entry['tail'] = entry['tail'].lower()
    if 'tail_type' in entry and isinstance(entry['tail_type'], str):
        entry['tail_type'] = entry['tail_type'].lower()


from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


# Inicializar modelos
summarizer = pipeline("summarization", model="t5-large", tokenizer="t5-large")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def smart_truncate(text, max_tokens=6, max_chars=200):
    """
    Trunca el texto de manera inteligente, asegurando palabras completas.
    
    Args:
        text (str): Texto de entrada
        max_tokens (int): Número máximo de tokens
        max_chars (int): Número máximo de caracteres
    
    Returns:
        str: Texto truncado con palabras completas
    """
    # Tokenizar el texto
    tokens = nltk.word_tokenize(text)
    
    # Si el texto es corto, devolverlo completo
    if len(tokens) <= max_tokens:
        return text
    
    # Truncar a max_tokens palabras
    truncated_tokens = tokens[:max_tokens]
    truncated_text = ' '.join(truncated_tokens)
    
    # Asegurar que no exceda max_chars
    if len(truncated_text) > max_chars:
        truncated_text = truncated_text[:max_chars]
        # Asegurar que termina con una palabra completa
        truncated_text = truncated_text.rsplit(' ', 1)[0]
    
    return truncated_text

# Función para resumir y calcular coherencia semántica
def process_entry(entry, summarizer, semantic_model, max_tokens=6):
    keys_to_process = ['head', 'relation', 'tail']
    metrics = {}
    
    for key in keys_to_process:
        text = entry.get(key, "")
        if text and len(text.split()) > max_tokens:  # Procesar solo strings largos
            # Truncar texto de manera inteligente antes de resumir
            truncated_text = smart_truncate(text, max_tokens=max_tokens)
            
            # Resumir el texto
            try:
                summarized = summarizer(
                    truncated_text, 
                    max_length=15, 
                    min_length=6, 
                    do_sample=False
                )[0]['summary_text']
            except Exception as e:
                print(f"Error al resumir {key}: {e}")
                summarized = truncated_text
            
            entry[key] = summarized  # Actualizar con el resumen
            
            # Calcular coherencia semántica
            embeddings = semantic_model.encode([text, summarized])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            metrics[f'{key}_similarity'] = similarity
        else:
            metrics[f'{key}_similarity'] = None  # No aplica
    
    return metrics

# Procesar JSON
processed_data = []
all_metrics = []

for entry in data:
    if isinstance(entry, dict):  # Asegurar que cada entrada sea un diccionario
        metrics = process_entry(entry, summarizer, semantic_model)
        processed_data.append(entry)
        all_metrics.append(metrics)

# Exportar JSON modificado
output_file = ""
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=2)

# Exportar métricas
metrics_file = ""
with open(metrics_file, 'w', encoding='utf-8') as file:
    json.dump(all_metrics, file, ensure_ascii=False, indent=2)
