from googletrans import Translator
import json

# Inicializa el traductor de googletrans
translator = Translator()

# Claves que serán traducidas
TRANSLATABLE_KEYS = ['head', 'head_type', 'relation', 'tail', 'tail_type']

# Función para traducir un texto de inglés a español
def translate_text(text):
    if text:  # Verifica que el texto no esté vacío
        try:
            # Realiza la traducción solo si el texto es inglés
            translation = translator.translate(text, src='en', dest='es').text
            return translation
        except Exception as e:
            print(f"Error de traducción para el texto '{text}': {e}")
            return text  # Devuelve el texto original en caso de error
    else:
        return ''

# Nombre del archivo JSON de entrada y salida

filename=""

output_filename = ''

# Carga el archivo JSON de entrada
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Itera sobre cada entrada de la lista
for entry in data:
    if isinstance(entry, dict):
        # Traduce las claves especificadas si existen en la entrada
        for key in TRANSLATABLE_KEYS:
            if key in entry:
                entry[key] = translate_text(entry[key])

# Guarda el archivo JSON modificado
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Archivo '{output_filename}' actualizado correctamente con las traducciones.")