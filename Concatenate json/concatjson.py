import pandas as pd
import csv
import json
import re

def extract_json_array(text):
    """
    Extrae específicamente arrays JSON que comienzan con [ y terminan con ],
    incluso si hay otro texto alrededor.
    """
    if not text or not isinstance(text, str):
        return None

    # Buscar el patrón más específico para arrays JSON
    pattern = r'\[\s*\{\s*\"[^\"]+\"\s*:\s*\"[^\"]+\"\s*,.*?\}\s*\]'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            json_data = json.loads(match)
            if isinstance(json_data, list) and len(json_data) > 0 and all(isinstance(item, dict) for item in json_data):
                return json_data
        except json.JSONDecodeError:
            continue

    return None

def process_csv_local(filepath):

    all_json_data = []
    processed_rows = 0
    json_found = 0

    try:
        # Leer el archivo CSV con pandas
        df = pd.read_csv(filepath, encoding='latin9')

        for index, row in df.iterrows():
            processed_rows += 1
            text = row.get('tripletas_respaldadas', "")

            if processed_rows <= 5:  # Mostrar las primeras 5 filas para debugging
                print(f"\n--- Fila {processed_rows} ---")
                print(f"Buscando JSON en: {text[:500]}...")  # Mostrar los primeros 500 caracteres

            json_data = extract_json_array(text)

            if json_data:
                json_found += 1
                print(f"¡JSON encontrado en fila {processed_rows}! Elementos: {len(json_data)}")
                all_json_data.extend(json_data)

        print(f"\n=== Resumen del procesamiento ===")
        print(f"Total de filas procesadas: {processed_rows}")
        print(f"Filas con JSON válido encontrado: {json_found}")
        print(f"Total de elementos JSON encontrados: {len(all_json_data)}")

        if all_json_data:
            output_filename = ''
            with open(output_filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(all_json_data, jsonfile, indent=2, ensure_ascii=False)

            print(f"\nArchivo {output_filename} creado y guardado localmente")
        else:
            print("\nNo se encontraron datos JSON válidos en la columna 'tripletas_respaldadas'")

    except Exception as e:
        print(f"Error procesando el archivo: {str(e)}")
        import traceback
        print(traceback.format_exc())

# Ejemplo de uso

filepath = ''
process_csv_local(filepath)
