import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class JSONComparator:
    def __init__(self, archivos):
        """
        Inicializa el comparador con nombres de archivos personalizados
        
        Args:
            archivos (dict): Diccionario con nombres de modelos como clave y rutas de archivos como valor
        """
        self.archivos = archivos
        self.datos_json = {}
        self.nombres_modelos = list(archivos.keys())
        
    def cargar_json(self):
        """Carga los archivos JSON"""
        for modelo, archivo in self.archivos.items():
            with open(archivo, 'r', encoding='utf-8') as f:
                self.datos_json[modelo] = json.load(f)
    
    def obtener_elementos(self):
        """
        Obtiene elementos de cada archivo con un método de comparación más sofisticado
        
        Returns:
            dict: Conjunto de elementos para cada modelo
        """
        elementos_sets = {}
        for modelo, data in self.datos_json.items():
            elementos = set()
            for item in data:
                # Método de comparación con pesos diferentes
                elementos.add((
                    item.get('head', '').lower(),     # Normalizar a minúsculas
                    item.get('head_type', ''),
                    item.get('relation', '').lower(),
                    item.get('tail', '').lower(),     # Normalizar a minúsculas
                    item.get('tail_type', '')
                ))
            elementos_sets[modelo] = elementos
        return elementos_sets
    
    def medir_solapamiento(self, set1, set2):
        """
        Calcula el solapamiento con un método más preciso
        
        Args:
            set1 (set): Primer conjunto de elementos
            set2 (set): Segundo conjunto de elementos
        
        Returns:
            float: Porcentaje de solapamiento
        """
        # Usar Jaccard Index con normalización adicional
        interseccion = set1.intersection(set2)
        union = set1.union(set2)
        return (len(interseccion) / len(union)) * 100 if len(union) > 0 else 0
    
    def comparar_archivos(self):
        """
        Compara archivos y genera resultados detallados
        
        Returns:
            tuple: DataFrames de resultados y matrices de solapamiento
        """
        self.cargar_json()
        elementos_sets = self.obtener_elementos()
        
        # Preparar listas para resultados
        resultados = []
        matriz_solapamiento = []
        matriz_diferentes = []
        
        # Comparación completa, omitiendo comparaciones consigo mismo
        for i, modelo1 in enumerate(self.nombres_modelos):
            fila_solapamiento = []
            fila_diferentes = []
            for j, modelo2 in enumerate(self.nombres_modelos):
                if i != j:
                    solapamiento = self.medir_solapamiento(
                        elementos_sets[modelo1], 
                        elementos_sets[modelo2]
                    )
                    elementos_diferentes = len(
                        elementos_sets[modelo1].symmetric_difference(elementos_sets[modelo2])
                    )
                    total_elementos = len(
                        elementos_sets[modelo1].union(elementos_sets[modelo2])
                    )
                    porcentaje_diferentes = (elementos_diferentes / total_elementos) * 100
                    
                    resultados.append({
                        'Modelo 1': modelo1,
                        'Modelo 2': modelo2,
                        'Porcentaje Solapamiento': solapamiento,
                        'Porcentaje Diferentes': porcentaje_diferentes
                    })
                    
                    fila_solapamiento.append(solapamiento)
                    fila_diferentes.append(porcentaje_diferentes)
                else:
                    # Para la diagonal, ponemos 0
                    fila_solapamiento.append(0)
                    fila_diferentes.append(0)
            
            matriz_solapamiento.append(fila_solapamiento)
            matriz_diferentes.append(fila_diferentes)
        
        # Crear DataFrame de resultados
        df_resultados = pd.DataFrame(resultados)
        
        return df_resultados, np.array(matriz_solapamiento), np.array(matriz_diferentes)
    
    def visualizar_heatmap(self, matriz_solapamiento, matriz_diferentes):
        """
        Genera heatmaps de solapamiento y elementos diferentes
        
        Args:
            matriz_solapamiento (np.array): Matriz de solapamiento
            matriz_diferentes (np.array): Matriz de elementos diferentes
        """
        plt.figure(figsize=(16, 6))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(matriz_solapamiento, 
                    annot=True, 
                    cmap='YlGnBu', 
                    xticklabels=self.nombres_modelos, 
                    yticklabels=self.nombres_modelos,
                    fmt='.2f')
        plt.title('Porcentaje de Solapamiento')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(matriz_diferentes, 
                    annot=True, 
                    cmap='YlOrRd', 
                    xticklabels=self.nombres_modelos, 
                    yticklabels=self.nombres_modelos,
                    fmt='.2f')
        plt.title('Porcentaje de Elementos Diferentes')
        
        plt.tight_layout()
        plt.show()
    
    def ejecutar_analisis(self):
        """Método principal para ejecutar todo el análisis"""
        df_resultados, matriz_solapamiento, matriz_diferentes = self.comparar_archivos()
        
        # Imprimir tabla de resultados
        print("Resultados de Comparación:")
        print(df_resultados.to_string(index=False))
        
        # Generar visualización
        self.visualizar_heatmap(matriz_solapamiento, matriz_diferentes)

# Ejemplo de uso
if __name__ == "__main__":
    archivos = {
        "Gemma 2": "....json",
        "Llama 3.1": "....json", 
        "OLMO": "...json", 
        "GPT-4o": "...json"
    }
    
    comparador = JSONComparator(archivos)
    comparador.ejecutar_analisis()