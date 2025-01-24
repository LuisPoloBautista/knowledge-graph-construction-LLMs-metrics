import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import ast


df = pd.read_csv("", encoding="latin9")


class ContextualRelevanceCalculator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Inicializa el calculador de relevancia contextual

        :param model_name: Nombre del modelo de embeddings a utilizar
        """
        self.embedding_model = SentenceTransformer(model_name)

    def extract_tripleta_text(self, tripleta):
        """
        Extrae texto representativo de una tripleta

        :param tripleta: Diccionario de tripleta
        :return: Cadena de texto representativa
        """
        return f"{tripleta.get('head', '')} {tripleta.get('relation', '')} {tripleta.get('tail', '')}"

    def calcular_relevancia_contextual(self, texto_completo, tripletas):
        """
        Calcula la relevancia contextual de tripletas

        :param texto_completo: Texto original de la noticia
        :param tripletas: Lista de tripletas en formato JSON
        :return: Lista de relevancias contextuales o lista vacía
        """
        # Manejar caso de lista vacía
        if not tripletas:
            return []

        # Embedding del texto completo
        embedding_texto = self.embedding_model.encode([texto_completo])[0]

        # Embeddings de las tripletas
        tripletas_textos = [self.extract_tripleta_text(t) for t in tripletas]

        # Manejar caso de tripletas vacías
        if not tripletas_textos:
            return []

        embeddings_tripletas = self.embedding_model.encode(tripletas_textos)

        # Calcular similitud de coseno
        relevancias = cosine_similarity([embedding_texto], embeddings_tripletas)[0]

        return relevancias.tolist()

    def parse_tripletas(self, tripletas_str):
        """
        Parsea diferentes formatos de tripletas

        :param tripletas_str: Cadena de tripletas
        :return: Lista de tripletas
        """
        # Manejar caso de cadena vacía o None
        if not tripletas_str or pd.isna(tripletas_str):
            return []

        try:
            # Intentar parsear como JSON
            return json.loads(tripletas_str.replace("'", '"'))
        except (json.JSONDecodeError, TypeError):
            try:
                # Intentar parsear como literal de Python
                return ast.literal_eval(tripletas_str)
            except (ValueError, SyntaxError):
                print(f"No se pudo parsear: {tripletas_str}")
                return []

    def procesar_dataframe(self, df):
        """
        Procesa un dataframe completo para calcular relevancia contextual

        :param df: DataFrame con columnas 'texto_completo' y 'tripletas_respaldadas'
        :return: DataFrame con columnas adicionales de relevancia
        """
        # Convertir columna de tripletas a lista de diccionarios
        df['tripletas_parsed'] = df['tripletas_respaldadas'].apply(self.parse_tripletas)

        # Calcular relevancia para cada fila
        df['relevancia_contextual'] = df.apply(
            lambda row: self.calcular_relevancia_contextual(
                row['texto_completo'],
                row['tripletas_parsed']
            ),
            axis=1
        )

        # Calcular relevancia promedio por fila
        df['relevancia_promedio'] = df['relevancia_contextual'].apply(
            lambda x: np.mean(x) if x else 0
        )

        return df

# Ejemplo de uso
if __name__ == '__main__':

    # Cargar datos
    df = pd.read_csv("", encoding="latin9")

    # Inicializar calculador
    calculador = ContextualRelevanceCalculator()

    # Procesar dataframe
    df_con_relevancia = calculador.procesar_dataframe(df)


    # Mostrar estadísticas de relevancia
    print("Relevancia Promedio:")
    print(df_con_relevancia['relevancia_promedio'].describe())