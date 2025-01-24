import pandas as pd
import json
import ast
from collections import Counter

filename = ""

df = pd.read_csv(filename, encoding="latin9")


class RedundanciaTripletas:
    def __init__(self):
        """
        Inicializa el analizador de redundancia de tripletas
        """
        pass

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

    def tripleta_a_tupla_hashable(self, tripleta):
        """
        Convierte una tripleta a una tupla hashable

        :param tripleta: Diccionario de tripleta
        :return: Tupla hashable
        """
        # Convertir los valores de la tripleta a tuplas si son listas
        # para que sean hashables
        tripleta_hashable = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in tripleta.items()
        }

        # Ordenar los items del diccionario para asegurar consistencia
        return tuple(sorted(tripleta_hashable.items()))

    def calcular_redundancia(self, tripletas):
        """
        Calcula métricas de redundancia para un conjunto de tripletas

        :param tripletas: Lista de tripletas
        :return: Diccionario con métricas de redundancia
        """
        if not tripletas:
            return {
                'total_tripletas': 0,
                'tripletas_unicas': 0,
                'redundancia_ratio': 0,
                'tripletas_repetidas': {},
                'redundancia_porcentaje': 0
            }

        # Convertir tripletas a tuplas hashables para conteo
        tripletas_tuple = [self.tripleta_a_tupla_hashable(tripleta) for tripleta in tripletas]

        # Contar ocurrencias de cada tripleta
        contador_tripletas = Counter(tripletas_tuple)

        # Métricas de redundancia
        total_tripletas = len(tripletas)
        tripletas_unicas = len(set(tripletas_tuple))

        # Tripletas repetidas (convertir de vuelta a diccionario)
        tripletas_repetidas = {}
        for tripleta_tuple, count in contador_tripletas.items():
            if count > 1:
                # Convertir tupla de vuelta a diccionario
                tripleta_dict = dict(tripleta_tuple)
                tripletas_repetidas[json.dumps(tripleta_dict)] = count

        # Calcular ratio de redundancia
        redundancia_ratio = (total_tripletas - tripletas_unicas) / total_tripletas if total_tripletas > 0 else 0
        redundancia_porcentaje = redundancia_ratio * 100

        return {
            'total_tripletas': total_tripletas,
            'tripletas_unicas': tripletas_unicas,
            'redundancia_ratio': redundancia_ratio,
            'tripletas_repetidas': tripletas_repetidas,
            'redundancia_porcentaje': redundancia_porcentaje
        }

    def procesar_dataframe(self, df):
        """
        Procesa un dataframe completo para calcular redundancia de tripletas

        :param df: DataFrame con columna 'tripletas_respaldadas'
        :return: DataFrame con columnas de redundancia
        """
        # Convertir columna de tripletas a lista de diccionarios
        df['tripletas_parsed'] = df['tripletas_respaldadas'].apply(self.parse_tripletas)

        # Calcular redundancia para cada fila
        df['redundancia'] = df['tripletas_parsed'].apply(self.calcular_redundancia)

        # Descomponer métricas de redundancia
        df['total_tripletas'] = df['redundancia'].apply(lambda x: x['total_tripletas'])
        df['tripletas_unicas'] = df['redundancia'].apply(lambda x: x['tripletas_unicas'])
        df['redundancia_ratio'] = df['redundancia'].apply(lambda x: x['redundancia_ratio'])
        df['redundancia_porcentaje'] = df['redundancia'].apply(lambda x: x['redundancia_porcentaje'])
        df['tripletas_repetidas'] = df['redundancia'].apply(lambda x: x['tripletas_repetidas'])

        return df

# Ejemplo de uso
if __name__ == '__main__':
    # Cargar datos
    df = pd.read_csv(filename, encoding="latin9")

    # Inicializar analizador de redundancia
    analizador = RedundanciaTripletas()

    # Procesar dataframe
    df_con_redundancia = analizador.procesar_dataframe(df)

    # Mostrar estadísticas de redundancia
    print("\nEstadísticas de Redundancia:")
    print("Total de Tripletas:", df_con_redundancia['total_tripletas'].sum())
    print("Tripletas Únicas:", df_con_redundancia['tripletas_unicas'].sum())
    print("Ratio de Redundancia Promedio: {:.2%}".format(df_con_redundancia['redundancia_ratio'].mean()))
    print("Porcentaje de Redundancia Promedio: {:.2f}%".format(df_con_redundancia['redundancia_porcentaje'].mean()))

    # Mostrar ejemplos de tripletas repetidas
    print("\nEjemplos de Tripletas Repetidas:")
    ejemplos_repetidos = df_con_redundancia[df_con_redundancia['redundancia_porcentaje'] > 0]
    for idx, row in ejemplos_repetidos.iterrows():
        if row['tripletas_repetidas']:
            print(f"\nFila {idx}:")
            print(json.dumps(row['tripletas_repetidas'], indent=2))