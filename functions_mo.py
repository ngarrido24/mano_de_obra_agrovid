import logging
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def final_result(dataset, year, number):
    try:
        # Función interna para obtener el primer día de la semana dado el número de semana y año
        def obtain_first_day_ok_week(week_number, year):
            try:
                # Calcular el primer día de la segunda semana del año (7 de enero como referencia)
                first_day_year = datetime(year, 1, 7)
                day = (week_number - 2) * 7
                first_day = first_day_year + timedelta(day + first_day_year.weekday())
                
                return first_day
            except Exception as e:
                logger.error(f"Error al calcular la fecha para la semana {week_number}: {e}")
                raise
        
        # Obtener una lista con el número de semana en el archivo de volumen
        weeks = dataset.columns[number:]  # Asume que las columnas relevantes empiezan desde la columna 3
        week_numbers = [int(i.split()[-1]) for i in weeks]
      
        # Convertir el número de semana en fecha
        dates = [obtain_first_day_ok_week(i, year) for i in week_numbers]
        logger.info(f"Fechas calculadas para cada semana: {dates}.")
        
        # Crear el mapeo de semanas a fechas
        mapping = dict(zip(weeks, dates))
        
        # Renombrar las columnas del dataset basado en el mapeo
        dataset.rename(mapping, axis=1, inplace=True)
        logger.info(f"Columnas renombradas exitosamente en el dataset.")
        
        return dataset

    except Exception as e:
        logger.error(f"Error en el proceso de transformación del dataset: {e}")
        raise

"-------------------------------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------------------------------------------------------------------------------------------------------"

def calculate_volume_distribution(volum_file_emb_subset_def, volum_distribution_subset_def, volum_file_emb_transform_def, new_columns_def):
    try:
        logger.info("Iniciando el cálculo de la matriz de distribución de volumen.")

        final_data = []

        for month in volum_distribution_subset_def.index:
            logger.info(f"Procesando el mes: {month}")
            monthly_result = []
            for i in range(46):  # 0 a 44 son 45 elementos
                result = (volum_file_emb_subset_def.iloc[i, :] * volum_distribution_subset_def.iloc[month, :]).sum()
                monthly_result.append(result)
            final_data.append(monthly_result)

        final_data_df = pd.DataFrame(final_data)
        logger.info("Matriz de resultados generada.")

        final_data_df_trans = final_data_df.transpose()
        final_data_df_trans.columns = new_columns_def
        logger.info("Matriz de resultados transpuesta y columnas renombradas.")

        volum_concat = volum_file_emb_transform_def[['FINCA', 'CONCEPTO']].reset_index(drop=True)
        volum_data_emb = pd.concat([volum_concat, final_data_df_trans], axis=1)
        logger.info("DataFrame final concatenado con los datos de volumen transformados.")

        return volum_data_emb

    except Exception as e:
        logger.error(f"Error durante el cálculo de la distribución de volumen: {e}")
        raise