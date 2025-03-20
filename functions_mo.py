import logging
import pandas as pd
from datetime import datetime, timedelta
from logging import Logger
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from mypy_boto3_dynamodb.client import DynamoDBClient
from mypy_boto3_s3.client import S3Client
from chalice import Chalice
from chalice import Response
from io import StringIO
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


"-reciclada------------------------------------------------------------------------------------------------------------------------------------------"

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

"-reciclada------------------------------------------------------------------------------------------------"

def group_by_month(df, df_2):
    try:
        # Log the start of the function
        logging.info("Inicia la función agrupando por mes.")

        # Convert column names to datetime format
        df.columns = pd.to_datetime(df.columns, errors='coerce')

        # Check if any columns could not be converted
        if df.columns.isna().any():
            logging.error("Algunas columnas no pudieron convertirse al formato.")
            raise ValueError("Algunas columnas no pudieron convertirse al formato fecha. por favor revisar.")

        # Group columns by month and sum values
        logging.info("agrupando las columnas por mes y sumandolas.")
        grouped_df = df.groupby(df.columns.to_period('M'), axis=1).sum()

        selected_columns = df_2.iloc[:, 1:3]
        grouped_df_final = pd.concat([selected_columns, grouped_df], axis=1)

        # Log the successful completion
        logging.info("Se realizó la agrupación con éxito.")
        return grouped_df_final

    except Exception as e:
        # Log the exception with an error level
        logging.error(f"An error occurred: {e}")
        raise
"--reciclada----------------------------------------------------------------------------------------------------------"

import pandas as pd
import logging

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def multiply_price(df1, df2, col_1, col_2, col_3, months):
    """
    Multiplica los valores mensuales de df1 por la columna 'TARIFA' de df2, asociando por 'FINCA'.
    
    Parameters:
    df1 (DataFrame): DataFrame con los valores mensuales.
    df2 (DataFrame): DataFrame con la columna 'TARIFA' asociada a cada 'FINCA'.
    months (list): Lista de columnas que representan los meses a multiplicar.

    Returns:
    DataFrame: DataFrame con los valores multiplicados.
    """
    try:
        logger.info("Inicia la función para multiplicar los DataFrames.")

        # Convertir los nombres de los meses a strings si son Period
        months = [str(m) for m in months]
        df1.columns = df1.columns.astype(str)  # Convertir nombres de columnas a string
        df2.columns = df2.columns.astype(str)

        # Merge de ambos DataFrames en base a la columna 'FINCA'
        logger.info("Realizando el merge sobre la columna FINCA")
        df_merged = pd.merge(df1, df2[[col_1, col_2, col_3]], on=col_1, how='left')


        # Verificar que la columna existe después del merge
        if f"{col_2}" not in df_merged.columns:
            raise KeyError(f"La columna {col_2} no se encuentra en el DataFrame después del merge.")

        # Multiplicación de cada mes por la TARIFA
        for month in months:
            try:
                df_merged[month] = df_merged[month].astype(float) * df_merged[col_2]
            except KeyError as e:
                logger.error(f"Error en la columna {month}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error inesperado al procesar la columna {month}: {e}")
                raise

        # Log de las columnas después de la multiplicación
        logger.info(f"Columnas posteriores a la multiplicación: {df_merged.columns.tolist()}")

        # Selección de columnas finales sin la columna 'TARIFA'
        result_df = df_merged[['FINCA', 'PROMEDIADO'] + months]

        logger.info("Proceso completado exitosamente.")
        return result_df

    except KeyError as e:
        logger.error(f"KeyError ocurrido: {e}")
        raise
    except pd.errors.MergeError as e:
        logger.error(f"Error durante el merge: {e}")
        raise
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado: {e}")
        raise

"---------------------------------------------------------------------------------------------------------"
def multiply_by_month_promediado(df1, df2, months):
    try:
        logger.info("Inicia la función para multiplicar ambos dataframes.")

        # Limpiar espacios en los nombres de columnas
        df1.columns = df1.columns.str.strip().astype(str)
        df2.columns = df2.columns.str.strip().astype(str)

        # Asegurar que las columnas de meses sean de tipo string
        months = [str(month) for month in months]

        # Asegurar que los valores en meses sean numéricos
        for month in months:
            df1[month] = pd.to_numeric(df1[month], errors='coerce')
            df2[month] = pd.to_numeric(df2[month], errors='coerce')

        # Merge basado en 'PROMEDIADO'
        logger.info("Se realiza el merge sobre la columna PROMEDIADO.")
        df_merged = pd.merge(df1, df2, on='PROMEDIADO', how='left', suffixes=('_df1', '_df2'))

        # Verificar columnas después del merge
        logger.info(f"Columnas después del merge: {df_merged.columns.tolist()}")

        # Multiplicar cada mes
        for month in months:
            col_df1 = f'{month}_df1'
            col_df2 = f'{month}_df2'

            if col_df1 in df_merged.columns and col_df2 in df_merged.columns:
                df_merged[col_df1] = pd.to_numeric(df_merged[col_df1], errors='coerce')
                df_merged[col_df2] = pd.to_numeric(df_merged[col_df2], errors='coerce')

                df_merged[month] = df_merged[col_df1] * df_merged[col_df2]
            else:
                logger.warning(f"No se encontraron columnas {col_df1} o {col_df2} en el DataFrame.")

        # Verificar si 'FINCA' está presente
        if 'FINCA' in df_merged.columns:
            result_df = df_merged[['FINCA'] + months]
        else:
            logger.warning("La columna 'FINCA' no está en el DataFrame después del merge.")
            result_df = df_merged[months]

        logger.info("Proceso completado exitosamente")
        return result_df

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

"-----------------------------------------------------------------------------------------------------------"
def object_to_dataframe(
    s3_client: S3Client,
    bucket_name: str,
    folder_name: str,
    partition: str,
    file_name: str,
    encoded: str = "utf-8",
    sep: str = ";",
) -> pd.DataFrame:
    """
    The function `object_to_dataframe` reads a CSV file from an S3 bucket using an S3 client, decodes
    the data, and returns it as a Pandas DataFrame, with an option to specify the separator.

    :param s3_client: S3Client object from the boto3 library, used for interacting with Amazon S3
    :type s3_client: S3Client
    :param bucket_name: The `bucket_name` parameter in the `object_to_dataframe` function refers to the
    name of the Amazon S3 bucket where the object is stored. This is the bucket from which the function
    will retrieve the object specified by the `key` parameter
    :type bucket_name: str
    :param key: The `key` parameter in the `object_to_dataframe` function is a string that represents
    the key of the object stored in an S3 bucket. It is used to specify the specific object that you
    want to retrieve and convert into a pandas DataFrame
    :type key: str
    :param encoded: The `encoded` parameter in the `object_to_dataframe` function specifies the encoding
    format used to decode the data read from the S3 object before converting it into a DataFrame. In
    this case, the default encoding format is set to "utf-8", but you can specify a different encoding
    format if, defaults to utf-8
    :type encoded: str (optional)
    :param sep: The `sep` parameter in the `object_to_dataframe` function is used to specify the
    delimiter that separates columns in the CSV file being read. By default, the delimiter is set to
    `;`. However, if the `key` parameter matches either "ER_SIMULADO/EXCEDENTE.csv, defaults to ;
    :type sep: str (optional)
    :return: A pandas DataFrame containing the data from the specified object in the S3 bucket.
    """

    # if file_name in (
    #     "EXCEDENTE.csv",
    #     "platano.csv",
    # ):
    #     sep = ","

    # if file_name == "EXCEDENTE.csv":
    #     encoded = "latin-1"

    key = f"{folder_name}/{partition}/{file_name}"

    obj = s3_client.get_object(Bucket=bucket_name, Key=key)
    data = obj["Body"].read().decode(encoded)
    df = pd.read_csv(StringIO(data), sep=sep)
    return df


"-------------------------------------------------------------------------------------------------"

def get_item_from_dynamodb_global(table_name: str, partition_key_value: str, logger: Logger, dynamodb_client):
    """
    Retrieves an item from a DynamoDB table without requiring a sort key if not needed.

    :param table_name: Name of the DynamoDB table.
    :param partition_key_value: Value of the partition key.
    :param logger: Logger instance for logging.
    :param dynamodb_client: Boto3 DynamoDB resource.
    :return: The retrieved item or None if not found.
    """
    try:
        # Access the table
        table = dynamodb_client.Table(table_name)

        # Get table key schema
        table_info = dynamodb_client.meta.client.describe_table(TableName=table_name)
        key_schema = table_info["Table"]["KeySchema"]

        # Identify partition key and sort key (if exists)
        key_dict = {key["AttributeName"]: partition_key_value for key in key_schema if key["KeyType"] == "HASH"}

        # If there is a sort key, add it (this assumes a default or required value)
        sort_keys = [key["AttributeName"] for key in key_schema if key["KeyType"] == "RANGE"]
        if sort_keys:
            logger.warning(f"Table {table_name} requires a sort key: {sort_keys}. Consider adjusting your query.")

        # Get the item
        response = table.get_item(Key=key_dict)

        # Check if the item exists in the response
        if "Item" in response:
            return response["Item"]
        else:
            logger.error(f"Item not found in table {table_name}.")
            return None

    except NoCredentialsError:
        logger.exception("AWS credentials not found.")
        raise Exception("AWS credentials not found.")
    except PartialCredentialsError:
        logger.exception("Incomplete AWS credentials configuration.")
        raise Exception("Incomplete AWS credentials configuration.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise Exception(f"An error occurred: {e}")

"-------------------------------------------------------------------------------------------------"

import pandas as pd

def multiply_p_social(df1, df2, df3, col_1, col_2, col_3, months):
    try:
        logger.info("Inicia la función sumar los dataframes de labor y promediados.")

        # Convertir los nombres de los meses a strings si son Period
        months = [str(m) for m in months]
        df1.columns = df1.columns.astype(str)
        df2.columns = df2.columns.astype(str)

        # Asegurar que los valores sean numéricos
        for col in months:
            df1[col] = pd.to_numeric(df1[col], errors='coerce')
            df2[col] = pd.to_numeric(df2[col], errors='coerce')

        # Sumar los valores por cada mes
        df_sum = df1[['FINCA']].copy()  # Crear df_sum con la columna FINCA
        for col in months:
            df_sum[col] = df1[col] + df2[col]

        logger.info("Suma realizada con éxito.")
       

        # Merge de ambos DataFrames en base a la columna 'FINCA'
        logger.info("Realizando el merge sobre la columna FINCA y PRESTACIONES")
        df_merged = pd.merge(df_sum, df3[[col_1, col_2, col_3]], on=col_1, how='left')

        # Verificar que la columna existe después del merge
        if col_2 not in df_merged.columns:
            raise KeyError(f"La columna {col_2} no se encuentra en el DataFrame después del merge.")

        # Multiplicación de cada mes por la TARIFA
        for month in months:
            try:
                df_merged[month] = df_merged[month].astype(float) * df_merged[col_2]
            except KeyError as e:
                logger.error(f"Error en la columna {month}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error inesperado al procesar la columna {month}: {e}")
                raise

        logger.info(f"Columnas posteriores a la multiplicación: {df_merged.columns.tolist()}")

        # Selección de columnas finales sin la columna 'TARIFA'
        result_df = df_merged[[col_1, col_3] + months]

        logger.info("Proceso completado exitosamente.")
        return result_df

    except KeyError as e:
        logger.error(f"KeyError ocurrido: {e}")
        raise
    except pd.errors.MergeError as e:
        logger.error(f"Error durante el merge: {e}")
        raise
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado: {e}")
        raise

"-------------------------------------------------------------------------------------------------"
def total_cost(df1, df2, df3, col_1, col_2, months):
    try:
        logger.info("Inicia la función sumar los dataframes de labor y promediados.")

        # Convertir los nombres de los meses a strings si son Period
        months = [str(m) for m in months]
        df1.columns = df1.columns.astype(str)
        df2.columns = df2.columns.astype(str)

        # Asegurar que los valores sean numéricos
        for col in months:
            df1[col] = pd.to_numeric(df1[col], errors='coerce')
            df2[col] = pd.to_numeric(df2[col], errors='coerce')

        # Sumar los valores por cada mes
        df_sum = df1[['FINCA']].copy()  # Crear df_sum con la columna FINCA
        for col in months:
            df_sum[col] = df1[col] + df2[col]+ df3[col]

        logger.info("Suma realizada con éxito.")

        # Merge de ambos DataFrames en base a la columna 'FINCA'
        logger.info("Realizando el merge sobre la columna FINCA y PRESTACIONES")
        df_merged = pd.merge(df_sum, df3[[col_1, col_2]], on=col_1, how='left')

        # Verificar que la columna existe después del merge
        if col_2 not in df_merged.columns:
            raise KeyError(f"La columna {col_2} no se encuentra en el DataFrame después del merge.")

        
        # Selección de columnas finales sin la columna 'TARIFA'
        result_df = df_merged[[col_1, col_2] + months]

        logger.info("Proceso completado exitosamente.")
        return result_df

    except KeyError as e:
        logger.error(f"KeyError ocurrido: {e}")
        raise
    except pd.errors.MergeError as e:
        logger.error(f"Error durante el merge: {e}")
        raise
    except Exception as e:
        logger.error(f"Ocurrió un error inesperado: {e}")
        raise


"-------------------------------------------------------------------------------------------------"


def get_item_from_dynamodb(table_name: str, key: dict, logger: Logger, dynamodb_client):
    """
    Retrieves an item from a DynamoDB table.

    :param table_name: Name of the DynamoDB table.
    :param key: Dictionary representing the key of the item to retrieve.
                 Example: {'partition_key': 'value', 'sort_key': 'value'}
    :return: The retrieved item or None if not found.
    """
    try:
        # Access the table
        table = dynamodb_client.Table(table_name)

        # Get the item
        response = table.get_item(Key=key)

        # Check if the item exists in the response
        if "Item" in response:
            return response["Item"]
        else:
            logger.error(f"Item not found in table {table_name}.")
            Exception(f"Item not found in table {table_name}.")

    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        Exception("AWS credentials not found.")
    except PartialCredentialsError:
        logger.error("Incomplete AWS credentials configuration.")
        Exception("Incomplete AWS credentials configuration.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        Exception(f"An error occurred: {e}")  


"-------------------------------------------------------------------------------------------------"

def get_ssm_parameter(parameter_name, logger: Logger, ssm_client, with_decryption=True):
    """
    Retrieve a parameter from AWS SSM Parameter Store.

    :param parameter_name: The name of the SSM parameter to retrieve.
    :param with_decryption: Boolean indicating if the parameter should be decrypted (for SecureString parameters).
    :param region_name: AWS region where the SSM parameter store is located.
    :return: The value of the SSM parameter.
    """
    try:
        # Retrieve the parameter
        response = ssm_client.get_parameter(
            Name=parameter_name, WithDecryption=with_decryption
        )
        logger.info(f"Parameter {parameter_name} retrieved successfully.")
        return response["Parameter"]["Value"]

    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        Exception("AWS credentials not found.")
    except PartialCredentialsError:
        logger.error("Incomplete AWS credentials configuration.")
        Exception("Incomplete AWS credentials configuration.")
    except ssm_client.exceptions.ParameterNotFound:
        logger.error(f"Parameter {parameter_name} not found.")
        Exception(f"Parameter {parameter_name} not found.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        Exception(f"An error occurred: {str(e)}")