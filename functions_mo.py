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

"-------------------------------------------------------------------------------------------------"

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