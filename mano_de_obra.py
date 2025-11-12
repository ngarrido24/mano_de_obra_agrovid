#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import boto3
import logging
from io import StringIO
from dotenv import load_dotenv
from pathlib import Path
import os
import polars as pl
from logging import Logger
import json
from chalice import Response
from datetime import datetime, timedelta
from unidecode import unidecode
import unicodedata
import warnings
import gc
import unicodedata
import re
from functions_mo import (
calculate_volume_distribution_factor,
calculate_volume_distribution_blocks,
get_item_from_dynamodb, 
get_item_from_dynamodb_global, 
get_ssm_parameter,
object_to_dataframe,
group_by_month,
multiply_price,
multiply_by_month_promediado,
multiply_p_social,
total_cost,
total_cost_block,
calculate_promediados_factor,
group_by_type,
group_by_type_sum,
merge_factor_by_id,
multipy_factor_combined,
multiply_p_social_block,
multipy_factor_combined_100021,
standardize_column_dates,
vlookup_function,
social_p_parcela,
total_cost_parcela,
labores_ciclicas, 
farm_order_process,
reparar_codificacion, 
calculate_volume_distribution_blocks_ciclics,
multiply_months_by_price_ciclics,
ciclics_labor_calculation,
multiply_by_month_promediado_ciclics, 
multiply_p_social_block_ciclics, 
quantity_fijas,
quantity_other_labors,
quantity_other_labors_800008, 
build_factor_quantity,
farm_order_process_concat,
column_replacement,
sum_by_months_parcela,
reorder_output_materials,
calculate_volume_distribution_gastos,
parametro_1, parametro_2, parametro_3,
parametro_4, parametro_5, parametro_6, 
parametro_6, parametro_8, parametro_9, 
parametro_10, parametro_11
) 


# In[2]:


#set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


# In[3]:


env_path = Path('C:/Users/Nata_/Documents/Etapa_1_proyecto/Simulador_mano_de_obra/mano_de_obra_agrovid/var.env')

load_dotenv(dotenv_path=env_path)
AWS_PROFILE = os.environ["aws_profile"]
SSM_PARAMETER_BUCKET_INPUT_FILES_NAME = os.environ['SSM_PARAMETER_BUCKET_INPUT_FILES_NAME']
SSM_PARAMETER_BUCKET_OUTPUT_FILES_NAME = os.environ['SSM_PARAMETER_BUCKET_OUTPUT_FILES_NAME']
REGION = os.environ['REGION']
SSM_PARAMETER_DYNAMODB_MODULE_CONFIG_NAME = os.environ['SSM_PARAMETER_DYNAMODB_MODULE_CONFIG_NAME']
SSM_PARAMETER_DYNAMODB_GLOBAL_MODULE_CONFIG_NAME = os.environ['SSM_PARAMETER_DYNAMODB_GLOBAL_MODULE_CONFIG_NAME']


# ### Función para despliegue

# In[5]:


def materiales(event, context):
    modulo: str = "modulo_mano_de_obra"
    modulo_revenue: str = "modulo_ingresos"
    PROCESS_TYPE = "er_simulado"
    modulo_global_materiales = "nombre_finca"
    file_revenue: str = "volumen.csv"
    body = event
    # # 4. Body (usually JSON, but can be other formats)
    # if event.get("body"):
    #     body = event["body"]
    #     # Assuming the body is JSON
    #     body = json.loads(body)
    #     print("body_json", body)
    #     print("type body_json", type(body))
    
    session = boto3.Session(profile_name=AWS_PROFILE)

    s3_client = session.client('s3', region_name=REGION)
    ssm_client = session.client("ssm", region_name=REGION)
    dynamodb_client = boto3.resource("dynamodb", region_name=REGION)

    bucket_input_files = get_ssm_parameter(
        parameter_name=SSM_PARAMETER_BUCKET_INPUT_FILES_NAME,
        logger=logger,
        ssm_client=ssm_client,
        with_decryption=True,
    )
    bucket_output_files = get_ssm_parameter(
        parameter_name=SSM_PARAMETER_BUCKET_OUTPUT_FILES_NAME,
        logger=logger,
        ssm_client=ssm_client,
        with_decryption=True,
    )
    dynamodb_table_name_module_config = get_ssm_parameter(
        parameter_name=SSM_PARAMETER_DYNAMODB_MODULE_CONFIG_NAME,
        logger=logger,
        ssm_client=ssm_client,
        with_decryption=True,
    )

    dynamodb_table_name_global_module_config = get_ssm_parameter(
        parameter_name=SSM_PARAMETER_DYNAMODB_GLOBAL_MODULE_CONFIG_NAME,
        logger=logger,
        ssm_client=ssm_client,
        with_decryption=True,
    )
    logger.info(f"{dynamodb_table_name_module_config}")

    module_config = get_item_from_dynamodb(
        table_name=dynamodb_table_name_module_config,
        key={"module": modulo, "process_type": PROCESS_TYPE},
        logger=logger,
        dynamodb_client=dynamodb_client,
    )

    global_config_materiales = get_item_from_dynamodb_global(
        table_name=dynamodb_table_name_global_module_config,
        partition_key_value=modulo_global_materiales,
        logger=logger,
        dynamodb_client=dynamodb_client,
   )
    
    
    input_files_names: dict = module_config["input_file_names_mano_de_obra"]
    farm_order: dict = global_config_materiales["fincas"]

    # Descarga el archivo CSV desde S3
    
    map_with_dfs = {}
    current_date = datetime.now().strftime("%Y-%m-%d")
    partition = body.get("partition", current_date)

    volum_file = object_to_dataframe(
        s3_client=s3_client,
        bucket_name=bucket_input_files,
        folder_name=f"{PROCESS_TYPE}/{modulo_revenue}",
        partition=partition,
        file_name=file_revenue,
    )

    try:
        for key, file in input_files_names.items():
            df = object_to_dataframe(
                s3_client=s3_client,
                bucket_name=bucket_input_files,
                folder_name=f"{PROCESS_TYPE}/{modulo}",
                partition=partition,
                file_name=file,
            )
            logger.info(f"file_key: {file} read successfully")
            map_with_dfs[key] = df
        volum_distribution = map_with_dfs["volum_distribution"]
    except Exception as e:
        logger.error(
            f"Error reding file {file} from {bucket_input_files}: error detail:{e}"
        )
        return Response(
            status_code=500,
            body=json.dumps(
                {
                    "message": f"Error reding file {file} from {bucket_input_files}: error detail:{e}"
                }
            ),
        )


# ### Importar los archivos

# In[6]:


modulo: str = "modulo_mano_de_obra"
PROCESS_TYPE = "er_simulado"
dynamo_global_materiales = "nombre_finca"
modulo_global: str = "archivos_comunes_global"
output_files: str = "salida_materiales"

   
session = boto3.Session(profile_name=AWS_PROFILE)

s3_client = session.client("s3", region_name=REGION)
ssm_client = session.client("ssm", region_name=REGION)
dynamodb_client = session.resource("dynamodb", region_name=REGION)

bucket_input_files = get_ssm_parameter(
        parameter_name=SSM_PARAMETER_BUCKET_INPUT_FILES_NAME,
        logger=logger,
        ssm_client=ssm_client,
        with_decryption=True,
    )

bucket_output_files = get_ssm_parameter(
        parameter_name=SSM_PARAMETER_BUCKET_OUTPUT_FILES_NAME,
        logger=logger,
        ssm_client=ssm_client,
        with_decryption=True,
    )
dynamodb_table_name_module_config = get_ssm_parameter(
        parameter_name=SSM_PARAMETER_DYNAMODB_MODULE_CONFIG_NAME,
        logger=logger,
        ssm_client=ssm_client,
        with_decryption=True,
    )

dynamodb_table_name_global_module_config = get_ssm_parameter(
        parameter_name=SSM_PARAMETER_DYNAMODB_GLOBAL_MODULE_CONFIG_NAME,
        ssm_client=ssm_client,
        logger=logger,
        with_decryption=True
    ) 
    
    
logger.info(f"{dynamodb_table_name_module_config}")

module_config = get_item_from_dynamodb(
        table_name=dynamodb_table_name_module_config,
        key={"module": modulo, "process_type": PROCESS_TYPE},
        logger=logger,
        dynamodb_client=dynamodb_client,
    )
module_config_output = get_item_from_dynamodb(
        table_name=dynamodb_table_name_module_config,
        key={"module": output_files, "process_type": PROCESS_TYPE},
        logger=logger,
        dynamodb_client=dynamodb_client,
    )

module_global_config = get_item_from_dynamodb(
        table_name=dynamodb_table_name_module_config,
        key={"module": modulo_global, "process_type": PROCESS_TYPE},
        logger=logger,
        dynamodb_client=dynamodb_client,)


global_config_materiales = get_item_from_dynamodb_global(
        table_name=dynamodb_table_name_global_module_config,
        partition_key_value=dynamo_global_materiales,
        logger=logger,
        dynamodb_client=dynamodb_client,
   )
 

input_files_names: dict = module_config["input_file_names_mo"]
input_global_file_names: dict = module_global_config["input_file_names_global"]
farm_order: dict = global_config_materiales["fincas"]
output_file_names_mat: dict = module_config_output["output_file_names_mat"]

    # Descarga el archivo CSV desde S3
map_with_dfs = {}
current_date = datetime.now().strftime("%Y-%m-%d")
partition = "2025-02-19"

#extraer los archivos del modulo global
try:
    for key, file in input_global_file_names.items():
        df = object_to_dataframe(
            s3_client=s3_client,
            bucket_name=bucket_input_files,
            folder_name=f"{PROCESS_TYPE}/{modulo_global}",
            partition=partition,
            file_name=file,
        )

        logger.info(f"file_key: {file} read successfully")
        map_with_dfs[key] = df
    volum_distribution = map_with_dfs["volum_distribution"]
    volum_file = map_with_dfs["volum_file"]
    sku = map_with_dfs["sku"]
    
except Exception as e:
    logger.error(
            f"Error reding file {file} from {bucket_input_files}: error detail:{e}"
    )
     

#extraer los archivos del modulo particular
try:
    for key, file in input_files_names.items():
        df = object_to_dataframe(
            s3_client=s3_client,
            bucket_name=bucket_input_files,
            folder_name=f"{PROCESS_TYPE}/{modulo}",
            partition=partition,
            file_name=file,
        )

        logger.info(f"file_key: {file} read successfully")
        map_with_dfs[key] = df
    cut_emp          = map_with_dfs["cut_emp"]
    promediado       = map_with_dfs["promediado"]
    revenue_sku      = map_with_dfs["revenue_sku"]
    desect_ha        = map_with_dfs["desect_ha"]
    parcela_mo       = map_with_dfs["parcela_mo"]
    esquema_mo       = map_with_dfs["esquema_mo"]
    curvas_c         = map_with_dfs["curvas_c"]
    ciclicas         = map_with_dfs["ciclicas"]
    f_labor          = map_with_dfs["f_labor"]
    other_labors     = map_with_dfs["other_labors"]
    curvas_c_mensual = map_with_dfs["curvas_c_mensual"]
    ajustes_mo       = map_with_dfs["ajustes_mo"]
    fuerza_laboral   = map_with_dfs["fuerza_laboral"]
    gastos_personal  = map_with_dfs["gastos_personal"]
    factor_gastos    = map_with_dfs["factor_gastos"]

except Exception as e:
    logger.error(
            f"Error reding file {file} from {bucket_input_files}: error detail:{e}"
    )

#extraer los archivos de la salida del modulo de materiales 
try:
    for key, file in output_file_names_mat.items():
        df = object_to_dataframe(
            s3_client=s3_client,
            bucket_name=bucket_output_files,
            folder_name=f"{PROCESS_TYPE}/{output_files}",
            partition=partition,
            file_name=file,
        )

        logger.info(f"file_key: {file} read successfully")
        map_with_dfs[key] = df
    salida_fertilizante   = map_with_dfs["salida_fertilizante"]
  
    
except Exception as e:
    logger.error(
            f"Error reding file {file} from {bucket_output_files}: error detail:{e}"
    )
     


# ### Corte y empaque

# #### labor 100001

# In[7]:


volum_file.columns = list(volum_file.columns[:3]) + [pd.to_datetime(col, dayfirst=True, errors='coerce') for col in volum_file.columns[3:]]


# In[8]:


volum_distribution.columns = list(volum_distribution.columns[:1]) + [pd.to_datetime(col, dayfirst=True, errors='coerce') for col in volum_distribution.columns[1:]]


# In[9]:


volum_file_subset = volum_file.iloc[:,3:]


# In[10]:


group_data = group_by_month(volum_file_subset, volum_file)
month_columns = group_data.iloc[:, 2:].columns


# In[11]:


volum_distribution_matrix = volum_distribution.iloc[:, 1:]
volum_file_matrix = volum_file[volum_file['CONCEPTO'] == 'CAJAS'].iloc[:, 3:]
volum_file_subset = volum_file[volum_file['CONCEPTO'] == 'CAJAS']


# In[12]:


full_matrix = volum_file_subset.merge(cut_emp, on=['FINCA'], how='inner')
full_matrix_100001 = full_matrix[full_matrix['ID'] == 100001]
factor_100001 = full_matrix_100001.FACTOR.reset_index(drop=True)


# In[13]:


quantity_100001 = calculate_volume_distribution_factor(volum_file_matrix, volum_distribution_matrix, volum_file_subset, month_columns, factor_100001)


# In[14]:


labor_100001 = multiply_price(quantity_100001, full_matrix_100001, 'FINCA', 'TARIFA', 'PROMEDIADO', month_columns)


# In[15]:


promediado_100001 = multiply_by_month_promediado(labor_100001, promediado, month_columns)


# In[16]:


social_p_100001 = multiply_p_social(labor_100001, promediado_100001,full_matrix_100001, 'FINCA', 'PRESTACIONES', 'LABOR',month_columns)




# In[17]:


cost_100001 = total_cost(labor_100001, promediado_100001, social_p_100001, 'FINCA', 'LABOR', month_columns)


# #### labor 100002 -5

# In[18]:


volum_distribution_matrix = volum_distribution.iloc[:, 1:]

volum_file_matrix_10002 = volum_file[volum_file['CONCEPTO'] == 'CORTADOS'].iloc[:, 3:]
volum_file_subset_10002 = volum_file[volum_file['CONCEPTO'] == 'CORTADOS']

full_matrix_100002 = volum_file_subset_10002.merge(cut_emp, on=['FINCA'], how='inner')
full_matrix_100002 = full_matrix[full_matrix['ID'] == 100002]
full_matrix_100003 = full_matrix[full_matrix['ID'] == 100003]
full_matrix_100004 = full_matrix[full_matrix['ID'] == 100004]
full_matrix_100005 = full_matrix[full_matrix['ID'] == 100005]

factor_100002 = full_matrix_100002.FACTOR.reset_index(drop=True)
factor_100003 = full_matrix_100003.FACTOR.reset_index(drop=True)
factor_100004 = full_matrix_100004.FACTOR.reset_index(drop=True)
factor_100005 = full_matrix_100005.FACTOR.reset_index(drop=True)



# In[19]:


quantity_100002 = calculate_volume_distribution_factor(volum_file_matrix_10002, volum_distribution_matrix, volum_file_subset_10002, month_columns, factor_100002)


# In[20]:


quantity_100003 = calculate_volume_distribution_factor(volum_file_matrix_10002, 
                                                       volum_distribution_matrix, 
                                                       volum_file_subset_10002, month_columns, 
                                                       factor_100003)


# In[21]:


quantity_100004 = calculate_volume_distribution_factor(volum_file_matrix_10002, 
                                                       volum_distribution_matrix, 
                                                       volum_file_subset_10002, month_columns, 
                                                       factor_100004)


# In[22]:


quantity_100005 = calculate_volume_distribution_factor(volum_file_matrix_10002, 
                                                       volum_distribution_matrix, 
                                                       volum_file_subset_10002, month_columns, 
                                                       factor_100005)


# In[23]:


labor_100002 = multiply_price(quantity_100002, full_matrix_100002, 'FINCA', 'TARIFA', 'PROMEDIADO', month_columns)
labor_100003 = multiply_price(quantity_100003, full_matrix_100003, 'FINCA', 'TARIFA', 'PROMEDIADO', month_columns)
labor_100004 = multiply_price(quantity_100004, full_matrix_100004, 'FINCA', 'TARIFA', 'PROMEDIADO', month_columns)
labor_100005 = multiply_price(quantity_100005, full_matrix_100005, 'FINCA', 'TARIFA', 'PROMEDIADO', month_columns)


# In[24]:


promediado_100002 = multiply_by_month_promediado(labor_100002, promediado, month_columns)
promediado_100003 = multiply_by_month_promediado(labor_100003, promediado, month_columns)
promediado_100004 = multiply_by_month_promediado(labor_100004, promediado, month_columns)
promediado_100005 = multiply_by_month_promediado(labor_100005, promediado, month_columns)


# In[25]:


social_p_100002 = multiply_p_social(labor_100002, 
                                    promediado_100002,full_matrix_100002, 
                                    'FINCA', 'PRESTACIONES', 
                                    'LABOR',month_columns)

social_p_100003 = multiply_p_social(labor_100003, 
                                    promediado_100003,full_matrix_100003, 
                                    'FINCA', 'PRESTACIONES', 
                                    'LABOR',month_columns)
social_p_100004 = multiply_p_social(labor_100004, 
                                    promediado_100004,full_matrix_100004, 
                                    'FINCA', 'PRESTACIONES', 
                                    'LABOR',month_columns)
social_p_100005 = multiply_p_social(labor_100005, 
                                    promediado_100005,full_matrix_100005, 
                                    'FINCA', 'PRESTACIONES', 
                                    'LABOR',month_columns)


# In[26]:


cost_100002 = total_cost(labor_100002, promediado_100002, social_p_100002, 'FINCA', 'LABOR', month_columns)
cost_100003 = total_cost(labor_100003, promediado_100003, social_p_100003, 'FINCA', 'LABOR', month_columns)
cost_100004 = total_cost(labor_100004, promediado_100004, social_p_100004, 'FINCA', 'LABOR', month_columns)
cost_100005 = total_cost(labor_100005, promediado_100005, social_p_100005, 'FINCA', 'LABOR', month_columns)


# #### labor promediados

# In[27]:


full_matrix_promediados_100006 = full_matrix[full_matrix['ID'] == 100006]
full_matrix_promediados_1000023 = full_matrix[full_matrix['ID'] == 100023]
factor_promediados_100006 = full_matrix_promediados_100006.FACTOR.reset_index(drop=True)
factor_promediados_1000023 = full_matrix_promediados_1000023.FACTOR.reset_index(drop=True)
factor_concat = pd.concat([factor_promediados_100006, factor_promediados_1000023], ignore_index=True)


# In[28]:


quantity_promediados = calculate_promediados_factor(volum_distribution_matrix, volum_file_subset, month_columns, factor_concat)


# In[29]:


quantity_promediados_100023 = quantity_promediados.iloc[46:, :]
quantity_promediados_100006 = quantity_promediados.iloc[0:46, :]


# In[30]:


labor_promediados_100023 = multiply_price(quantity_promediados_100023, 
                                          full_matrix_promediados_1000023, 
                                          'FINCA', 'TARIFA', 'PROMEDIADO', 
                                          month_columns)


# In[31]:


labor_promediados_100006 = multiply_price(quantity_promediados_100006, 
                                          full_matrix_promediados_100006, 
                                          'FINCA', 'TARIFA', 'PROMEDIADO', 
                                          month_columns)


# In[32]:


promediado_100023 = multiply_by_month_promediado(labor_promediados_100023, promediado, month_columns)
promediado_100006 = multiply_by_month_promediado(labor_promediados_100006, promediado, month_columns)


# In[33]:


social_promediado_100023 = multiply_p_social(labor_promediados_100023, 
                                    promediado_100023 ,full_matrix_promediados_1000023, 
                                    'FINCA', 'PRESTACIONES', 
                                    'LABOR',month_columns)

social_promediado_1000006 = multiply_p_social(labor_promediados_100006,
                                    promediado_100006 ,full_matrix_promediados_100006, 
                                    'FINCA', 'PRESTACIONES', 
                                    'LABOR',month_columns)


# In[34]:


cost_promediado_100023 = total_cost(labor_promediados_100023, promediado_100023, social_promediado_100023, 'FINCA', 'LABOR', month_columns)
cost_promediado_100006 = total_cost(labor_promediados_100006, promediado_100006, social_promediado_1000006, 'FINCA', 'LABOR', month_columns)


# #### Ajustes (archivo plano) - 100007 y 100022

# In[35]:


full_matrix_promediados_100007 = full_matrix[full_matrix['ID'] == 100007]
full_matrix_promediados_100022 = full_matrix[full_matrix['ID'] == 100022]
quantity_promediados_100007 = ajustes_mo[ajustes_mo['LABOR'] == "AJUSTE DE COSECHA"]
quantity_promediados_100022 = ajustes_mo[ajustes_mo['LABOR'] == "AJUSTE DE EMPAQUE"]
full_matrix_promediados_100007_std = reparar_codificacion(full_matrix_promediados_100007, 'FINCA', aplicar_fix_cienaga=False)
full_matrix_promediados_100022_std = reparar_codificacion(full_matrix_promediados_100022, 'FINCA', aplicar_fix_cienaga=False)



# In[36]:


labor_promediados_100007 = multiply_price(quantity_promediados_100007, 
                                          full_matrix_promediados_100007_std, 
                                          'FINCA', 'TARIFA', 'PROMEDIADO', 
                                          month_columns)

promediado_100007 = multiply_by_month_promediado(labor_promediados_100007, promediado, month_columns)

social_promediado_100007 = multiply_p_social(labor_promediados_100007, 
                                    promediado_100007 ,full_matrix_promediados_100007, 
                                    'FINCA', 'PRESTACIONES', 
                                    'LABOR',month_columns)

cost_promediado_100007 = total_cost(labor_promediados_100007, promediado_100007, social_promediado_100007, 'FINCA', 'LABOR', month_columns)


# In[37]:


labor_promediados_100022 = multiply_price(quantity_promediados_100022, 
                                          full_matrix_promediados_100022_std, 
                                          'FINCA', 'TARIFA', 'PROMEDIADO', 
                                          month_columns)

promediado_100022 = multiply_by_month_promediado(labor_promediados_100022, promediado, month_columns)

social_promediado_100022 = multiply_p_social(labor_promediados_100022, 
                                    promediado_100022 ,full_matrix_promediados_100022, 
                                    'FINCA', 'PRESTACIONES', 
                                    'LABOR',month_columns)

cost_promediado_100022 = total_cost(labor_promediados_100022, promediado_100022, social_promediado_100022, 'FINCA', 'LABOR', month_columns)


# #### cantidades modulo ingreso

# In[38]:


revenue_sku_subset = revenue_sku[revenue_sku['SKU']!= 'Exc-']
revenue_sku_subset.loc[:, 'Farm'] = revenue_sku_subset['Farm'].astype(str).apply(unidecode)


# In[39]:


input_mo_join = pd.merge(revenue_sku_subset, sku[['SKU', 'TIPO']] ,how = 'left', on = 'SKU')
input_mo_join.rename(columns={'Farm':'FINCA'}, inplace=True)


# In[40]:


mo_grouped_total = group_by_type(input_mo_join, farms=farm_order)


# In[41]:


mo_pre_100008 = merge_factor_by_id(100008, mo_grouped_total, cut_emp)
mo_pre_100008_final = multipy_factor_combined(mo_pre_100008, 'PRE')
mo_pre_100008_final_subset = mo_pre_100008_final.iloc[:,2:54]
full_matrix_100008 = full_matrix[full_matrix['ID'] == 100008]
factor_100008 = full_matrix_100008.FACTOR.reset_index(drop=True)


# In[42]:


mo_100009 = group_by_type_sum(input_mo_join, farms=farm_order, types_sum=['EDK', 'ESP'])
mo_100009_factor = merge_factor_by_id(100009, mo_100009, cut_emp)
mo_100009_final =  multipy_factor_combined(mo_100009_factor, 'EDK+ESP')
mo_100009_final_subset = mo_100009_final.iloc[:,2:54]


# In[43]:


mo_100010 = group_by_type_sum(input_mo_join, farms=farm_order, types_sum=['ESP', 'MG'])
mo_100010_factor = merge_factor_by_id(100010, mo_100010, cut_emp)
mo_100010_final = multipy_factor_combined(mo_100010_factor, 'ESP+MG')
mo_100010_final_subset = mo_100010_final.iloc[:,2:54]


# In[44]:


mo_100011_factor = merge_factor_by_id(100011, mo_grouped_total, cut_emp)
mo_100011_final = multipy_factor_combined(mo_100011_factor, 'PRE')
mo_100011_final_subset = mo_100011_final.iloc[:,2:54]


# In[45]:


mo_100012_factor = merge_factor_by_id(100012, mo_grouped_total, cut_emp)
mo_100012_final = multipy_factor_combined(mo_100012_factor, 'WT')
mo_100012_final_subset = mo_100012_final.iloc[:,2:54]


# In[46]:


mo_100013_factor = merge_factor_by_id(100013, mo_grouped_total, cut_emp)
mo_100013_final = multipy_factor_combined(mo_100013_factor, 'MG')
mo_100013_final_subset = mo_100013_final.iloc[:,2:54]


# In[47]:


mo_08_13_concat = pd.concat([mo_pre_100008_final, mo_100009_final, mo_100010_final, 
                                   mo_100011_final, mo_100012_final, mo_100013_final], axis = 0)
mo_08_13_concat_subset = mo_08_13_concat.iloc[:,2:54]


# In[48]:


quantity_08_13 = calculate_volume_distribution_blocks(mo_08_13_concat_subset, volum_distribution_matrix, volum_file_subset, month_columns)


# In[49]:


id_list = [100008,100009, 100010, 100011, 100012, 100013]
full_matrix_block = full_matrix[full_matrix['ID'].isin(id_list)]
full_matrix_block.loc[:, 'FINCA'] = full_matrix_block['FINCA'].astype(str).apply(unidecode) 
full_matrix_block_subset = full_matrix_block[['ID', 'FINCA', 'PROMEDIADO', 'TARIFA']]


# In[50]:


# Convertimos la columna 'FINCA' a categoría con el orden deseado
full_matrix_block_subset['FINCA'] = pd.Categorical(
    full_matrix_block_subset['FINCA'],
    categories=farm_order,
    ordered=True
)

# Ordenamos por ID descendente y luego por FINCA según farm_order
full_matrix_block_subset_sorted = full_matrix_block_subset.sort_values(
    by=['ID', 'FINCA'],
    ascending=[True, True]  # ID descendente, FINCA según orden de categorías
)


# In[51]:


labor_08_13_concat = pd.concat([quantity_08_13.reset_index(drop=True), 
                                   full_matrix_block_subset_sorted.reset_index(drop=True)], 
                                   axis = 1)


# In[52]:


# Paso 1: Guardar una vez los valores originales de los meses
if f'original_{month_columns[0]}' not in labor_08_13_concat.columns:
    for col in month_columns:
        labor_08_13_concat[f'original_{col}'] = labor_08_13_concat[col]

# Paso 2: Multiplicar siempre desde los originales
for col in month_columns:
    labor_08_13_concat[col] = labor_08_13_concat[f'original_{col}'] * labor_08_13_concat['TARIFA']



# In[53]:


labor_08_13_subset = labor_08_13_concat[['FINCA', 'PROMEDIADO'] + list(month_columns)]
labor_08_13_subset.columns = labor_08_13_subset.columns.astype(str).str.strip()


# In[54]:


promediado_08_13 = multiply_by_month_promediado(labor_08_13_subset, promediado, month_columns)


# In[55]:


# Convertimos la columna 'FINCA' a categoría con el orden deseado
full_matrix_block['FINCA'] = pd.Categorical(
    full_matrix_block['FINCA'],
    categories=farm_order,
    ordered=True
)

# Ordenamos por ID descendente y luego por FINCA según farm_order
full_matrix_block_sorted = full_matrix_block.sort_values(
    by=['ID', 'FINCA'],
    ascending=[True, True]  # ID descendente, FINCA según orden de categorías
).reset_index(drop = True)


# In[56]:


full_matrix_block_sorted_social = full_matrix_block_sorted[['FINCA', 'PRESTACIONES', 'LABOR', 'ID']]
labor_08_13_social = pd.concat([labor_08_13_subset, full_matrix_block_sorted_social], axis = 1)


# In[57]:


social_p_08_13 = multiply_p_social_block(labor_08_13_social, promediado_08_13,full_matrix_block_sorted_social,month_columns)


# In[58]:


labor_08_13_social_cost = labor_08_13_social.iloc[:, 1:]
cost_08_13 = total_cost_block(labor_08_13_social_cost, promediado_08_13, social_p_08_13, 'FINCA', 'LABOR', month_columns)


# In[59]:


id_list_2 = [100014,100015, 100016, 100017, 100018, 100019]
full_matrix_14_19_id = full_matrix[full_matrix['ID'].isin(id_list_2)]


# In[60]:


#este codigo se utiliza para hacer un bucle for sobre todos los id que se encuentran en la lista id_list_2 y su formula corresponde con 
#la función calculate_volume_distribution_factor idem para el resto de calculos de labores, promediados, y costo total
quantity_list = []

for id_value in id_list_2:
    factor_series = full_matrix_14_19_id[full_matrix_14_19_id['ID'] == id_value]['FACTOR'].reset_index(drop=True)

    quantity_df = calculate_volume_distribution_factor(
        volum_file_matrix,
        volum_distribution_matrix,
        volum_file_subset,
        month_columns,
        factor_series
    )

    quantity_df['ID'] = id_value  # Añades el ID para identificar luego
    quantity_list.append(quantity_df)

# Unir todos los quantity en un solo DataFrame
quantity_all_df = pd.concat(quantity_list, ignore_index=True)


# In[61]:


labor_list = []

for id_value in id_list_2:
    # Filtrar los datos que corresponden a ese ID
    quantity_df = quantity_all_df[quantity_all_df['ID'] == id_value].reset_index(drop=True)
    filtered_df = full_matrix_14_19_id[full_matrix_14_19_id['ID'] == id_value].reset_index(drop=True)

    labor_df = multiply_price(
        quantity_df,
        filtered_df,
        'FINCA',
        'TARIFA',
        'PROMEDIADO',
        month_columns
    )

    labor_df['ID'] = id_value
    labor_list.append(labor_df)

# Unir todos los resultados en un solo DataFrame
final_labor_df_14_19 = pd.concat(labor_list, ignore_index=True)


# In[62]:


promediado_results = []

for id_value in id_list_2:
    labor_df = final_labor_df_14_19[final_labor_df_14_19['ID'] == id_value].reset_index(drop=True)

    promediado_df = multiply_by_month_promediado(
        labor_df,
        promediado,
        month_columns
    )
    
    promediado_df['ID'] = id_value
    promediado_results.append(promediado_df)

# Concatenar en un solo DataFrame final
final_promediado_df_14_19 = pd.concat(promediado_results, ignore_index=True)


# In[63]:


social_p_results = []

for id_value in id_list_2:
    labor_df = final_labor_df_14_19[final_labor_df_14_19['ID'] == id_value].reset_index(drop=True)
    promediado_df = final_promediado_df_14_19[final_promediado_df_14_19['ID'] == id_value].reset_index(drop=True)
    filtered_matrix = full_matrix_14_19_id[full_matrix_14_19_id['ID'] == id_value].reset_index(drop=True)

    social_df = multiply_p_social(
        labor_df,
        promediado_df,
        filtered_matrix,
        'FINCA',
        'PRESTACIONES',
        'LABOR',
        month_columns
    )
    
    social_df['ID'] = id_value
    social_p_results.append(social_df)

# Concatenar resultados en un solo DataFrame final
final_social_p_df_14_19 = pd.concat(social_p_results, ignore_index=True)


# In[64]:


cost_results = []

for id_value in id_list_2:
    labor_df = final_labor_df_14_19[final_labor_df_14_19['ID'] == id_value].reset_index(drop=True)
    promediado_df = final_promediado_df_14_19[final_promediado_df_14_19['ID'] == id_value].reset_index(drop=True)
    social_p_df = final_social_p_df_14_19[final_social_p_df_14_19['ID'] == id_value].reset_index(drop=True)

    cost_df = total_cost(
        labor_df,
        promediado_df,
        social_p_df,
        'FINCA',
        'LABOR',
        month_columns
    )
    
    cost_df['ID'] = id_value
    cost_results.append(cost_df)

# Concatenar resultados en un solo DataFrame final
final_cost_df_14_19 = pd.concat(cost_results, ignore_index=True)


# In[65]:


volum_file_matrix_binaria = (volum_file_matrix > 0).astype(int)


# In[66]:


full_matrix_100020 = full_matrix[full_matrix['ID'] == 100020]
factor_100020 = full_matrix_100020.FACTOR.reset_index(drop=True)


# In[67]:


quantity_df_100020 = calculate_volume_distribution_factor(
        volum_file_matrix_binaria,
        volum_distribution_matrix,
        volum_file_subset,
        month_columns,
        factor_100020
    )


# In[68]:


labor_100020 = multiply_price(quantity_df_100020, 
                              full_matrix_100020, 
                              'FINCA', 'TARIFA', 'PROMEDIADO', 
                              month_columns)


# In[69]:


promediado_100020 = multiply_by_month_promediado(labor_100020, 
                                                 promediado, 
                                                 month_columns) 


# In[70]:


social_p_100020 = multiply_p_social(labor_100020, 
                                    promediado_100020,
                                    full_matrix_100020,
                                    'FINCA', 'PRESTACIONES', 'LABOR',
                                    month_columns)


# In[71]:


cost_100020 = total_cost(labor_100020, 
                          promediado_100020, 
                          social_p_100020, 
                          'FINCA', 'LABOR', 
                          month_columns)


# In[72]:


mo_100009_final =  multipy_factor_combined(mo_100009_factor, 'EDK+ESP')


# In[73]:


mo_100021_factor = merge_factor_by_id(100021, mo_grouped_total, cut_emp)
mo_100021_mg = multipy_factor_combined_100021(mo_100021_factor, 'MG', 3)
mo_100021_edk = multipy_factor_combined_100021(mo_100021_factor, 'EDK',2)
mo_100021_mg_subset = mo_100021_mg.iloc[:,2:54].reset_index(drop=True)
mo_100021_edk_subset = mo_100021_edk.iloc[:,2:54].reset_index(drop=True)


# In[74]:


factor_100021 = full_matrix[full_matrix['ID'] == 100021].reset_index(drop = True).FACTOR
full_matrix_100021 = full_matrix[full_matrix['ID'] == 100021].reset_index(drop = True)
mg_plus_edk = pd.DataFrame()

for date in mo_100021_edk_subset.columns:
    mg_plus_edk[date] = mo_100021_mg_subset[date] + mo_100021_edk_subset[date]


# In[75]:


quantity_mg_edk = calculate_volume_distribution_factor(
        mg_plus_edk,
        volum_distribution_matrix,
        volum_file_subset,
        month_columns,
        factor_100021
    )


# In[76]:


labor_100021 = multiply_price(
        quantity_mg_edk,
        full_matrix_100021,
        'FINCA',
        'TARIFA',
        'PROMEDIADO',
        month_columns
    )


# In[77]:


promediado_100021 = multiply_by_month_promediado(labor_100021,
                                          promediado, 
                                          month_columns
                                          )
      


# In[78]:


social_100021 = multiply_p_social(labor_100021,
                                  promediado_100021,
                                  full_matrix_100021,
                                  'FINCA',
                                  'PRESTACIONES',
                                  'LABOR',
                                   month_columns)


# In[79]:


cost_100021 = total_cost(
        labor_100021,
        promediado_100021,
        social_100021,
        'FINCA',
        'LABOR',
        month_columns
    )


# #### CONCATENAMOS TODOS LOS ID POR LABOR, PROMEDIADO, P SOCIALES Y COSTO

# In[80]:


id_order_list = [100001, 100002, 100003, 100004, 100005, 100006, 100023, 
                  100007, 100008, 100009, 100010, 100011, 100012, 100013,
                  100014, 100015, 100016, 100017, 100018, 100019, 100020,
                  100021, 100022]


# In[81]:


order_map = {v: i for i, v in enumerate(id_order_list)}

full_matrix_sorted = (
    full_matrix.assign(_k=full_matrix['ID'].map(order_map).fillna(len(id_order_list)))
      .sort_values('_k', kind='mergesort')   # estable: preserva orden relativo de “otros”
      .drop(columns='_k')
      .reset_index(drop=True)
)


# In[82]:


#labor 

total_labor_concat = pd.concat([labor_100001, labor_100002, labor_100003, labor_100004,
                                labor_100005, labor_promediados_100006, labor_promediados_100007, 
                                labor_promediados_100023, 
                                labor_08_13_subset, final_labor_df_14_19, labor_100020,
                                labor_100021, labor_promediados_100022], axis = 0).reset_index(drop =True)


total_labor_concat_output = pd.concat([total_labor_concat, full_matrix_sorted[['GRUPO']]], axis = 1).reset_index(drop =True)
total_labor_concat_output_std = reparar_codificacion(total_labor_concat_output, 'FINCA', aplicar_fix_cienaga=True)

ce_labor_result = total_labor_concat_output.groupby(by=['FINCA', 'GRUPO']).sum().reset_index()
ce_labor_result_std = reparar_codificacion(ce_labor_result, 'FINCA', aplicar_fix_cienaga=False)
col_out = ['PROMEDIADO', 'ID', 'LABOR', 'PRESTACIONES']
ce_labor_result_filtrado = ce_labor_result_std[[c for c in ce_labor_result_std.columns if c not in col_out]]
ce_labor_result_order = farm_order_process(ce_labor_result_filtrado, farm_order, 'FINCA', 'GRUPO')
ce_labor_result_order.iloc[:, 2].sum()



# In[83]:


#promediado

total_promediado_concat = pd.concat([promediado_100001, promediado_100002, promediado_100003,
                                     promediado_100004, promediado_100005, 
                                     promediado_100006, promediado_100007, promediado_100023, promediado_08_13,
                                     final_promediado_df_14_19, promediado_100020,
                                     promediado_100021, promediado_100022], axis=0).reset_index(drop =True)

total_promediado_concat_output = pd.concat([total_promediado_concat, full_matrix_sorted[['GRUPO']]], axis = 1).reset_index(drop =True)
total_promediado_concat_output_std = reparar_codificacion(total_promediado_concat_output, 'FINCA', aplicar_fix_cienaga=True)


ce_promediado_result = total_promediado_concat_output_std.groupby(by=['FINCA', 'GRUPO']).sum().reset_index()
ce_promediado_result_std = reparar_codificacion(ce_promediado_result, 'FINCA', aplicar_fix_cienaga=False)
ce_promediado_result_filtrado = ce_promediado_result_std[[c for c in ce_promediado_result_std.columns if c not in col_out]]
ce_promediado_result_order = farm_order_process(ce_promediado_result_filtrado, farm_order, 'FINCA', 'GRUPO')
ce_promediado_result_order.iloc[:, 2].sum()



# In[84]:


#social
total_social_concat = pd.concat([social_p_100001, social_p_100002, social_p_100003,
                                 social_p_100004, social_p_100005, social_promediado_1000006,
                                 social_promediado_100007, 
                                 social_promediado_100023, social_p_08_13, final_social_p_df_14_19,
                                 social_p_100020, social_100021, social_promediado_100022], axis =0).reset_index(drop =True)

total_social_concat_output = pd.concat([total_social_concat, full_matrix_sorted[['GRUPO']]], axis = 1).reset_index(drop =True)
total_social_concat_output_std = reparar_codificacion(total_social_concat_output, 'FINCA', aplicar_fix_cienaga=True)

ce_social_result = total_social_concat_output_std.groupby(by=['FINCA', 'GRUPO']).sum().reset_index()
ce_social_result_std = reparar_codificacion(ce_social_result, 'FINCA', aplicar_fix_cienaga=False)
ce_social_result_filtrado = ce_social_result_std[[c for c in ce_social_result_std.columns if c not in col_out]]
ce_social_result_order = farm_order_process(ce_social_result_filtrado, farm_order, 'FINCA', 'GRUPO')
ce_social_result_order.iloc[:, 2].sum()


# In[85]:


#cost NOTA (EL ID 100022 Y 100007 son ajustes deben proporcionar el archivo plano)
total_cost_concat = pd.concat([cost_100001, cost_100002, cost_100003,
                               cost_100004, cost_100005, cost_promediado_100006,
                               cost_promediado_100007, cost_promediado_100023,
                               cost_08_13, final_cost_df_14_19, cost_100020,
                               cost_100021, cost_promediado_100022], axis =0).reset_index(drop=True)


total_cost_concat_output = pd.concat([total_cost_concat, full_matrix_sorted[['GRUPO']]], axis = 1).reset_index(drop =True)
total_cost_concat_output_std = reparar_codificacion(total_cost_concat_output, 'FINCA', aplicar_fix_cienaga=True)


ce_cost_result = total_cost_concat_output_std.groupby(by=['FINCA', 'GRUPO']).sum().reset_index()
ce_cost_result_std = reparar_codificacion(ce_cost_result, 'FINCA', aplicar_fix_cienaga=False)
ce_cost_result_filtrado = ce_cost_result_std[[c for c in ce_cost_result_std.columns if c not in col_out]]
ce_cost_result_order = farm_order_process(ce_cost_result_filtrado, farm_order, 'FINCA', 'GRUPO')
ce_cost_result_order.iloc[:, 2].sum()


# In[86]:


corte_empaque = pd.concat([ce_labor_result_order, ce_promediado_result_order.iloc[:, 2:],
                           ce_social_result_order.iloc[:, 2:], ce_cost_result_order.iloc[:, 2:]],
                           axis=1)


# In[87]:


#SALIDA MO DE CORTE Y EMPAQUE
corte_empaque.head(3)


# ### Parcela

# In[88]:


ha = volum_file[volum_file['CONCEPTO'] == 'HA CULTIVABLE'].iloc[:, 3:].replace(0, np.nan).reset_index(drop = True)
emb = volum_file[volum_file['CONCEPTO'] == 'EMBOLSADOS'].iloc[:, 3:].reset_index(drop = True)
desect_ha_subset = desect_ha.iloc[:, 1:]


# In[89]:


original_columns = desect_ha_subset.columns
columns_std  = pd.to_datetime(original_columns, errors = 'raise', dayfirst=True)
desect_ha_subset.columns = columns_std


# In[90]:


desect_emb = (desect_ha_subset / ha) * emb *0.5


# In[91]:


numerador = emb - desect_emb
denominador = ha - desect_ha_subset
denominador = denominador.replace(0, np.nan)
rac_ha = numerador.divide(denominador)
rac_ha = rac_ha.replace([np.inf, -np.inf], np.nan).fillna(0)
rac_ha[rac_ha < 0] = np.nan
rac_ha = rac_ha.fillna(0)


# In[92]:


rac_ha_farms = pd.concat([rac_ha, desect_ha.FINCA], axis = 1)
rac_ha_farms_std = reparar_codificacion(rac_ha_farms, 'FINCA', aplicar_fix_cienaga=False)
df_1_join = parcela_mo[parcela_mo['LABOR'] == 'PARCELA'][['FINCA', 'ESQUEMA']]
df_1_join_std = reparar_codificacion(df_1_join, 'FINCA', aplicar_fix_cienaga=False)


# In[93]:


rac_ha_farms_join = df_1_join_std.merge(rac_ha_farms_std, on='FINCA', how='inner').iloc[:,1:]
schema_result = vlookup_function(rac_ha_farms_join, esquema_mo, 'VALOR')
schema_result_percent = vlookup_function(rac_ha_farms_join, esquema_mo, 'PORCENTAJE')


# In[94]:


parcial_result = abs((desect_ha_subset - ha)).fillna(0)


# In[95]:


q_result = parcial_result.multiply(parcela_mo[parcela_mo['LABOR'] == 'PARCELA'].VUELTAS, axis=0)
amount_reesult = (q_result * schema_result) 
amount_reesult_final = amount_reesult.iloc[:, 0:-1]


# In[96]:


#AYUDANTE 
schema_result_std = standardize_column_dates(schema_result.iloc[:, 0:-1])
parcela_ay = parcela_mo[parcela_mo['LABOR']=='AYUDANTE']
parcela_plan = parcela_mo[parcela_mo['LABOR']=='PLANTILLA']
matrix_vtas = pd.DataFrame(np.repeat(parcela_ay['VUELTAS'].values[:, np.newaxis], len(ha.columns), axis=1),
                             index=ha.index, columns=ha.columns)
q_initial_result = ((ha - desect_ha_subset).divide(matrix_vtas.replace(0, np.nan))) * schema_result_percent
q_ay = q_initial_result.replace([np.inf, -np.inf], np.nan).fillna(0)
q_ay_std = standardize_column_dates(q_ay.iloc[:, 0:-1])
q_ay_amount = (q_ay_std * 65000)


# In[97]:


#PLANTILLA
matrix_vtas_plan = pd.DataFrame(np.repeat(parcela_plan['VUELTAS'].values[:, np.newaxis], len(ha.columns), axis=1),
                             index=ha.index, columns=ha.columns)
q_plan = desect_ha_subset.divide(matrix_vtas_plan.replace(0, np.nan)).fillna(0)
q_plan_std = standardize_column_dates(q_plan.iloc[:, 0:])
q_plan_amount = (q_plan_std * 65000)


# In[98]:


#EL PROBLEMA ESTA EN LA DIFERENCIA DE COLUMNAS AL HACER EL CONCAT 
amount_new_columns = amount_reesult_final.columns 
q_ay_amount.columns = amount_new_columns
q_plan_amount.columns = amount_new_columns


# In[99]:


q_total_results = pd.concat([q_result, q_ay, q_plan], axis = 0)
q_plan_results_amount = pd.concat([amount_reesult, q_ay_amount, q_plan_amount], axis = 0)


# In[100]:


quantity_parcela = calculate_volume_distribution_blocks(q_total_results, volum_distribution_matrix, volum_file_subset, month_columns)
##HASTA AQUI VALIDADO ESTÁ OOK 


# In[101]:


labor_parcela = calculate_volume_distribution_blocks(q_plan_results_amount, volum_distribution_matrix, volum_file_subset, month_columns)


# In[102]:


labor_parcela.columns = [
    col.strftime("%Y-%m") if isinstance(col, pd.Period) else str(col).strip()
    for col in labor_parcela.columns
]


# In[103]:


parcela_mo_subset = parcela_mo[['FINCA', 'PROMEDIADO', 'LABOR']]
labor_parcela_concat = pd.concat([parcela_mo_subset, labor_parcela], axis = 1)


# In[104]:


promediado_parcela = multiply_by_month_promediado(labor_parcela_concat, promediado, month_columns)


# In[105]:


social_parcela = social_p_parcela(labor_parcela_concat,promediado_parcela, parcela_mo.PRESTACIONES, month_columns )


# In[106]:


cost_parcela = total_cost_parcela(labor_parcela_concat,promediado_parcela, social_parcela, month_columns)
cost_parcela_std = reparar_codificacion(cost_parcela, 'FINCA')


# #### Agregado de labor en parcela para poder hacer el groupby por labor con labores ciclicas

# In[107]:


#LABOR PARCELA
cols_parcela_out = ['PROMEDIADO']
labor_parcela_resul = labor_parcela_concat[[c for c in labor_parcela_concat.columns if c not in cols_parcela_out]]
labor_parcela_std = reparar_codificacion(labor_parcela_resul, 'FINCA', aplicar_fix_cienaga=True)
labor_parcela_final = labor_parcela_std.groupby(by = ['FINCA']).sum().reset_index()
labor_parcela_final.LABOR = "PARCELA"
labor_parcela_final_order = farm_order_process(labor_parcela_final, farm_order, 'FINCA', 'LABOR')


# In[108]:


#PROMEDIO PARCELA
promediado_parcela_concat = pd.concat([promediado_parcela, parcela_mo[['LABOR']]], axis= 1)
promediado_parcela_concat_std = reparar_codificacion(promediado_parcela_concat, 'FINCA', aplicar_fix_cienaga=True)
promediado_parcela_final = promediado_parcela_concat_std.groupby(by = ['FINCA']).sum().reset_index()
promediado_parcela_final.LABOR = "PARCELA"
promediado_parcela_final_order = farm_order_process(promediado_parcela_final, farm_order, 'FINCA', 'LABOR')


# In[109]:


#SOCIAL PARCELA
social_parcela_concat = pd.concat([social_parcela, parcela_mo[['LABOR']]], axis= 1)
social_parcela_concat_std = reparar_codificacion(social_parcela_concat, 'FINCA', aplicar_fix_cienaga=True)
social_parcela_final = social_parcela_concat_std[[c for c in social_parcela_concat_std.columns if c not in cols_parcela_out]]
social_parcela_final_gb = social_parcela_final.groupby(by = ['FINCA']).sum().reset_index()
social_parcela_final_gb.LABOR = "PARCELA"
social_parcela_final_order = farm_order_process(social_parcela_final_gb, farm_order, 'FINCA', 'LABOR')


# In[110]:


#COSTO PARCELA
cost_parcela_concat = pd.concat([cost_parcela, parcela_mo[['LABOR']]], axis= 1)
cost_parcela_std = reparar_codificacion(cost_parcela_concat, 'FINCA', aplicar_fix_cienaga=True)
cost_parcela_resul = cost_parcela_std[[c for c in cost_parcela_std.columns if c not in cols_parcela_out]]
cost_parcela_final_gb = cost_parcela_resul.groupby(by = ['FINCA']).sum().reset_index()
cost_parcela_final_gb.LABOR = "PARCELA"
cost_parcela_final_order = farm_order_process(cost_parcela_final_gb, farm_order, 'FINCA', 'LABOR')


# 

# ### Labores ciclicas

# In[111]:


#Dataframes a utilizar 
ha_cultiv = volum_file[volum_file['CONCEPTO'] == 'HA CULTIVABLE'].iloc[:, 1:].reset_index(drop =True)
emb_ciclic = volum_file[volum_file['CONCEPTO'] == 'EMBOLSADOS'].iloc[:, 1:].reset_index(drop =True)
cortados_ciclic = volum_file[volum_file['CONCEPTO'] == 'CORTADOS'].iloc[:, 1:].reset_index(drop =True)
pre_oper_ciclic = volum_file[volum_file['CONCEPTO'] == 'HA EN PREOPERATIVO'].iloc[:, 1:].reset_index(drop =True)
prod_ciclic = volum_file[volum_file['CONCEPTO'] == 'HA EN PRODUCCIÓN'].iloc[:, 1:].reset_index(drop =True)
curvas_c_std = standardize_column_dates(curvas_c)
ha_cultiv_std = standardize_column_dates(ha_cultiv)
emb_ciclic_std = standardize_column_dates(emb_ciclic)
cortados_ciclic_std = standardize_column_dates(cortados_ciclic)
pre_oper_ciclic_std = standardize_column_dates(pre_oper_ciclic)
prod_ciclic_std = standardize_column_dates(prod_ciclic)


# In[112]:


#estandarizar algunos casos detectados con tildes que afectan el proceso
ha_cultiv_std_unicod = reparar_codificacion(ha_cultiv_std, 'FINCA')
emb_ciclic_std_unicod = reparar_codificacion(emb_ciclic_std, 'FINCA')
cortados_ciclic_std_unicod = reparar_codificacion(cortados_ciclic_std, 'FINCA')
pre_oper_ciclic_std_unicod = reparar_codificacion(pre_oper_ciclic_std, 'FINCA')
prod_ciclic_std_unicod = reparar_codificacion(prod_ciclic_std, 'FINCA')


# In[113]:


#Filtro por ID a procesar para HA CULTIVABLE
ciclicas_cultiv = ciclicas[ciclicas['ID'].isin([200001, 200002, 400003, 500003, 500004, 600007, 600010,
                                                1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 1000007, 1000008])]
ha_cultiv_merge = ha_cultiv_std_unicod.merge(ciclicas_cultiv[['ID', 'FINCA', 
                                            'CURVA', 'FACTOR']], on = 'FINCA', how = 'outer').sort_values(by = 'ID', 
                                            ascending=True).reset_index(drop =True)
ha_cultiv_merge_curve_c = ha_cultiv_merge.merge(curvas_c_std, on = 'CURVA', how = 'inner')

#APLICAMOS LA MULTIPLICACIÓN DEL DATAFRAME DE HA CULTIVABLE
process_q_ha_cultiv = labores_ciclicas(ha_cultiv_merge_curve_c, 'ID', 'FINCA', 'FACTOR').dropna(axis = 0)
#process_q_ordered = farm_order_process(process_q_ha_cultiv, farm_order, columna_finca='FINCA', columna_id='ID')


# In[114]:


#Filtro por ID a procesar para EMBOLSADOS
ciclicas_emb = ciclicas[ciclicas['ID'].isin([200003, 300001, 300002,300003, 300009])]
emb_ciclic_merge = emb_ciclic_std_unicod.merge(ciclicas_emb[['ID', 'FINCA', 
                                            'CURVA', 'FACTOR']], on = 'FINCA', how = 'outer').sort_values(by = 'ID', 
                                            ascending=True).reset_index(drop =True)
emb_ciclic_merge_curve_c = emb_ciclic_merge.merge(curvas_c_std, on = 'CURVA', how = 'inner')

#APLICAMOS LA MULTIPLICACIÓN DEL DATAFRAME DE EMBOLSE
process_q_ciclicas_emb = labores_ciclicas(emb_ciclic_merge_curve_c).dropna(axis = 0)
#process_q_ordered_emb = farm_order_process(process_q_ciclicas_emb, farm_order, columna_finca='FINCA', columna_id='ID')


# In[115]:


#Filtro por ID a procesar para CORTADOS
cortados_ciclic_id = ciclicas[ciclicas['ID'].isin([300004, 300007, 300008])]
cortados_ciclic_merge = cortados_ciclic_std_unicod.merge(cortados_ciclic_id[['ID', 'FINCA', 
                                            'CURVA', 'FACTOR']], on = 'FINCA', how = 'outer').sort_values(by = 'ID', 
                                            ascending=True).reset_index(drop =True)
cortados_ciclic_merge_curve_c = cortados_ciclic_merge.merge(curvas_c_std, on = 'CURVA', how = 'inner')

#APLICAMOS LA MULTIPLICACIÓN DEL DATAFRAME DE EMBOLSE
process_q_cortados_ciclic = labores_ciclicas(cortados_ciclic_merge_curve_c).dropna(axis = 0)
#process_q_ordered_cortados = farm_order_process(process_q_cortados_ciclic, farm_order, columna_finca='FINCA', columna_id='ID')


# In[116]:


#Filtro por ID a procesar para PREOPERATIVO
pre_oper_ciclic_id = ciclicas[ciclicas['ID'].isin([400002, 400004, 400005, 500002, 500005, 500006, 
                                                600002, 600005,600008, 600009])]
pre_oper_ciclic_merge = pre_oper_ciclic_std_unicod.merge(pre_oper_ciclic_id[['ID', 'FINCA', 
                                            'CURVA', 'FACTOR']], on = 'FINCA', how = 'outer').sort_values(by = 'ID', 
                                            ascending=True).reset_index(drop =True)
pre_oper_ciclic_merge_curve_c = pre_oper_ciclic_merge.merge(curvas_c_std, on = 'CURVA', how = 'inner')

#APLICAMOS LA MULTIPLICACIÓN DEL DATAFRAME DE PREOPERATIVO
process_q_pre_oper_ciclic = labores_ciclicas(pre_oper_ciclic_merge_curve_c).dropna(axis = 0)
#process_q_ordered_pre_oper = farm_order_process(process_q_pre_oper_ciclic, farm_order, columna_finca='FINCA', columna_id='ID')


# In[117]:


#Filtro por ID a procesar para PRODUCCIÓN
prod_ciclic_id = ciclicas[ciclicas['ID'].isin([400001, 500001,600001, 600004])]
prod_ciclic_merge = prod_ciclic_std_unicod.merge(prod_ciclic_id[['ID', 'FINCA', 
                                            'CURVA', 'FACTOR']], on = 'FINCA', how = 'outer').sort_values(by = 'ID', 
                                            ascending=True).reset_index(drop =True)
prod_ciclic_merge_curve_c = prod_ciclic_merge.merge(curvas_c_std, on = 'CURVA', how = 'inner')

#APLICAMOS LA MULTIPLICACIÓN DEL DATAFRAME PRODUCCIÓN
process_q_prod_ciclic = labores_ciclicas(prod_ciclic_merge_curve_c).dropna(axis = 0)
#process_q_ordered_prod_ciclic = farm_order_process(process_q_prod_ciclic, farm_order, columna_finca='FINCA', columna_id='ID')


# In[118]:


#filtro solamente procesando curva y factor
process_factor_id = ciclicas[ciclicas['ID'].isin([200004, 300005,300006])]

process_factor_merge = process_factor_id.merge(curvas_c_std, on = 'CURVA', how = 'inner')

#nos quedamos con las columnas de interés (fechas)
columnas_fijas = ['ID', 'FINCA', 'MAESTRA', 'CURVA', 'PROMEDIADO', 'PRESTACIONES', 'FACTOR', 'GRUPO']
columnas_fecha = [col for col in process_factor_merge.columns if col not in columnas_fijas]
#multiplicamos por el factor
process_factor_merge[columnas_fecha] = process_factor_merge[columnas_fecha].multiply(process_factor_merge['FACTOR'], axis=0)
final_columns = ['ID', 'FINCA'] + columnas_fecha


# In[119]:


process_factor_merge_filter = process_factor_merge[final_columns]


# In[120]:


final_ciclics = pd.concat([process_q_ha_cultiv, process_q_ciclicas_emb, process_q_cortados_ciclic , process_q_pre_oper_ciclic
                           , process_q_prod_ciclic, process_factor_merge_filter],axis = 0)

#final_ciclics_std = reparar_codificacion(final_ciclics, 'FINCA')


# In[121]:


final_ciclics_ordered = farm_order_process(final_ciclics, farm_order, 'FINCA', 'ID')


# In[122]:


quantity_ciclics = calculate_volume_distribution_blocks_ciclics(final_ciclics_ordered, volum_distribution_matrix, None, month_columns)


# In[123]:


labor_ciclics = ciclics_labor_calculation(quantity_ciclics, ciclicas)


# In[124]:


labor_ciclics_merged = labor_ciclics.merge(ciclicas[['PROMEDIADO', 'ID', 'FINCA']], on = ['ID', 'FINCA'], how = 'inner')


# In[125]:


promediado_ciclics = multiply_by_month_promediado_ciclics(labor_ciclics_merged.iloc[:, 1:], promediado, month_columns)
promediado_ciclics_id_concat = pd.concat([promediado_ciclics, labor_ciclics_merged[['ID']]], axis = 1)


# In[126]:


social_p_ciclics = multiply_p_social_block_ciclics(labor_ciclics_merged, promediado_ciclics_id_concat,ciclicas,month_columns)


# In[127]:


cost_ciclics = total_cost_block(labor_ciclics_merged, promediado_ciclics_id_concat, social_p_ciclics, 'FINCA', 'ID', month_columns)


# #### Calculo para Salida MO de parcela y ciclics

# In[128]:


#LABOR CILCICS -PARCELA
parcela_id_list = ["200001", "200002", "200003", "200004"]
labor_ciclics_parcela = labor_ciclics_merged[labor_ciclics_merged['ID'].isin(parcela_id_list)]
labor_ciclics_parcela_gb = labor_ciclics_parcela.groupby(by=['FINCA']).sum().reset_index()
labor_ciclics_parcela_subset = labor_ciclics_parcela_gb[[c for c in labor_ciclics_parcela_gb.columns if c not in cols_parcela_out]]
labor_ciclics_parcela_subset.ID = "PARCELA" 
labor_ciclics_parcela_std  = reparar_codificacion(labor_ciclics_parcela_subset, 'FINCA', aplicar_fix_cienaga=True)
labor_ciclics_parcela_order = farm_order_process_concat(labor_ciclics_parcela_std , farm_order, 'FINCA')
labor_ciclics_parcela_order.rename(columns={"ID":"LABOR"}, inplace=True)
labor_mo_parcela = sum_by_months_parcela(labor_parcela_final_order, labor_ciclics_parcela_order, month_columns)
labor_mo_parcela.iloc[:, 2].sum()


# In[129]:


#PROMEDIADO CILCICS -PARCELA
promediado_ciclics_parcela = promediado_ciclics_id_concat[promediado_ciclics_id_concat['ID'].isin(parcela_id_list)]
promediado_ciclics_parcela_gb = promediado_ciclics_parcela.groupby(by=['FINCA']).sum().reset_index()
promediado_ciclics_parcela_subset = promediado_ciclics_parcela_gb[[c for c in promediado_ciclics_parcela_gb.columns if c not in cols_parcela_out]]
promediado_ciclics_parcela_subset.ID = "PARCELA" 
promediado_ciclics_parcela_std  = reparar_codificacion(promediado_ciclics_parcela_subset, 'FINCA', aplicar_fix_cienaga=True)
promediado_ciclics_parcela_order = farm_order_process_concat(promediado_ciclics_parcela_std , farm_order, 'FINCA')
promediado_ciclics_parcela_order.rename(columns={"ID":"LABOR"}, inplace=True)
promediado_mo_parcela = sum_by_months_parcela(promediado_parcela_final_order, promediado_ciclics_parcela_order, month_columns)
promediado_mo_parcela.iloc[:, 2].sum()


# In[130]:


#SOCIAL CICLICS -PARCELA
social_ciclics_parcela = social_p_ciclics[social_p_ciclics['ID'].isin(parcela_id_list)]
social_ciclics_parcela_gb = social_ciclics_parcela.groupby(by=['FINCA']).sum().reset_index()
social_ciclics_parcela_subset = social_ciclics_parcela_gb[[c for c in social_ciclics_parcela_gb.columns if c not in cols_parcela_out]]
social_ciclics_parcela_subset.ID = "PARCELA" 
social_ciclics_parcela_std  = reparar_codificacion(social_ciclics_parcela_subset, 'FINCA', aplicar_fix_cienaga=True)
social_ciclics_parcela_order = farm_order_process_concat(social_ciclics_parcela_std , farm_order, 'FINCA')
social_ciclics_parcela_order.rename(columns={"ID":"LABOR"}, inplace=True)
social_mo_parcela = sum_by_months_parcela(social_parcela_final_order, social_ciclics_parcela_order, month_columns)
social_mo_parcela.iloc[:, 3].sum()


# In[131]:


#COSTO CICLICS -PARCELA
cost_mo_parcela = total_cost_block(labor_mo_parcela, promediado_mo_parcela, social_mo_parcela, 'FINCA', 'LABOR', month_columns)
cost_mo_parcela.iloc[:, 2].sum() 


# In[132]:


#SALIDA DE PARCELA
final_parcela_output_mo = pd.concat([labor_mo_parcela, promediado_mo_parcela.iloc[:, 2:], 
                               social_mo_parcela.iloc[:, 2:], cost_mo_parcela.iloc[:, 2:]], axis=1)
final_parcela_output_mo.rename(columns={'LABOR': 'GRUPO'}, inplace=True)


# In[133]:


final_parcela_output_mo.head(3)


# #### Calculo salida labores cliclics exclusivas

# In[134]:


#TRAEMOS LA LABOR PARA PODER AGRUPAR
labor_ciclics_mo = pd.concat([labor_ciclics_merged, ciclicas[['GRUPO']]], axis = 1)
promediado_ciclics_id_concat_mo = pd.concat([promediado_ciclics_id_concat, ciclicas[['GRUPO']]], axis = 1)
social_p_ciclics_mo = pd.concat([social_p_ciclics, ciclicas[['GRUPO']]], axis = 1)
cost_ciclics_mo = pd.concat([cost_ciclics, ciclicas[['GRUPO']]], axis = 1)


# In[135]:


#SALIDA PARA LABOR CICLICAS
labor_ciclics_ouput = labor_ciclics_mo.groupby(by = ['GRUPO','FINCA']).sum().reset_index()
labor_ciclics_ouput_order =reorder_output_materials(labor_ciclics_ouput, farm_order, 'FINCA', 
                                                    ciclicas.loc[ciclicas['GRUPO'].ne('PARCELA'), 
                                                                 'GRUPO'].unique(), 'GRUPO')
df = labor_ciclics_ouput_order.rename(columns=lambda c: str(c).strip())

labor_ciclics_ouput_mo = df.filter(items=["FINCA", "GRUPO"]).join(
    df.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$')
)


# In[136]:


labor_ciclics_ouput_mo


# In[137]:


#SALIDA PARA PROMEDIADOS CICLICAS
promediado_ciclics_ouput = promediado_ciclics_id_concat_mo.groupby(by = ['GRUPO','FINCA']).sum().reset_index()
promediado_ciclics_ouput_order =reorder_output_materials(promediado_ciclics_ouput, farm_order, 'FINCA', 
                                                    ciclicas.loc[ciclicas['GRUPO'].ne('PARCELA'), 
                                                                 'GRUPO'].unique(), 'GRUPO')
df_2 = promediado_ciclics_ouput_order.rename(columns=lambda c: str(c).strip())

promediado_ciclics_ouput_mo = df_2.filter(items=["FINCA", "GRUPO"]).join(
    df_2.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$')
)


# In[138]:


#SALIDA PARA SOCIAL CICLICAS
social_ciclics_ouput = social_p_ciclics_mo.groupby(by = ['GRUPO','FINCA']).sum().reset_index()
social_ciclics_ouput_order =reorder_output_materials(social_ciclics_ouput, farm_order, 'FINCA', 
                                                    ciclicas.loc[ciclicas['GRUPO'].ne('PARCELA'), 
                                                                 'GRUPO'].unique(), 'GRUPO')
df_3 = social_ciclics_ouput_order.rename(columns=lambda c: str(c).strip())

social_ciclics_ouput_mo = df_2.filter(items=["FINCA", "GRUPO"]).join(
    df_3.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$')
)


# In[139]:


cost_ciclics_mo = total_cost_block(labor_ciclics_ouput_mo, promediado_ciclics_ouput_mo, social_ciclics_ouput_mo, 'FINCA', 'GRUPO', month_columns)
cost_ciclics_mo.iloc[:, 2].sum()


# In[140]:


#SALIDA DE LABORES CICLICAS CONCATENADA
final_ciclics_labor_mo = pd.concat([labor_ciclics_ouput_mo, promediado_ciclics_ouput_mo.iloc[:, 2:],
                              social_ciclics_ouput_mo.iloc[:, 2:], cost_ciclics_mo.iloc[:,2:]], axis = 1)


# ### Labores fijas

# In[141]:


#arreglar URTENTE OJO OJO OJO OJO OJO """"""""""
quantity_fijas_result = quantity_fijas(volum_distribution, f_labor, month_columns, 'SEMANA', mult_cols =('FACTOR',), id_mode = "str" )
quantity_fijas_result.iloc[:, 2].sum()


# In[142]:


labor_fijas = ciclics_labor_calculation(quantity_fijas_result, f_labor)


# In[143]:


labor_fijas_merged = labor_fijas.merge(f_labor[['PROMEDIADO', 'FINCA', 'ID']], on =['FINCA', 'ID'], how = 'inner')


# In[144]:


promediado_fijas = multiply_by_month_promediado_ciclics(labor_fijas_merged.iloc[:, 1:], promediado, month_columns)
promediado_fijas_id_concat = pd.concat([promediado_fijas, labor_fijas_merged[['ID', 'FINCA']]], axis = 1)


# In[145]:


social_p_fijas = multiply_p_social_block_ciclics(labor_fijas_merged, promediado_fijas_id_concat, f_labor, month_columns)


# In[146]:


cost_fijas = total_cost_block(labor_fijas_merged, promediado_fijas_id_concat, social_p_fijas, 'FINCA', 'ID', month_columns)


# In[147]:


cost_fijas.iloc[:, 2].sum()


# #### Salida labores fijas

# In[148]:


#CONCATENAMOS LABOR FIJA PARA OBTENER EL GRUPO
labor_fijas_merged_output = pd.concat([labor_fijas_merged, f_labor[['GRUPO']]], axis=1)
promediado_fijas_output = pd.concat([promediado_fijas_id_concat, f_labor[['GRUPO']]], axis=1)
social_fijas_output  = pd.concat([social_p_fijas, f_labor[['GRUPO']]], axis=1)


# In[149]:


f_labor.GRUPO.unique()


# In[150]:


#SALIDA PARA LABOR FIJAS
labor_fijas_mo = labor_fijas_merged_output.groupby(by = ['GRUPO','FINCA']).sum().reset_index()
labor_fijas_mo_std = reparar_codificacion(labor_fijas_mo, 'FINCA', aplicar_fix_cienaga=True)
labor_fijas_mo_order =reorder_output_materials(labor_fijas_mo_std , farm_order, 'FINCA', 
                                                    f_labor.GRUPO.unique(), 'GRUPO')

fijas_df = labor_fijas_mo_order.rename(columns=lambda c: str(c).strip())

#ordena las columnas del dataframe
labor_fijas_ouput_mo = fijas_df.filter(items=["FINCA", "GRUPO"]).join(
    fijas_df.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$')
)


# In[151]:


#SALIDA PARA PROMEDIADO FIJAS
promediado_fijas_mo = promediado_fijas_output.groupby(by = ['GRUPO','FINCA']).sum().reset_index()
promediado_fijas_mo_std = reparar_codificacion(promediado_fijas_mo, 'FINCA', aplicar_fix_cienaga=True)
promediado_fijas_mo_order =reorder_output_materials(promediado_fijas_mo_std , farm_order, 'FINCA', 
                                                    f_labor.GRUPO.unique(), 'GRUPO')

fijas_df_2 = promediado_fijas_mo_order.rename(columns=lambda c: str(c).strip())

promediado_fijas_ouput_mo = fijas_df_2.filter(items=["FINCA", "GRUPO"]).join(
    fijas_df_2.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$')
)


# In[152]:


#SALIDA PARA SOCIAL FIJAS
social_fijas_mo = social_fijas_output.groupby(by = ['GRUPO','FINCA']).sum().reset_index()
social_fijas_mo_std = reparar_codificacion(social_fijas_mo, 'FINCA', aplicar_fix_cienaga=True)
social_fijas_mo_order =reorder_output_materials(social_fijas_mo_std , farm_order, 'FINCA', 
                                                    f_labor.GRUPO.unique(), 'GRUPO')

fijas_df_3 = social_fijas_mo_order.rename(columns=lambda c: str(c).strip())

social_fijas_ouput_mo = fijas_df_3.filter(items=["FINCA", "GRUPO"]).join(
    fijas_df_3.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$')
)


# In[153]:


cost_fijas_output_mo = total_cost_block(labor_fijas_ouput_mo, promediado_fijas_ouput_mo, social_fijas_ouput_mo, 'FINCA', 'GRUPO', month_columns)
cost_fijas_output_mo.iloc[:, 2].sum()


# In[154]:


#SALIDA DE LABORES FIJAS CONCATENADA
final_fijas_labor_mo = pd.concat([labor_fijas_ouput_mo, promediado_fijas_ouput_mo.iloc[:,2:],  
                                  social_fijas_ouput_mo.iloc[:,2:], cost_fijas_output_mo.iloc[:,2:]], axis = 1)


# ### Labores varias

# In[155]:


salida_fertilizante_std = reparar_codificacion(salida_fertilizante,'FINCA', aplicar_fix_cienaga=True)
salida_fertilizante_std_sacos = salida_fertilizante_std[salida_fertilizante_std['UNIDAD'] =='SAC']
salida_fertilizante_std_yeso = salida_fertilizante_std[salida_fertilizante_std['UNIDAD'] =='SACO'].reset_index(drop =True)
salida_fertilizante_std_compost = salida_fertilizante_std[salida_fertilizante_std['UNIDAD'] =='BULTO'].reset_index(drop =True)


# In[156]:


factor_800001 = other_labors[other_labors['ID']== 800001].FACTOR
factor_800002 = other_labors[other_labors['ID']== 800002].FACTOR
factor_800003 = other_labors[other_labors['ID']== 800003].FACTOR
factor_800004 = other_labors[other_labors['ID']== 800004].reset_index().FACTOR
factor_800005 = other_labors[other_labors['ID']== 800005].reset_index().FACTOR
factor_800006 = other_labors[other_labors['ID']== 800006].reset_index().FACTOR
factor_800008 = other_labors[other_labors['ID']== 800008].reset_index().FACTOR
factor_800009 = other_labors[other_labors['ID']== 800009].reset_index().FACTOR
factor_800010 = other_labors[other_labors['ID']== 800010].FACTOR



factor_1000012 = other_labors[other_labors['ID']== 1000012].FACTOR
factor_1000013 = other_labors[other_labors['ID']== 1000013].FACTOR
factor_1000014 = other_labors[other_labors['ID']== 1000014].FACTOR
factor_1000015 = other_labors[other_labors['ID']== 1000015].FACTOR
factor_1000016 = other_labors[other_labors['ID']== 1000016].FACTOR


# In[157]:


def multiply_mat_factor(df: pd.DataFrame, factor, start_col: int = 2) -> pd.DataFrame:
    out = df.copy(deep=True)
    out.iloc[:, start_col:] = out.iloc[:, start_col:].multiply(factor, axis=0)
    return out


# In[158]:


#calculo de la matriz de cantidades para todos los ID
factor_800001_quantity = quantity_other_labors(pre_oper_ciclic, volum_distribution, month_columns, factor_800001, multiplicar_por_s=True)
factor_800002_quantity = quantity_other_labors(pre_oper_ciclic, volum_distribution, month_columns, factor_800002, multiplicar_por_s=True)
factor_800003_quantity = quantity_other_labors(ha_cultiv, volum_distribution, month_columns, factor_800003, multiplicar_por_s=True)

factor_800004_quantity = multiply_mat_factor(salida_fertilizante_std_sacos, factor_800004, 2)
factor_800004_quantity_ordered = farm_order_process_concat(factor_800004_quantity, farm_order, 'FINCA').iloc[: , 1:]

factor_800005_quantity = multiply_mat_factor(salida_fertilizante_std_yeso, factor_800005, 2)
factor_800005_quantity_ordered = farm_order_process_concat(factor_800005_quantity, farm_order, 'FINCA').iloc[: , 1:]

factor_800006_quantity = multiply_mat_factor(salida_fertilizante_std_compost, factor_800006, 2)
factor_800006_quantity_ordered = farm_order_process_concat(factor_800006_quantity, farm_order, 'FINCA').iloc[: , 1:]


factor_800008_quantity = quantity_other_labors_800008(volum_distribution, farm_order, month_columns, factor_800008 ,'SEMANA')
factor_800009_quantity = quantity_other_labors(ha_cultiv, volum_distribution, month_columns, factor_800009, multiplicar_por_s=False)


# In[159]:


#calculo de la matriz de cantidades para todos los ID
factor_800010_quantity = salida_fertilizante_std.groupby(by='FINCA', sort=False, as_index=False).sum()
factor_800010_quantity_subset = factor_800010_quantity.loc[:, factor_800010_quantity.columns != 'UNIDAD']
factor_800010_quantity_ordered = farm_order_process_concat(factor_800010_quantity_subset, farm_order, 'FINCA')


# In[160]:


#calculo de la matriz de cantidades para todos los ID
factor_1100001_quantity = build_factor_quantity(curvas_c_mensual, other_labors, 1100001)
factor_1100002_quantity = build_factor_quantity(curvas_c_mensual, other_labors, 1100002)
factor_1100003_quantity = build_factor_quantity(curvas_c_mensual, other_labors, 1100003)
factor_1000009_quantity = build_factor_quantity(curvas_c_mensual, other_labors, 1000009)
factor_1000010_quantity = build_factor_quantity(curvas_c_mensual, other_labors, 1000010)
factor_1000011_quantity = build_factor_quantity(curvas_c_mensual, other_labors, 1000011)
factor_1000012_quantity = quantity_other_labors(ha_cultiv, volum_distribution, month_columns, factor_1000012, multiplicar_por_s=True)
factor_1000013_quantity = quantity_other_labors(ha_cultiv, volum_distribution, month_columns, factor_1000013, multiplicar_por_s=True)
factor_1000014_quantity = quantity_other_labors(ha_cultiv, volum_distribution, month_columns, factor_1000014, multiplicar_por_s=True)
factor_1000015_quantity = quantity_other_labors(ha_cultiv, volum_distribution, month_columns, factor_1000015, multiplicar_por_s=True)
factor_1000016_quantity = quantity_other_labors(ha_cultiv, volum_distribution, month_columns, factor_1000016, multiplicar_por_s=True)


# In[161]:


# 1) Columnas base
nuevas_columnas = list(factor_800001_quantity.columns)

# 2) Lista de dataframes a estandarizar (ajusta el nombre 800010 si en tu entorno es 8000010)
frames = [
    factor_800001_quantity, factor_800002_quantity, 
    factor_800003_quantity, factor_800004_quantity_ordered,
    factor_800005_quantity_ordered, factor_800006_quantity_ordered, 
    factor_800008_quantity, factor_800009_quantity,
    factor_800010_quantity_ordered, factor_1100001_quantity,
    factor_1100002_quantity, factor_1100003_quantity, factor_1000009_quantity,
    factor_1000010_quantity, factor_1000011_quantity,factor_1000012_quantity,
    factor_1000013_quantity, factor_1000014_quantity,
    factor_1000015_quantity, factor_1000016_quantity
]

# 3) Estandarizar columnas con un for
frames_std = [column_replacement(df, nuevas_columnas) for df in frames]

# 4) Concatenar
quantity_labor_varias = pd.concat(frames_std, axis=0, ignore_index=True)


# In[162]:


quantity_labor_varias_concat = pd.concat([other_labors[['ID']], quantity_labor_varias], axis = 1)


# In[163]:


labor_varias = ciclics_labor_calculation(quantity_labor_varias_concat, other_labors)


# In[164]:


labor_varias_merged = labor_varias.merge(other_labors[['PROMEDIADO', 'FINCA', 'ID']], on =['FINCA', 'ID'], how = 'inner')
promediado_varias = multiply_by_month_promediado_ciclics(labor_varias_merged.iloc[:, 1:], promediado, month_columns)
promediado_varias_id_concat = pd.concat([promediado_varias, labor_varias_merged[['ID']]], axis = 1)


# In[165]:


social_p_varias = multiply_p_social_block_ciclics(labor_varias_merged , promediado_varias_id_concat, other_labors, month_columns )


# In[166]:


cost_varias = total_cost_block(labor_varias_merged, promediado_varias_id_concat, social_p_varias, 'FINCA', 'ID', month_columns)
cost_varias.iloc[:, 2].sum()


# #### Concatenar labores varias salida MO

# In[167]:


#CONCATENAMOS LABOR VARIAS PARA OBTENER EL GRUPO
labor_varias_merged_output = pd.concat([labor_varias_merged, other_labors[['GRUPO']]], axis=1)
promediado_varias_output = pd.concat([promediado_varias_id_concat, other_labors[['GRUPO']]], axis=1)
social_varias_output  = pd.concat([social_p_varias, other_labors[['GRUPO']]], axis=1)


# In[168]:


#SALIDA PARA LABOR VARIAS
labor_varias_mo = labor_varias_merged_output.groupby(by = ['GRUPO','FINCA']).sum().reset_index()
labor_varias_mo_std = reparar_codificacion(labor_varias_mo, 'FINCA', aplicar_fix_cienaga=True)
labor_varias_mo_order =reorder_output_materials(labor_varias_mo_std , farm_order, 'FINCA', 
                                                    other_labors.GRUPO.unique(), 'GRUPO')

varias_df = labor_varias_mo_order.rename(columns=lambda c: str(c).strip())

#ordena las columnas del dataframe
labor_varias_ouput_mo = varias_df.filter(items=["FINCA", "GRUPO"]).join(
    varias_df.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$')
)


# In[169]:


#SALIDA PARA PROMEDIADO VARIAS
promediado_varias_mo = promediado_varias_output.groupby(by = ['GRUPO','FINCA']).sum().reset_index()
promediado_varias_mo_std = reparar_codificacion(promediado_varias_mo, 'FINCA', aplicar_fix_cienaga=True)
promediado_varias_mo_order =reorder_output_materials(promediado_varias_mo_std , farm_order, 'FINCA', 
                                                    other_labors.GRUPO.unique(), 'GRUPO')

varias_df_2 = promediado_varias_mo_order.rename(columns=lambda c: str(c).strip())

#ordena las columnas del dataframe
promediado_varias_ouput_mo = varias_df_2.filter(items=["FINCA", "GRUPO"]).join(
    varias_df_2.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$')
)


# In[170]:


#SALIDA PARA SOCIAL VARIAS
social_varias_mo = social_varias_output.groupby(by = ['GRUPO','FINCA']).sum().reset_index()
social_varias_mo_std = reparar_codificacion(social_varias_mo, 'FINCA', aplicar_fix_cienaga=True)
social_varias_mo_order =reorder_output_materials(social_varias_mo_std , farm_order, 'FINCA', 
                                                    other_labors.GRUPO.unique(), 'GRUPO')

varias_df_3 = social_varias_mo_order.rename(columns=lambda c: str(c).strip())

#ordena las columnas del dataframe
social_varias_ouput_mo = varias_df_3.filter(items=["FINCA", "GRUPO"]).join(
    varias_df_3.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$')
)


# In[171]:


cost_varias_ouput_mo = total_cost_block(labor_varias_ouput_mo, promediado_varias_ouput_mo, social_varias_ouput_mo, 'FINCA', 'GRUPO', month_columns)
cost_varias_ouput_mo.iloc[:, 2].sum()


# In[172]:


#SALIDA DE LABORES VARIAS CONCATENADA
final_varias_labor_mo = pd.concat([labor_varias_ouput_mo, promediado_varias_ouput_mo.iloc[:,2:],  
                                  social_varias_ouput_mo.iloc[:,2:], cost_varias_ouput_mo.iloc[:,2:]], axis = 1)


# ## SALIDA MO

# In[173]:


salida_mo = pd.concat([corte_empaque, final_parcela_output_mo, final_ciclics_labor_mo, 
                       final_fijas_labor_mo, final_varias_labor_mo], axis = 0)


salida_mo.iloc[:, -1].sum()


# In[174]:


final_fijas_labor_mo.GRUPO.value_counts()


# In[175]:


salida_mo.to_csv('salida_mo.csv', sep = ';', index = False)


# In[176]:


salida_mo.GRUPO.value_counts()


# # GASTOS DE PERSONAL

# In[177]:


#obtenemos la serie de pandas con el factor fuerza laboral
fl = fuerza_laboral.FUERZA_LABORAL

#nos quedamos con las columnas de meses unicamente 
ha_gastos_subset = ha_cultiv.iloc[:, 2:]

#igualamos las columnas de meses para evitar discrepancias 
mask_columns = volum_distribution_matrix.columns
ha_gastos_subset.columns = mask_columns


# In[178]:


fuerza_laboral_matrix  = calculate_volume_distribution_gastos(ha_gastos_subset, 
                                                              volum_distribution_matrix, 
                                                              volum_file_subset_10002, month_columns, fl)

fuerza_laboral_matrix_concat = pd.concat([fuerza_laboral[['FINCA']], fuerza_laboral_matrix, ], axis = 1)

gastos_personal_std = reparar_codificacion(gastos_personal, 'DETALLE', aplicar_fix_cienaga=False)


# In[179]:


factor_at = gastos_personal[gastos_personal['DETALLE'] == 'Auxilio de Transporte'].PARAMETROS
factor_tp = gastos_personal[gastos_personal['DETALLE'] == 'Transporte de personal'].PARAMETROS
factor_dot = gastos_personal[gastos_personal['DETALLE'] == 'Dotaciones'].PARAMETROS
factor_hrs_recreacion = gastos_personal[gastos_personal['DETALLE'] == 'Hras capacitacion y recreacion'].PARAMETROS
factor_capacitacion = gastos_personal[gastos_personal['DETALLE'] == 'Otras capacitaciones'].PARAMETROS
factor_family = gastos_personal[gastos_personal['DETALLE'] == 'Dia dela familia'].PARAMETROS
factor_ax_varios = gastos_personal[gastos_personal['DETALLE'] == 'Auxilios varios'].PARAMETROS
factor_bono = gastos_personal[gastos_personal['DETALLE'] == 'Bono por vacaciones'].PARAMETROS
factor_bienestar = gastos_personal[gastos_personal['DETALLE'] == 'Actividades de Bienestar'].PARAMETROS 
factor_firma = gastos_personal[gastos_personal['DETALLE'] == 'Firma de pacto + Fin de ano'].PARAMETROS 
factor_incapacidad = gastos_personal[gastos_personal['DETALLE'] == 'Incapacidades'].PARAMETROS 
factor_indemnizaciones = gastos_personal[gastos_personal['DETALLE'] == 'Indemnizaciones'].PARAMETROS 
factor_sgsst = gastos_personal[gastos_personal['DETALLE'] == 'SGSST'].PARAMETROS 
factor_deporte = gastos_personal[gastos_personal['DETALLE'] == 'Gastos deportivos'].PARAMETROS 
factor_conductores = gastos_personal[gastos_personal['DETALLE'] == 'Conductores'].PARAMETROS 
factor_buses = gastos_personal[gastos_personal['DETALLE'] == 'Mantenimiento de buses'].PARAMETROS


# In[180]:


param_aux_trans = parametro_1(fuerza_laboral_matrix_concat, factor_at, ceil=False)
param_trans_pers =parametro_2(fuerza_laboral_matrix_concat, factor_tp, month_columns)
param_dotacion =parametro_3(fuerza_laboral_matrix_concat, factor_dot, month_columns)
param_hras_recreacion =parametro_4(fuerza_laboral_matrix_concat, factor_hrs_recreacion, factor_gastos)
param_capacitacion = parametro_5(fuerza_laboral_matrix_concat, factor_capacitacion, 11000, 25000, 300000, ceil=False)
param_family = parametro_6(fuerza_laboral_matrix_concat, factor_family)
param_ax_varios = parametro_1(fuerza_laboral_matrix_concat, factor_ax_varios, ceil=False)
param_pactos = parametro_8(fuerza_laboral_matrix_concat)
param_bono = parametro_1(fuerza_laboral_matrix_concat, factor_bono, ceil=False)
param_bienestar = parametro_9(fuerza_laboral_matrix_concat, factor_bienestar, 10000, 100000, 15000, 10000, ceil=False)
param_firma = parametro_10(fuerza_laboral_matrix_concat, factor_firma)
param_incapacidad = parametro_1(fuerza_laboral_matrix_concat, factor_incapacidad, ceil=False)
param_indemnizacion = parametro_1(fuerza_laboral_matrix_concat, factor_indemnizaciones, ceil=False)
param_sgsst = parametro_11(fuerza_laboral_matrix_concat, factor_sgsst, ceil=False)
param_deporte = parametro_2(fuerza_laboral_matrix_concat, factor_deporte, month_columns)
param_conductores = parametro_1(fuerza_laboral_matrix_concat, factor_conductores, ceil=False)
param_buses = parametro_1(fuerza_laboral_matrix_concat, factor_buses, ceil=False)


# In[181]:


param_columns = param_aux_trans.columns 
param_dotacion.columns = param_columns


# In[182]:


#concatenamos todas las fincas de todos los tipos de gastos

final_gastos_personal = pd.concat([param_aux_trans, param_trans_pers, param_dotacion,
                                   param_hras_recreacion, param_capacitacion, param_family,
                                   param_ax_varios, param_pactos, param_bono, 
                                   param_bienestar, param_firma, param_incapacidad, 
                                   param_indemnizacion, param_sgsst, param_deporte, 
                                   param_conductores, param_buses], axis = 0)

final_gastos_personal['TOTAL'] = final_gastos_personal.select_dtypes(include="number").sum(axis=1)


# In[183]:


#ahora concatenamos el detalle de gastos de personal donde se especifica el tipo de gasto
final_gastos_personal_detalle = pd.concat([final_gastos_personal.reset_index(drop=True), 
                                           gastos_personal[['DETALLE']].reset_index(drop=True)], axis = 1)


# In[185]:


print(final_gastos_personal_detalle.iloc[:, 2].sum())
final_gastos_personal_detalle.to_csv('gastos_personal_detalle.csv', sep= ';', index=False)
#prueba


# In[ ]:





# 
