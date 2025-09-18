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
from itertools import product
import pandas as pd
from itertools import product
import pandas as pd
from itertools import product
import logging
from unidecode import unidecode
import unicodedata
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def calculate_volume_distribution_factor(
    volum_file_emb_subset_def,
    volum_distribution_subset_def,
    volum_file_emb_transform_def,
    month_columns,
    adjustment_factors  # <- nuevo parámetro
):
    try:
        logger.info("Iniciando el cálculo de la matriz de distribución de volumen.")

        final_data = []

        for month in volum_distribution_subset_def.index:
            monthly_result = []
            for i in range(46):  # 0 a 45 son 46 elementos
                result = (volum_file_emb_subset_def.iloc[i, :] * volum_distribution_subset_def.iloc[month, :]).sum()
                monthly_result.append(result)
            final_data.append(monthly_result)

        final_data_df = pd.DataFrame(final_data)
        
        final_data_df_trans = final_data_df.transpose()
        final_data_df_trans.columns = month_columns
        

        # Aplicar factores de ajuste (multiplicación fila a fila)
        final_data_df_trans = final_data_df_trans.multiply(adjustment_factors, axis=0)
        logger.info("Factores de ajuste aplicados al DataFrame transpuesto.")

        volum_concat = volum_file_emb_transform_def[['FINCA', 'CONCEPTO']].reset_index(drop=True)
        volum_data_emb = pd.concat([volum_concat, final_data_df_trans], axis=1)

        return volum_data_emb

    except Exception as e:
        logger.error(f"Error durante el cálculo de la distribución de volumen: {e}")
        raise

"---------------------------------------------------------------------------------------------------"
def calculate_volume_distribution_gastos(
    volum_file_emb_subset_def: pd.DataFrame,
    volum_distribution_subset_def: pd.DataFrame,
    volum_file_emb_transform_def: pd.DataFrame,  # compat
    month_columns,  # puede ser list, tuple, np.ndarray o pd.Index
    factor_series: pd.Series,
    block_size: int = 46
) -> pd.DataFrame:
    try:
        total = len(volum_file_emb_subset_def)
        if total % block_size:
            logger.warning("Filas no múltiplo de %d; se truncará el excedente.", block_size)
        rows = (total // block_size) * block_size
        if rows == 0:
            raise ValueError("No hay filas completas para el tamaño de bloque especificado.")

        if not isinstance(factor_series, pd.Series) or len(factor_series) != rows:
            raise ValueError(f"factor_series debe ser pd.Series de longitud {rows}.")

        base = volum_file_emb_subset_def.iloc[:rows].astype(float)
        dist = volum_distribution_subset_def.astype(float)

        sums = dist.sum(axis=1).replace(0, np.nan).to_numpy()         # (meses,)
        factors = factor_series.to_numpy().reshape(-1, 1)              # (rows,1)

        # (rows x cols) @ (cols x meses) -> (rows x meses)
        M = base.to_numpy() @ dist.to_numpy().T
        M = (M / sums) * factors                                       # divide por suma del mes y aplica factor por fila
        M = np.round(M)                                                 # redondeo hacia arriba

        # Manejo robusto de columnas (evita "truth value of Index is ambiguous")
        if month_columns is not None and len(month_columns) == dist.shape[0]:
            cols = list(month_columns)
        else:
            cols = list(dist.index)

        out = pd.DataFrame(M, index=base.index, columns=cols)
        return out

    except Exception as e:
        logger.error(f"Error durante el cálculo de la distribución de volumen: {e}")
        raise
"----------------------------------------------------------------------------------------------------"
def calculate_volume_distribution_blocks(
    volum_file_emb_subset_def: pd.DataFrame,
    volum_distribution_subset_def: pd.DataFrame,
    volum_file_emb_transform_def: pd.DataFrame,
    month_columns: list
) -> pd.DataFrame:
    try:
        logger.info("Iniciando cálculo de volumen por bloques de 46 filas.")

        final_data = []
        block_size = 46
        total_rows = volum_file_emb_subset_def.shape[0]

        if total_rows % block_size != 0:
            logger.warning("El número total de filas no es múltiplo de 46. Se truncará el exceso.")
        
        num_blocks = total_rows // block_size

        for month in volum_distribution_subset_def.index:
            monthly_result = []

            for block in range(num_blocks):
                block_start = block * block_size
                block_end = block_start + block_size
                result_block = []

                for i in range(block_start, block_end):
                    result = (volum_file_emb_subset_def.iloc[i, :] * volum_distribution_subset_def.iloc[month, :]).sum()
                    result_block.append(result)

                monthly_result.extend(result_block)

            final_data.append(monthly_result)

        # Convertir a DataFrame y transponer
        final_data_df = pd.DataFrame(final_data).transpose()
        final_data_df.columns = month_columns

        logger.info("Cálculo de matriz de volumen completado correctamente.")
        return final_data_df

    except Exception as e:
        logger.error(f"Error durante el cálculo de la distribución de volumen: {e}")
        raise


"-------------------------------------------------------------------------------------------------------"
def calculate_volume_distribution_blocks_ciclics(
    volum_file_emb_subset_def: pd.DataFrame,
    volum_distribution_subset_def: pd.DataFrame,
    volum_file_emb_transform_def: pd.DataFrame,  # no se usa; se mantiene por compatibilidad
    month_columns: list,
    id_col: str = "ID",
    finca_col: str = "FINCA",
) -> pd.DataFrame:
    try:
        logger.info("Iniciando cálculo de volumen por bloques de 46 filas.")

        # --- util: normalizar nombres de columnas parseables a fecha ---
        def _norm_week_cols(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            rename_map = {}
            for c in df.columns:
                d = pd.to_datetime(str(c), errors="coerce")
                if pd.notna(d):
                    rename_map[c] = d.strftime("%Y-%m-%d")
            return df.rename(columns=rename_map)

        # Guardar meta (ID/FINCA) antes de quedarnos solo con semanas
        base_meta = volum_file_emb_subset_def[[id_col, finca_col]].copy()

        # Normalizar encabezados
        A = _norm_week_cols(volum_file_emb_subset_def)
        D = _norm_week_cols(volum_distribution_subset_def)

        # Detectar columnas semanales (parseables a fecha)
        def _is_date_label(c):
            return pd.notna(pd.to_datetime(str(c), errors="coerce"))
        weeks_A = [c for c in A.columns if _is_date_label(c)]
        weeks_D = [c for c in D.columns if _is_date_label(c)]

        # Alinear por semanas comunes, respetando el orden de A
        common_weeks = [c for c in weeks_A if c in weeks_D]
        if not common_weeks:
            raise ValueError("No hay semanas en común entre ambos dataframes.")

        if len(common_weeks) < len(weeks_A):
            logger.warning(f"Semanas de A sin pesos en D (se ignoran): "
                           f"{[c for c in weeks_A if c not in weeks_D]}")
        if len(common_weeks) < len(weeks_D):
            logger.warning(f"Semanas de D sin datos en A (se ignoran): "
                           f"{[c for c in weeks_D if c not in weeks_A]}")

        A = A[common_weeks].apply(pd.to_numeric, errors="coerce").fillna(0.0)   # (N x W)
        D = D[common_weeks].apply(pd.to_numeric, errors="coerce").fillna(0.0)   # (M x W)

        # Config de bloques
        final_data = []
        block_size = 46
        total_rows = A.shape[0]
        if total_rows % block_size != 0:
            logger.warning("El número total de filas no es múltiplo de 46. Se truncará el exceso.")
        num_blocks = total_rows // block_size
        n_used = num_blocks * block_size  # filas efectivamente usadas

        # Numpy para acelerar
        A_np = A.to_numpy()                       # (N x W)
        D_np = D.to_numpy()                       # (M x W)
        n_months = D_np.shape[0]

        for m in range(n_months):
            monthly_result = []
            w = D_np[m, :]                        # (W,)

            for b in range(num_blocks):
                s = b * block_size
                e = s + block_size
                # (46 x W) @ (W,) → (46,)
                block_vals = A_np[s:e, :] @ w
                monthly_result.extend(block_vals.tolist())

            final_data.append(monthly_result)

        # Salida
        final_df = pd.DataFrame(final_data).transpose()  # (n_used x M)

        if len(month_columns) != n_months:
            logger.warning(f"month_columns ({len(month_columns)}) != filas de pesos ({n_months}). "
                           f"Se numerarán 0..{n_months-1}.")
            month_columns = list(range(n_months))
        final_df.columns = month_columns

        # --- Adjuntar ID y FINCA al inicio (solo las filas usadas) ---
        meta_used = base_meta.iloc[:n_used].reset_index(drop=True)
        final_df = final_df.reset_index(drop=True)
        final_df.insert(0, finca_col, meta_used[finca_col].values)
        final_df.insert(0, id_col,    meta_used[id_col].values)

        # opcional: para conservar el índice original en las filas usadas
        # final_df.index = volum_file_emb_subset_def.index[:n_used]

        logger.info("Cálculo de matriz de volumen completado correctamente.")
        return final_df

    except Exception as e:
        logger.error(f"Error durante el cálculo de la distribución de volumen: {e}")
        raise

"------------------------------------------------------------------------------------------------------"
def quantity_fijas(
    df1: pd.DataFrame,
    fincas_df: pd.DataFrame,
    month: list,                
    mes_col: str = "SEMANA",
    mult_cols=("FACTOR",),
    id_mode: str = "str",        # 'str' | 'zfill' | 'int'  ← NUEVO
    id_width: int | None = None  # ancho para 'zfill'       ← NUEVO
) -> pd.DataFrame:
    """
    Suma por mes las filas de df1 (según mes_col) y multiplica por los factores de fincas_df.
    - 'month' es la lista (longitud 12) con las etiquetas de salida (ej. ["2024-01", ..., "2024-12"]).
    - 'mult_cols' acepta string o iterable con nombres de columnas en fincas_df.
    - 'id_mode':
        * 'str'   -> ID como texto limpio (recomendado si en el otro DF es object/string).
        * 'zfill' -> ID como texto con ceros a la izquierda (usa id_width o lo infiere).
        * 'int'   -> ID numérico (Int64).
    """

    # Helper interno: normaliza ID
    def _normalize_id(s: pd.Series, mode="str", width=None) -> pd.Series:
        # limpia espacios (incluye NBSP) y quita sufijo '.0' típico de Excel
        out = (s.astype("string")
                 .str.replace("\u00a0", " ", regex=False)
                 .str.strip()
                 .str.replace(r"\.0+$", "", regex=True))
        if mode == "int":
            return pd.to_numeric(out, errors="coerce").astype("Int64")
        if mode == "zfill":
            # conserva NaN sin convertirlos a '000...'
            na = out.isna()
            w = int(width) if width else max(1, int(out.dropna().str.len().max() or 1))
            out = out.fillna("").str.zfill(w)
            return out.mask(na)
        # modo 'str'
        return out

    # Mapeo ES->orden pasado
    meses_es = [
        "ENERO","FEBRERO","MARZO","ABRIL","MAYO","JUNIO",
        "JULIO","AGOSTO","SEPTIEMBRE","OCTUBRE","NOVIEMBRE","DICIEMBRE"
    ]
    mapa = dict(zip(meses_es, month))

    # --- 1) Mensualizar df1 ---
    d = df1.copy()
    d.columns = d.columns.astype(str).str.strip()
    etiquetas = (
        d[mes_col]
        .astype(str).str.strip().str.upper()
        .replace(mapa)
    )
    valores = (
        d.drop(columns=[mes_col])
         .apply(pd.to_numeric, errors="coerce")
         .fillna(0.0)
         .sum(axis=1)
    )
    base = (
        valores.groupby(etiquetas)
        .sum()
        .reindex(month, fill_value=0.0)
        .values
        .astype(float)
    )  # shape: (12,)

    # --- 2) Multiplicadores por (FINCA, ID) ---
    g = fincas_df.copy()
    g.columns = g.columns.astype(str).str.strip()
    g["FINCA"] = g["FINCA"].astype(str).str.strip()
    # ←← Normalización de ID
    g["ID"] = _normalize_id(g["ID"], mode=id_mode, width=id_width)

    # aceptar string o lista/tupla
    mult_cols = [mult_cols] if isinstance(mult_cols, str) else list(mult_cols)
    colmap = {c.lower(): c for c in g.columns}
    mult = pd.Series(1.0, index=g.index, dtype=float)
    for req in mult_cols:
        key = str(req).strip().lower()
        if key not in colmap:
            raise KeyError(f"'{req}' no existe en fincas_df. Columnas: {list(g.columns)}")
        mult *= pd.to_numeric(g[colmap[key]], errors="coerce").fillna(0.0)

    # --- 3) Construir salida ---
    vals = mult.to_numpy()[:, None] * base[None, :]  # (n_filas x 12)
    out = pd.DataFrame(vals, columns=month)
    out.insert(0, "ID", g["ID"].values)
    out.insert(0, "FINCA", g["FINCA"].values)
    return out
"--------------------------------------------------------------------------------------------------------"
def multiply_months_by_price_ciclics(
    df_months: pd.DataFrame,   # df1: ID, FINCA, columnas de meses/fechas
    df_tarifas: pd.DataFrame,  # df2: ID, ..., TARIFA
    id_col: str = "ID",
    tarifa_col: str = "TARIFA",
    finca_col: str = "FINCA",
    month_cols: list | None = None,
    fillna_tarifa: float = 0.0,     # si falta tarifa para un ID, usa este valor
    keep_tarifa_col: bool = False,  # si True, deja una columna TARIFA en la salida
) -> pd.DataFrame:
    out = df_months.copy()

    # 1) Definir columnas de meses/fechas si no se pasan
    if month_cols is None:
        month_cols = [c for c in out.columns if c not in {id_col, finca_col}]

    # 2) Mapa ID -> TARIFA
    tarifas = (df_tarifas[[id_col, tarifa_col]]
               .drop_duplicates(subset=[id_col], keep="first")
               .set_index(id_col)[tarifa_col])

    # 3) Traer tarifa por ID
    out["_tarifa_tmp"] = out[id_col].map(tarifas)
    out["_tarifa_tmp"] = pd.to_numeric(out["_tarifa_tmp"], errors="coerce").fillna(fillna_tarifa)

    # 4) A numérico meses/fechas y multiplicar por fila
    out[month_cols] = out[month_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out[month_cols] = out[month_cols].mul(out["_tarifa_tmp"], axis=0)

    # 5) Opcional: dejar o quitar la tarifa
    if keep_tarifa_col:
        out[tarifa_col] = out["_tarifa_tmp"]
    out = out.drop(columns=["_tarifa_tmp"])

    return out

"--------------------------------------------------------------------------------------------------------"
def quantity_other_labors(df1: pd.DataFrame,
                          df2: pd.DataFrame,
                          meses_ordenados: list,
                          factor=None,
                          multiplicar_por_s: bool = True) -> pd.DataFrame:
    meses = [str(m)[:7] for m in meses_ordenados]
    orden_fincas = df1['FINCA'].drop_duplicates().tolist()

    # Normaliza nombres a 'YYYY-MM-DD' (salvo claves)
    def _ymd(c):
        if c in ('CONCEPTO','FINCA','MES'): return c
        t = pd.to_datetime(str(c), errors='coerce')
        return str(c) if pd.isna(t) else t.strftime('%Y-%m-%d')
    df1 = df1.rename(columns={c:_ymd(c) for c in df1.columns}).copy()
    df2 = df2.rename(columns={c:_ymd(c) for c in df2.columns}).copy()

    # Asegura 'MES' en df2 (columna / heurística / índice)
    if 'MES' not in df2.columns:
        up = lambda s: s.astype(str).str.strip().str.upper()
        meses_es = {'ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO','JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE'}
        cand = next((c for c in df2.columns if str(c).strip().upper()=='MES'), None)
        if cand: df2 = df2.rename(columns={cand:'MES'})
        else:
            col = next((c for c in df2.columns if up(df2[c]).isin(meses_es).mean()>=.8), None)
            if col: df2 = df2.rename(columns={col:'MES'})
            elif up(pd.Index(df2.index)).isin(meses_es).mean()>=.8:
                df2 = df2.reset_index().rename(columns={df2.columns[0]:'MES'})
            else:
                raise KeyError("df2 debe contener columna 'MES' (ENERO..DICIEMBRE).")

    # Semanas comunes + agregación
    semanas = sorted((set(df1.columns)-{'CONCEPTO','FINCA'}) & (set(df2.columns)-{'MES'}))
    df1_ag = df1[['FINCA']+semanas].groupby('FINCA', as_index=False, sort=False).sum(numeric_only=True)

    # MES ES -> 'MM'
    mapa = {'ENERO':'01','FEBRERO':'02','MARZO':'03','ABRIL':'04','MAYO':'05','JUNIO':'06','JULIO':'07','AGOSTO':'08','SEPTIEMBRE':'09','OCTUBRE':'10','NOVIEMBRE':'11','DICIEMBRE':'12'}
    df2['_MES_NUM'] = df2['MES'].astype(str).str.strip().str.upper().map(mapa)

    # Adaptador de factor (Serie -> vector por FINCA para el mes actual)
    def _fvec(mes:str) -> np.ndarray:
        n = len(orden_fincas)
        if factor is None: return np.ones(n)
        if np.isscalar(factor): return np.full(n, float(factor))
        if isinstance(factor, pd.Series):
            mm = mes[5:7]
            idx = factor.index
            if isinstance(idx, pd.MultiIndex) and idx.nlevels==2:
                for lev in (0,1):
                    try:
                        uf = factor.unstack(lev)
                        for key in (mes, mm):
                            cols = [c for c in uf.columns if str(c)==str(key)]
                            if cols:
                                return uf[cols[0]].reindex(orden_fincas).fillna(1.0).astype(float).to_numpy()
                    except: pass
            if any(f in factor.index for f in orden_fincas):
                return factor.reindex(orden_fincas).fillna(1.0).astype(float).to_numpy()
            if mes in factor.index or mm in factor.index:
                val = float(factor.get(mes, factor.get(mm, 1.0))); return np.full(n, val)
            if len(factor)==n and pd.api.types.is_integer_dtype(factor.index):
                return factor.reset_index(drop=True).fillna(1.0).astype(float).to_numpy()
        return np.ones(len(orden_fincas))

    # Cálculo
    res = pd.DataFrame({'FINCA': df1_ag['FINCA']})
    for mes in meses:
        pos = df2.index[df2['_MES_NUM']==mes[5:7]]
        if not len(pos): res[mes]=0.0; continue
        fila = df2.loc[pos[0]]
        pref = mes+'-'
        sem_mes = sorted({c for c in semanas if c.startswith(pref)} |
                         {c for c in semanas if pd.to_numeric(fila.get(c,0), errors='coerce')>0})
        if not sem_mes: res[mes]=0.0; continue

        w = fila[sem_mes].astype(float).fillna(0.0).to_numpy()
        s = w.sum()
        if s==0: res[mes]=0.0; continue

        v = df1_ag[sem_mes].astype(float).fillna(0.0).to_numpy()
        promedio = v.dot(w)/s                              # ← 170
        res[mes] = promedio * _fvec(mes) * (s if multiplicar_por_s else 1.0)

    return res.set_index('FINCA').reindex(orden_fincas).reset_index()[['FINCA']+meses]

"--------------------------------------------------------------------------------------------------------"

def build_factor_quantity(curvas_c_mensual: pd.DataFrame,
                          other_labors: pd.DataFrame,
                          id_value: int = 1100001) -> pd.DataFrame:
    """
    Toma curvas_c_mensual y other_labors, filtra por ID, multiplica columnas mensuales
    por FACTOR fila a fila, excluye columnas no requeridas y reordena FINCA primero.
    """
    # 1) Filtrar por ID y hacer merge
    factor_df = other_labors[other_labors['ID'] == id_value]
    merged = curvas_c_mensual.merge(factor_df, on='CURVA', how='inner')

    # 2) Seleccionar columnas 1..12 (asumiendo son los meses) y multiplicar por FACTOR
    curvas_columns = merged.columns[1:13]
    merged.loc[:, curvas_columns] = (
        merged.loc[:, curvas_columns].apply(pd.to_numeric, errors='coerce')
        .mul(pd.to_numeric(merged['FACTOR'], errors='coerce'), axis=0)
    )

    # 3) Excluir columnas no necesarias
    cols_excluir = ['FACTOR', 'PROMEDIADO', 'MAESTRA', 'CURVA', 'ID', 'PRESTACIONES', 'GRUPO']
    out = merged.loc[:, ~merged.columns.isin(cols_excluir)]

    # 4) Detectar columnas mes (YYYY-MM) y reordenar: FINCA + meses
    month_cols = [c for c in out.columns
                  if isinstance(c, str) and pd.to_datetime(c, format='%Y-%m', errors='coerce') is not pd.NaT]

    return out[['FINCA', *month_cols]]

"--------------------------------------------------------------------------------------------------------"
def calculate_promediados_factor(
    volum_distribution_subset_def,
    volum_file_emb_transform_def,
    month_column,
    adjustment_factors  # <- debe tener 92 elementos
):
    try:
        logger.info("Iniciando el cálculo de la matriz de distribución de volumen con 92 factores.")

        if len(adjustment_factors) != 92:
            raise ValueError("La serie de factores debe contener exactamente 92 elementos.")

        final_data = []

        for month in volum_distribution_subset_def.index:
            logger.info(f"Procesando el mes: {month}")
            monthly_result = []

            # Primer bloque (0–45)
            for i in range(46):
                factor = adjustment_factors[i]
                result = (factor * volum_distribution_subset_def.iloc[month, :]).sum()
                monthly_result.append(result)

            # Segundo bloque (46–91)
            for i in range(46, 92):
                factor = adjustment_factors[i]
                result = (factor * volum_distribution_subset_def.iloc[month, :]).sum()
                monthly_result.append(result)

            final_data.append(monthly_result)

        final_data_df = pd.DataFrame(final_data)  # shape: (n_months, 92)
        logger.info("Matriz de resultados generada.")

        final_data_df_trans = final_data_df.transpose()  # shape: (92, n_months)
        final_data_df_trans.columns = month_column
        logger.info("Matriz de resultados transpuesta y columnas renombradas.")

        volum_concat_base = volum_file_emb_transform_def[['FINCA', 'CONCEPTO']].reset_index(drop=True)
        volum_concat = pd.concat([volum_concat_base, volum_concat_base], ignore_index=True)
        volum_data_emb = pd.concat([volum_concat, final_data_df_trans], axis=1)
        
        logger.info("DataFrame final concatenado con los datos de volumen transformados.")

        return volum_data_emb

    except Exception as e:
        logger.error(f"Error durante el cálculo de la distribución de volumen: {e}")
        raise





"---------------------------------------------------------------------------------------------------------"
def ciclics_labor_calculation(df_months: pd.DataFrame, df_tarifas: pd.DataFrame) -> pd.DataFrame:
    out = df_months.copy()

    # Normalización mínima para las llaves
    def _norm(s: pd.Series) -> pd.Series:
        return (s.astype(str)
                 .str.strip()
                 .str.replace(r"\.0$", "", regex=True))

    # Determinar clave: ID + FINCA si ambas existen
    key = ['ID']
    if 'FINCA' in df_months.columns and 'FINCA' in df_tarifas.columns:
        key.append('FINCA')

    # Normalizar claves en ambos DF
    for c in key:
        out[c] = _norm(out[c])
        df_tarifas[c] = _norm(df_tarifas[c])

    # Tomar TARIFA por clave exacta (sin colapsar por solo ID)
    tarifas = df_tarifas[key + ['TARIFA']].drop_duplicates(subset=key, keep='first').copy()
    tarifas['TARIFA'] = pd.to_numeric(tarifas['TARIFA'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)

    # Unir y multiplicar
    out = out.merge(tarifas, on=key, how='left')
    excluir = set(key + ['TARIFA'])
    month_cols = [c for c in out.columns if c not in excluir]

    out[month_cols] = out[month_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    out[month_cols] = (out[month_cols].T * out['TARIFA']).T

    return out.drop(columns=['TARIFA'])

"---------------------------------------------------------------------------------------------------------"
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
"----------------------------------------------------------------------------------------------------------"
def multiply_by_month_promediado_ciclics(df1, df2, months):
    try:
        # Normaliza nombres de columnas
        df1 = df1.copy()
        df2 = df2.copy()
        df1.columns = df1.columns.astype(str).str.strip()
        df2.columns = df2.columns.astype(str).str.strip()
        months = [str(m) for m in months]

        # Convierte a numérico solo si la columna existe
        for m in months:
            if m in df1.columns:
                df1[m] = pd.to_numeric(df1[m], errors='coerce')
            if m in df2.columns:
                df2[m] = pd.to_numeric(df2[m], errors='coerce')

        # Merge por PROMEDIADO
        if 'PROMEDIADO' not in df1.columns or 'PROMEDIADO' not in df2.columns:
            raise KeyError("Falta la columna 'PROMEDIADO' en alguno de los DataFrames.")
        df_merged = pd.merge(df1, df2, on='PROMEDIADO', how='left', suffixes=('_df1', '_df2'))

        # Detecta posible columna multiplicadora si df2 no tiene meses
        candidatas = ['TARIFA', 'FACTOR', 'VALOR', 'COEF', 'MULTIPLICADOR']
        mult_col = None
        for base in candidatas:
            if base in df_merged.columns:
                mult_col = base
                break
            if f"{base}_df2" in df_merged.columns:
                mult_col = f"{base}_df2"
                break
        if mult_col is not None:
            df_merged[mult_col] = pd.to_numeric(df_merged[mult_col], errors='coerce')

        # Multiplicación por mes
        for m in months:
            # Encuentra la columna de df1 tras el merge (con o sin sufijo)
            if m in df_merged.columns:
                col_df1 = m
            elif f"{m}_df1" in df_merged.columns:
                col_df1 = f"{m}_df1"
            else:
                # Si el mes no existe en df1, no hay nada que multiplicar
                continue

            # Determina el factor: mes de df2 o columna multiplicadora única
            if f"{m}_df2" in df_merged.columns:
                factor = pd.to_numeric(df_merged[f"{m}_df2"], errors='coerce')
            elif mult_col is not None:
                factor = df_merged[mult_col]
            else:
                # Ni mes en df2 ni multiplicador único: no se puede calcular este mes
                continue

            df_merged[m] = pd.to_numeric(df_merged[col_df1], errors='coerce').fillna(0.0) * factor.fillna(0.0)

        # Devuelve FINCA + meses si FINCA existe; si no, solo meses
        if 'FINCA' in df_merged.columns:
            cols = ['FINCA'] + [m for m in months if m in df_merged.columns]
        else:
            cols = [m for m in months if m in df_merged.columns]

        result_df = df_merged[cols].copy()
        return result_df

    except Exception as e:
        # Propaga con contexto claro
        raise RuntimeError(f"Error en multiply_by_month_promediado: {e}") from e
"-----------------------------------------------------------------------------------------------------------"
import pandas as pd

def social_p_parcela(df1, df2, escalar, month_columns):
    """
    Suma df1 + df2 por columnas mensuales y multiplica cada fila por un escalar correspondiente.
    
    Parámetros:
    - df1: DataFrame con columnas ['FINCA', 'PROMEDIADO', '2024-01', ..., '2024-12']
    - df2: DataFrame con columnas ['FINCA', '2024-01', ..., '2024-12']
    - escalar: pd.Series de una dimensión con misma longitud que df1 y df2
    - month_columns: lista con nombres de columnas mensuales como ['2024-01', ..., '2024-12']

    Retorna:
    - DataFrame con columnas ['FINCA', 'PROMEDIADO'] + columnas mensuales ajustadas
    """

    try:
        # Normalizar nombres de columnas
        df1.columns = df1.columns.astype(str).str.strip()
        df2.columns = df2.columns.astype(str).str.strip()
        month_columns = [str(col).strip() for col in month_columns]

        # Validar tamaños
        if not (len(df1) == len(df2) == len(escalar)):
            raise ValueError("df1, df2 y escalar deben tener la misma cantidad de filas")

        # Convertir columnas mensuales a numérico si no lo son
        for col in month_columns:
            df1[col] = pd.to_numeric(df1[col], errors='coerce')
            df2[col] = pd.to_numeric(df2[col], errors='coerce')

        # Sumar columnas de meses
        df_sum = df1[month_columns].add(df2[month_columns], fill_value=0)

        # Multiplicar por escalar fila a fila
        df_scaled = df_sum.multiply(escalar, axis=0)

        # Renombrar columnas si se desea (opcional)
        # df_scaled.columns = [f"{col}_ajustado" for col in df_scaled.columns]

        # Combinar con columnas base
        result = pd.concat([df1[['FINCA', 'PROMEDIADO']], df_scaled], axis=1)

        return result

    except Exception as e:
        print(f"Error en ajustar_por_escalar: {e}")
        raise


"-----------------------------------------------------------------------------------------------------------"
"""def standardize_column_dates(df):
    # Intenta convertir las columnas a datetime si es posible
    try:
        new_columns = pd.to_datetime(df.columns, errors='coerce')
        # Si hay fechas nulas (NaT), indica columnas mal formateadas
        if new_columns.isnull().any():
            raise ValueError("Algunas columnas no pudieron convertirse correctamente.")
        df.columns = new_columns
    except Exception as e:
        print(f"Error al estandarizar columnas de fechas: {e}")
    return df"""

def standardize_column_dates(df):
    new_columns = []
    for col in df.columns:
        try:
            # Intentar parsear como fecha (solo si realmente lo es)
            parsed_date = pd.to_datetime(col, dayfirst=True, errors='raise')
            new_columns.append(parsed_date.strftime('%Y-%m-%d'))  # convertir a formato estandarizado
        except Exception:
            new_columns.append(col)  # mantener columna original si no es fecha
    df.columns = new_columns
    return df

"-----------------------------------------------------------------------------------------------------------"
def labores_ciclicas(df: pd.DataFrame, columna_id: str = 'ID', 
                                     columna_finca: str = 'FINCA', columna_factor: str = 'FACTOR') -> pd.DataFrame:
    """
    Multiplica columnas de fechas con sufijos '_x' y '_y' y un FACTOR por fila.
    Conserva solo las columnas de resultado más ID y FINCA.
    
    Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        columna_id (str): Nombre de la columna ID a conservar.
        columna_finca (str): Nombre de la columna FINCA a conservar.
        columna_factor (str): Nombre de la columna que contiene el factor por fila.

    Retorna:
        pd.DataFrame: DataFrame con columnas de resultado + ID + FINCA.
    """
    # Identificar columnas con sufijos _x y _y
    cols_x = sorted([col for col in df.columns if col.endswith('_x')])
    cols_y = sorted([col for col in df.columns if col.endswith('_y')])

    # Validar que el número de columnas coincida
    if len(cols_x) != len(cols_y):
        raise ValueError("El número de columnas '_x' y '_y' no coincide.")

    # Crear nuevas columnas con el resultado de la multiplicación
    for col_x, col_y in zip(cols_x, cols_y):
        new_col = col_x.replace('_x', '')  # Ejemplo: '6/01/2024'
        df[new_col] = df[col_x] * df[col_y] * df[columna_factor]

    # Seleccionar solo columnas necesarias
    columnas_resultado = [col.replace('_x', '') for col in cols_x]
    columnas_finales = [columna_id, columna_finca] + columnas_resultado

    return df[columnas_finales]
"-----------------------------------------------------------------------------------------------------------"
def column_replacement(df, nuevas_columnas): 
    nuevas_columnas = list(nuevas_columnas)
    if len(nuevas_columnas) != df.shape[1]:
        raise ValueError(f"Cantidad no coincide: {len(nuevas_columnas)} vs {df.shape[1]}")
    out = df.copy()
    out.columns = nuevas_columnas
    return out

"-----------------------------------------------------------------------------------------------------------"
def sum_by_months_parcela(df1, df2, months):
    # 1) Quita espacios en la lista de meses y en los nombres de columnas
    months = [str(m).strip() for m in months]
    df1 = df1.copy(); df2 = df2.copy()
    df1.columns = df1.columns.map(lambda c: str(c).strip())
    df2.columns = df2.columns.map(lambda c: str(c).strip())

    # 2) Asegura que existan todas las columnas de months
    for m in months:
        if m not in df1.columns: df1[m] = 0.0
        if m not in df2.columns: df2[m] = 0.0

    # 3) SOLO quitar espacios en celdas de tipo texto (evita usar .str sobre DataFrame)
    df1[months] = df1[months].apply(
        lambda s: s.astype(str).str.replace('\u00a0',' ', regex=False).str.strip()
                  if pd.api.types.is_object_dtype(s) else s
    )
    df2[months] = df2[months].apply(
        lambda s: s.astype(str).str.replace('\u00a0',' ', regex=False).str.strip()
                  if pd.api.types.is_object_dtype(s) else s
    )

    # 4) Tu lógica original
    A = df1[months].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    B = df2[months].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    out = df1[["FINCA","LABOR"]].copy()
    out[months] = A + B
    return out


"-----------------------------------------------------------------------------------------------------------"
def farm_order_process(df, orden_fincas, columna_finca='FINCA', columna_id='ID'):
    # Paso 1: crear una columna auxiliar con el orden de la finca
    finca_to_order = {finca: i for i, finca in enumerate(orden_fincas)}
    df['FINCA_ORDEN'] = df[columna_finca].map(finca_to_order)

    # Paso 2: ordenar primero por ID, luego por el orden de finca
    df = df.sort_values(by=[columna_id, 'FINCA_ORDEN'], ascending=[True, True])

    # Paso 3: eliminar columna auxiliar
    return df.drop(columns='FINCA_ORDEN').reset_index(drop=True)


def farm_order_process_concat(df, orden_fincas, columna_finca='FINCA'):
    # Si FINCA está en el índice, lo pasamos a columna
    if columna_finca not in df.columns and df.index.name == columna_finca:
        df = df.reset_index()

    return (df
            .sort_values(
                by=columna_finca,
                key=lambda s: pd.Categorical(s, categories=orden_fincas, ordered=True),
                kind='mergesort'              # orden estable
            )
            .reset_index(drop=True))



"------------------------------------------------------------------------------------------------------------"
def reorder_output_materials(df, f_order, value, unit_lists, value_group):

    try:
        # Crear una columna temporal para el orden
        df['order'] = df[value].apply(lambda x: f_order.index(x) if x in f_order else len(f_order))
        
        # Ordenar el DataFrame basado en la columna temporal
        df_sorted = df.sort_values('order').drop(columns=['order'])
        
        # Log de información sobre la reordenación exitosa
        logger.info("Reorganización del DataFrame realizada con éxito.")

        data_unit = []

        for unit in unit_lists:
            final_quantity_unit_filtered = df_sorted[df_sorted[value_group] == unit]
            data_unit.append(final_quantity_unit_filtered)
        logger.info("Reorganización de los grupos /unidad realizada con éxito")

        
        output_ferti_2 = pd.DataFrame(pd.concat(data_unit, ignore_index=True))
    
        return output_ferti_2
    
    except Exception as e:
        # Log de la excepción capturada
        logger.info(f"Ocurrió un error durante la reorganización: {e}")
        return None

"------------------------------------------------------------------------------------------------------------"
from datetime import datetime
from itertools import product
import pandas as pd
import logging

def group_by_type(input_df: pd.DataFrame, farms: list) -> pd.DataFrame:
    """
    Agrupa los valores del DataFrame por TIPO y FINCA, asegurando que se incluyan todas las fincas en farms.
    Convierte los nombres de columnas de fechas (desde la tercera hasta la penúltima columna) al tipo Timestamp.

    Args:
        input_df (pd.DataFrame): DataFrame con columnas 'TIPO', 'FINCA' y columnas numéricas de fechas.
        farms (list): Lista de fincas que deben estar presentes en el resultado final.

    Returns:
        pd.DataFrame: DataFrame agrupado con todas las combinaciones TIPO-FINCA y valores numéricos sumados.
    """
    try:
        logging.info("Iniciando limpieza de columnas TIPO y FINCA...")
        input_df['FINCA'] = input_df['FINCA'].astype(str).str.strip()
        input_df['TIPO'] = input_df['TIPO'].astype(str).str.strip()

        logging.info("Renombrando columnas de fechas al tipo Timestamp (posición 3 a penúltima)...")
        fecha_column_map = {}
        target_cols = input_df.columns[2:-1]  # desde la tercera hasta la penúltima
        for col in target_cols:
            try:
                # Convierte a Timestamp
                nueva_fecha = pd.to_datetime(col.split()[0], format="%d/%m/%Y")
                fecha_column_map[col] = nueva_fecha
            except (ValueError, IndexError):
                continue  # Ignora columnas que no se pueden convertir

        input_df.rename(columns=fecha_column_map, inplace=True)

        logging.info("Obteniendo valores únicos de TIPO...")
        tipos_presentes = input_df['TIPO'].unique()

        logging.info("Creando combinaciones TIPO-FINCA completas...")
        combinaciones = pd.DataFrame(list(product(tipos_presentes, farms)), columns=['TIPO', 'FINCA'])

        logging.info("Agrupando por TIPO y FINCA...")
        mo_grouped = input_df.groupby(['TIPO', 'FINCA']).sum(numeric_only=True).reset_index()

        logging.info("Uniendo con combinaciones completas...")
        mo_grouped_completo = combinaciones.merge(mo_grouped, on=['TIPO', 'FINCA'], how='left')

        logging.info("Rellenando NaNs con ceros...")
        mo_grouped_completo.fillna(0, inplace=True)

        logging.info("Proceso de agrupación finalizado exitosamente.")
        return mo_grouped_completo

    except KeyError as e:
        logging.error(f"Columna faltante en el DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado: {e}")
        raise


"--------------------------------------------------------------------------------------------------------------"

def group_by_type_sum(input_df: pd.DataFrame, farms: list, types_sum: list = []) -> pd.DataFrame:
    """
    Agrupa los valores del DataFrame por TIPO y FINCA, asegurando que se incluyan todas las fincas en el orden de `farms`,
    y permite agregar una fila extra con la suma de los TIPO indicados en tipos_a_sumar.

    Args:
        input_df (pd.DataFrame): DataFrame con columnas 'TIPO', 'FINCA' y columnas numéricas de fechas.
        farms (list): Lista de fincas que deben estar presentes en el resultado final.
        tipos_a_sumar (list): Lista de TIPO que deben ser sumados como grupo adicional.

    Returns:
        pd.DataFrame: DataFrame agrupado con todas las combinaciones TIPO-FINCA, incluyendo la suma de los tipos indicados.
    """
    try:
        logging.info("Iniciando limpieza de columnas TIPO y FINCA...")
        input_df['FINCA'] = input_df['FINCA'].astype(str).str.strip()
        input_df['TIPO'] = input_df['TIPO'].astype(str).str.strip()

        logging.info("Obteniendo valores únicos de TIPO...")
        tipos_presentes = input_df['TIPO'].unique()

        logging.info("Creando combinaciones TIPO-FINCA completas...")
        combinaciones = pd.DataFrame(list(product(tipos_presentes, farms)), columns=['TIPO', 'FINCA'])

        logging.info("Agrupando por TIPO y FINCA...")
        mo_grouped = input_df.groupby(['TIPO', 'FINCA']).sum(numeric_only=True).reset_index()

        logging.info("Uniendo con combinaciones completas...")
        mo_grouped_completo = combinaciones.merge(mo_grouped, on=['TIPO', 'FINCA'], how='left')

        logging.info("Rellenando NaNs con ceros...")
        mo_grouped_completo.fillna(0, inplace=True)

        # Asegurar que las fincas estén en el orden definido por 'farms'
        mo_grouped_completo['FINCA'] = pd.Categorical(mo_grouped_completo['FINCA'], categories=farms, ordered=True)
        mo_grouped_completo.sort_values(by=['TIPO', 'FINCA'], inplace=True)

        # Si hay tipos a sumar, agregamos una fila por finca
        if types_sum:
            logging.info(f"Sumando tipos especificados: {types_sum}")
            subset_sum = mo_grouped_completo[mo_grouped_completo['TIPO'].isin(types_sum)]

            suma_tipos = subset_sum.groupby('FINCA').sum(numeric_only=True).reset_index()
            suma_tipos.insert(0, 'TIPO', '+'.join(types_sum))

            # Asegurar el orden también para la suma
            suma_tipos['FINCA'] = pd.Categorical(suma_tipos['FINCA'], categories=farms, ordered=True)
            suma_tipos.sort_values(by='FINCA', inplace=True)

            logging.info("Agregando fila de suma de tipos al DataFrame final...")
            mo_grouped_completo = pd.concat([mo_grouped_completo, suma_tipos], ignore_index=True)

        logging.info("Proceso de agrupación y orden finalizado exitosamente.")
        return mo_grouped_completo

    except KeyError as e:
        logging.error(f"Columna faltante en el DataFrame: {e}")
        raise
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado: {e}")
        raise
"--------------------------------------------------------------------------------------------------------------"
def reparar_codificacion(df, columna, aplicar_fix_cienaga: bool = False):
    """
    Reconvierte texto mal decodificado (ej. 'ALÃ\x8d' → 'ALÍ') y elimina tildes (acentos).
    Si aplicar_fix_cienaga=True, corrige 'CIA‰NAGA' / 'CIÃ‰NAGA' a 'CIENAGA'.
    """
    def fix_encoding_and_remove_accents(texto):
        if pd.isnull(texto):
            return texto

        # Asegurar cadena sin invocar str() (evita conflictos si 'str' fue reasignado)
        s = f"{texto}"

        try:
            # Repara codificación mal leída como UTF-8 en vez de Latin-1
            s = s.encode('latin-1').decode('utf-8')
        except Exception:
            pass  # Si no puede reconvertir, lo deja como está

        # --- FIX opcional para CIENAGA ---
        if aplicar_fix_cienaga:
            s_cmp = s.strip().upper()
            if s_cmp in ("CIA‰NAGA", "CIÃ‰NAGA"):
                s = "CIENAGA"

        # Elimina tildes
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
        return s

    df[columna] = df[columna].apply(fix_encoding_and_remove_accents)
    return df

"--------------------------------------------------------------------------------------------------------------"
import pandas as pd
import re

def multiplicar_mes_por_serie_filas(df: pd.DataFrame,
                                    factor: pd.Series,
                                    month_cols=None,
                                    inplace: bool = False,
                                    na_factor: float = 1.0) -> pd.DataFrame:
    """
    Multiplica las columnas mensuales (YYYY-MM) por una Serie de factores por FILA.
    Alinea por POSICIÓN (no por etiquetas). La longitud de `factor` debe ser igual a len(df).

    Params
    ------
    df: DataFrame de entrada.
    factor: pd.Series de longitud igual a df (p. ej. índices 138..183).
    month_cols: lista opcional de columnas a multiplicar.
    inplace: si True, modifica df en sitio.
    na_factor: reemplazo para NaN en factor (por defecto 1.0 = sin cambio).

    Retorna
    -------
    DataFrame (o el mismo si inplace=True).
    """
    # Detectar columnas mensuales si no se pasan
    if month_cols is None:
        pat = re.compile(r'^\d{4}-\d{2}$')
        month_cols = sorted([c for c in df.columns if pat.match(f"{c}")])

    if not month_cols:
        raise ValueError("No se encontraron columnas mensuales con formato 'YYYY-MM'.")

    if len(factor) != len(df):
        raise ValueError(f"Longitudes distintas: len(factor)={len(factor)} vs len(df)={len(df)}.")

    sacos_result = df if inplace else df.copy()

    # Asegurar numérico y preparar factor por posición
    sacos_result[month_cols] = sacos_result[month_cols].apply(pd.to_numeric, errors='coerce')
    fac_vals = pd.to_numeric(factor, errors='coerce').fillna(na_factor).to_numpy()

    # Multiplicación fila a fila (por posición)
    sacos_result.loc[:, month_cols] = sacos_result.loc[:, month_cols].to_numpy() * fac_vals.reshape(-1, 1)

    return sacos_result
"-------------------------------------------------------------------------------------------------------------"
def quantity_other_labors_800008(
    matriz_semanas: pd.DataFrame,
    farm_order: list[str],
    month_columns: list[str],
    factor,                      # escalar o Serie/array de longitud = len(farm_order)
    semana_col: str = "SEMANA"
) -> pd.DataFrame:
    """
    1) Suma cada fila (mes) de `matriz_semanas` en las columnas de semanas.
    2) Transpone al orden de `month_columns`.
    3) Crea matriz (FINCA x month_columns) llenando con 1.0 y multiplica por la serie mensual.
    4) Multiplica cada fila por `factor` desde la col 1 (FINCA queda intacta), por POSICIÓN.
       - Si `factor` es escalar -> aplica a todas las filas.
       - Si es Serie/array -> debe tener len(farm_order). Se ignoran etiquetas de índice.
    Devuelve: DataFrame con 'FINCA' + month_columns (shape: len(farm_order) × (1+len(month_columns))).
    """
    if semana_col not in matriz_semanas.columns:
        raise ValueError(f"'{semana_col}' no existe en matriz_semanas.")

    # 1) Sumar por fila (mes)
    sumas_por_mes = (
        matriz_semanas
        .drop(columns=[semana_col])
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
    )
    if len(sumas_por_mes) != len(month_columns):
        raise ValueError(
            f"meses en matriz={len(sumas_por_mes)} vs month_columns={len(month_columns)}"
        )

    # 2) Serie mensual indexada por month_columns
    serie_mensual = pd.Series(sumas_por_mes.to_numpy(), index=month_columns, dtype="float64")

    # 3) Matriz base FINCA x meses con 1.0 y escalada por columnas
    base = pd.DataFrame({"FINCA": farm_order})
    for m in month_columns:
        base[m] = 1.0
    base.loc[:, month_columns] = base.loc[:, month_columns].multiply(serie_mensual, axis="columns")

    # 4) Multiplicar por factor por FILA (desde col 1), alineando por POSICIÓN
    out = base.copy(deep=True)
    if np.isscalar(factor):
        out.iloc[:, 1:] = out.iloc[:, 1:].multiply(float(factor))
    else:
        fac_arr = np.asarray(factor, dtype=float)
        if fac_arr.shape[0] != len(farm_order):
            raise ValueError(f"len(factor)={fac_arr.shape[0]} debe igualar len(farm_order)={len(farm_order)}")
        out.iloc[:, 1:] = out.iloc[:, 1:].multiply(fac_arr, axis=0)  # posición, no etiquetas

    return out

"--------------------------------------------------------------------------------------------------------------"
def vlookup_aprox_value(valor, esquema, tabla_esquemas, campo_retorno='VALOR'):
    tabla_filtrada = tabla_esquemas[tabla_esquemas['ESQUEMA'].str.lower() == esquema.lower()]
    tabla_filtrada = tabla_filtrada.sort_values(by='VAL')
    coincidentes = tabla_filtrada[tabla_filtrada['VAL'] <= valor]
    if not coincidentes.empty:
        return coincidentes.iloc[-1][campo_retorno]
    else:
        return np.nan
    
def vlookup_function(df_base, tabla_esquemas, campo_retorno='VALOR'):
    resultado = pd.DataFrame(index=df_base.index)

    if 'ESQUEMA' not in df_base.columns:
        raise ValueError("La columna 'ESQUEMA' no existe en df_base")

    fecha_cols = df_base.columns.drop('ESQUEMA')

    for idx in df_base.index:
        esquema = df_base.loc[idx, 'ESQUEMA']
        for fecha in fecha_cols:
            valor = df_base.loc[idx, fecha]
            resultado.loc[idx, fecha] = vlookup_aprox_value(valor, esquema, tabla_esquemas, campo_retorno)

    resultado['ESQUEMA'] = df_base['ESQUEMA']
    return resultado

"--------------------------------------------------------------------------------------------------------------"

def merge_factor_by_id(id_value: int, df_type: pd.DataFrame, df_factor: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra las fincas del DataFrame `cut_emp` por el ID dado, limpia tildes en la columna FINCA,
    y realiza un merge con el factor correspondiente para añadir la columna FACTOR. Si no hay FACTOR, lo llena con 0.

    Args:
        id_value (int): ID a filtrar en cut_emp.
        mo_grouped_total (pd.DataFrame): DataFrame base a enriquecer.
        cut_emp (pd.DataFrame): DataFrame con columnas 'ID', 'FACTOR', 'FINCA'.

    Returns:
        pd.DataFrame: DataFrame combinado con columna FACTOR asociada a cada FINCA.
    """
    try:
        # Limpieza de tildes en FINCA
        df_factor.loc[:, 'FINCA'] = df_factor['FINCA'].astype(str).apply(unidecode)

        # Filtrar por ID
        cut_filtered = df_factor[df_factor['ID'] == id_value]

        # Hacer merge
        result = pd.merge(df_type, cut_filtered[['FACTOR', 'FINCA']], how='left', on='FINCA')

        # Rellenar valores nulos con 0
        result['FACTOR'] = result['FACTOR'].fillna(0)

        return result

    except KeyError as e:
        logging.error(f"Columna faltante: {e}")
        raise
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        raise



def multipy_factor_combined(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """
    Filtra el DataFrame por un tipo específico y multiplica las columnas numéricas semanales por el valor de FACTOR.

    Args:
        df (pd.DataFrame): DataFrame que contiene columnas 'TIPO', 'FINCA', 'FACTOR' y columnas numéricas.
        tipo (str): Valor de la columna 'TIPO' a filtrar (por ejemplo 'PRE').

    Returns:
        pd.DataFrame: DataFrame filtrado y ajustado por el FACTOR.
    """
    try:
        logging.info(f"Filtrando por TIPO = {tipo}")
        df_tipo = df[df['TIPO'] == tipo].copy()

        # Identificar columnas numéricas que deben multiplicarse
        weekly_columns = df_tipo.columns.difference(['TIPO', 'FINCA', 'FACTOR'])

        logging.info("Aplicando FACTOR a columnas numéricas...")
        df_tipo.loc[:, weekly_columns] = df_tipo[weekly_columns].multiply(df_tipo['FACTOR'], axis=0)

        return df_tipo

    except KeyError as e:
        logging.error(f"Columna faltante: {e}")
        raise
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        raise



"-------------------------------------------------------------------------------------------------------------"
def multipy_factor_combined_100021(df: pd.DataFrame, tipo: str, value: int) -> pd.DataFrame:
    """
    Filtra el DataFrame por un tipo específico y multiplica las columnas numéricas semanales por el valor de FACTOR.

    Args:
        df (pd.DataFrame): DataFrame que contiene columnas 'TIPO', 'FINCA', 'FACTOR' y columnas numéricas.
        tipo (str): Valor de la columna 'TIPO' a filtrar (por ejemplo 'PRE').

    Returns:
        pd.DataFrame: DataFrame filtrado y ajustado por el FACTOR.
    """
    try:
        logging.info(f"Filtrando por TIPO = {tipo}")
        df_tipo = df[df['TIPO'] == tipo].copy()

        # Identificar columnas numéricas que deben multiplicarse
        weekly_columns = df_tipo.columns.difference(['TIPO', 'FINCA', 'FACTOR'])

        logging.info("Aplicando FACTOR a columnas numéricas...")
        df_tipo.loc[:, weekly_columns] = df_tipo[weekly_columns].multiply(value / 960 * df_tipo['FACTOR'], axis=0)

        return df_tipo

    except KeyError as e:
        logging.error(f"Columna faltante: {e}")
        raise
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        raise








"--------------------------------------------------------------------------------------------------------------"
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

"-----------------------------------------------------------------------------------------------"

import pandas as pd
import logging

def multiply_p_social_block(labor_df, promediado_df, full_matrix_df, month_columns):
    """
    Suma los valores mensuales entre dos DataFrames y los multiplica por 'PRESTACIONES'
    según un DataFrame maestro.

    Args:
        labor_df (pd.DataFrame): DataFrame con columnas mensuales, 'FINCA' e 'ID'
        promediado_df (pd.DataFrame): Segundo DataFrame con mismas columnas mensuales
        full_matrix_df (pd.DataFrame): DataFrame con columnas ['FINCA', 'PRESTACIONES', 'ID']
        month_columns (list): Lista de columnas de meses (strings como '2024-01', etc.)

    Returns:
        pd.DataFrame: DataFrame con las columnas sumadas y multiplicadas por PRESTACIONES
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info("Inicia la función de suma y multiplicación con PRESTACIONES.")

        # Asegurar que los nombres de las columnas sean strings
        month_columns = [str(m).strip() for m in month_columns]

        # Convertir columnas mensuales a numérico (por si hay errores o strings)
        labor_df[month_columns] = labor_df[month_columns].apply(pd.to_numeric, errors='coerce')
        promediado_df[month_columns] = promediado_df[month_columns].apply(pd.to_numeric, errors='coerce')

        # Crear DataFrame de suma con 'FINCA' e 'ID'
        df_sum = labor_df[['FINCA', 'ID']].copy()
        df_sum[month_columns] = labor_df[month_columns].add(
            promediado_df[month_columns], fill_value=0
        )

        # Eliminar columna de índice si es necesario (como 'ID' inicial)
        df_sum = df_sum.iloc[:, 1:]  # elimina la primera columna 

        logger.info("Suma realizada con éxito.")

        # Merge con full_matrix para obtener PRESTACIONES
        df_sum_merged = pd.merge(
            df_sum,
            full_matrix_df[['FINCA', 'PRESTACIONES', 'LABOR', 'ID']],
            on=['FINCA', 'ID'],
            how='left'
        )

        # Multiplicar columnas mensuales por PRESTACIONES
        for col in month_columns:
            df_sum_merged[col] = df_sum_merged[col] * df_sum_merged['PRESTACIONES']

        logger.info("Multiplicación con PRESTACIONES completada.")

        return df_sum_merged

    except Exception as e:
        logger.error(f"Ocurrió un error en la función: {e}")
        raise


"-------------------------------------------------------------------------------------------------"
def multiply_p_social_block_ciclics(labor_df, promediado_df, full_matrix_df, month_columns):
    months = [str(m).strip() for m in month_columns]

    ldf = labor_df.copy()
    pdf = promediado_df.copy()
    fdf = full_matrix_df.copy()

    # Normaliza nombres de columnas
    for df in (ldf, pdf, fdf):
        df.columns = df.columns.map(lambda c: str(c).strip())

    # Asegura columnas mensuales
    for m in months:
        if m not in ldf.columns: ldf[m] = 0.0
        if m not in pdf.columns: pdf[m] = 0.0

    # A numérico
    ldf[months] = ldf[months].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pdf[months] = pdf[months].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # ---------- SUMA (labor + promediado) ----------
    if "ID" in pdf.columns:
        # Alinear por FINCA + ID (solo suma promediado al ID correspondiente)
        pdf_unique = (pdf.drop_duplicates(subset=["FINCA","ID"], keep="first")
                        .set_index(["FINCA","ID"])[months])
        idx = pd.MultiIndex.from_frame(ldf[["FINCA","ID"]])
        prom_aligned = pdf_unique.reindex(idx).fillna(0.0).to_numpy()
    else:
        # Fallback: alinear por FINCA (una fila por FINCA)
        pdf_unique = (pdf.drop_duplicates(subset=["FINCA"], keep="first")
                        .set_index("FINCA")[months])
        prom_aligned = pdf_unique.reindex(ldf["FINCA"]).fillna(0.0).to_numpy()

    labor_vals  = ldf[months].to_numpy(dtype=float)
    summed_vals = labor_vals + prom_aligned   # misma forma que labor_df[months]

    # ---------- PRESTACIONES por (FINCA, ID) ----------
    fdf_unique = (fdf[["FINCA","ID","PRESTACIONES"]]
                    .drop_duplicates(subset=["FINCA","ID"], keep="first")
                    .copy())
    fdf_unique["PRESTACIONES"] = pd.to_numeric(
        fdf_unique["PRESTACIONES"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    ).fillna(0.0)

    pairs = pd.MultiIndex.from_frame(ldf[["FINCA","ID"]])
    prest = fdf_unique.set_index(["FINCA","ID"])["PRESTACIONES"].reindex(pairs).fillna(0.0).to_numpy()

    # ---------- Multiplicación final ----------
    result_vals = summed_vals * prest.reshape(-1, 1)

    # Armar resultado final
    result = pd.DataFrame(result_vals, columns=months)
    result.insert(0, "ID", ldf["ID"].values)
    result.insert(0, "FINCA", ldf["FINCA"].values)

    return result


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
def total_cost_block(df1, df2, df3, col_1, col_2, months):
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
        df_sum = df1[[col_1, col_2]].copy()  # Crear df_sum con la columna FINCA
        for col in months:
            df_sum[col] = df1[col] + df2[col]+ df3[col]

        logger.info("Suma realizada con éxito.")

        # Merge de ambos DataFrames en base a la columna 'FINCA'
        logger.info("Realizando el merge sobre la columna FINCA y LABOR")
        df_merged = pd.merge(df_sum, df3[[col_1, col_2]],  on=[col_1, col_2], how='left')

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
def total_cost_parcela(df1, df2, df3, month_columns):
    """
    Suma fila a fila los valores de df1 + df2 + df3 en las columnas mensuales.

    Parámetros:
    - df1, df2, df3: DataFrames con columnas de meses.
    - month_columns: lista de columnas mensuales como ['2024-01', ..., '2024-12']

    Retorna:
    - DataFrame con FINCA, PROMEDIADO y columnas mensuales sumadas.
    """

    try:
        # Normalizar columnas
        for df in [df1, df2, df3]:
            df.columns = df.columns.astype(str).str.strip()
        month_columns = [str(col).strip() for col in month_columns]

        # Asegurar numéricos en columnas mensuales
        for df in [df1, df2, df3]:
            for col in month_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sumar los tres DataFrames por columnas
        df_sum = df1[month_columns].add(df2[month_columns], fill_value=0)
        df_sum = df_sum.add(df3[month_columns], fill_value=0)

        # Combinar con columnas clave
        result = pd.concat([df1[['FINCA', 'PROMEDIADO']], df_sum], axis=1)

        return result

    except Exception as e:
        print(f"Error en sumar_tres_dataframes: {e}")
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

"-------------------------------------------------------------------------------------"
# funciones de gastos de personal

def parametro_1(df: pd.DataFrame, row_factor, ceil: bool=False) -> pd.DataFrame:
    # Alinear row_factor al índice de df (por posición si no coincide)
    rf = row_factor if isinstance(row_factor, pd.Series) and row_factor.index.equals(df.index) \
         else pd.Series(np.asarray(row_factor), index=df.index)

    # Columnas de meses (YYYY-MM)
    mcols = df.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$').columns
    if mcols.empty:
        raise ValueError("No se encontraron columnas de meses con formato YYYY-MM.")

    out = df.copy()
    out[mcols] = out[mcols].apply(pd.to_numeric, errors="coerce").mul(rf.to_numpy().reshape(-1,1))
    if ceil:
        out[mcols] = np.ceil(out[mcols].to_numpy())
    return out

"----------------------------------------------------------------------"

import numpy as np
import pandas as pd

def parametro_2(
    df: pd.DataFrame,
    row_factor,                    # pd.Series | array-like de longitud == len(df)
    months = None                  # lista de meses; si None, detecta YYYY-MM
) -> pd.DataFrame:
    if "FINCA" not in df.columns:
        raise ValueError("Se requiere la columna 'FINCA' en df.")
    n = len(df)
    rf = np.asarray(row_factor)
    if rf.shape[0] != n:
        raise ValueError(f"row_factor debe tener longitud {n}; recibido {rf.shape[0]}.")

    # Determinar columnas de meses
    if months is None:
        mcols = df.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$').columns.tolist()
        if not mcols:
            raise ValueError("No se encontraron columnas de meses con formato YYYY-MM; provee 'months'.")
    else:
        mcols = list(months)

    # Construir salida: FINCA + matriz de 1s (float) del mismo shape (n x len(mcols))
    out = pd.DataFrame(index=df.index)
    out["FINCA"] = df["FINCA"]
    out[mcols] = 1.0

    # Multiplicar por fila (cada fila * rf[i])
    out[mcols] = out[mcols].to_numpy() * rf.reshape(-1, 1)

    return out





def parametro_3(df: pd.DataFrame, row_factor, months: list | None = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "FINCA" not in df.columns: raise ValueError("Se requiere la columna 'FINCA'.")

    n, rf = len(df), np.asarray(row_factor)
    if rf.shape[0] != n: raise ValueError(f"row_factor debe tener longitud {n}.")

    pat = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')

    # Determinar columnas de meses (respeta el orden de 'months' si viene)
    if months is None:
        mcols = [c for c in df.columns if pat.match(c)]
        if not mcols: raise ValueError("No se detectaron columnas YYYY-MM.")
    else:
        months = [str(m).strip() for m in months]
        def find_col(m):
            if m in df.columns: return m
            mm = re.match(r'^\d{4}-(\d{2})$', m)
            if not mm: return None
            suf = "-" + mm.group(1)
            for c in df.columns:
                if pat.match(c) and c.endswith(suf): return c
            return None
        mcols = [c for m in months if (c:=find_col(m)) is not None]
        if not mcols: raise ValueError("Ninguna columna de 'months' está en el DataFrame (tras normalizar).")

    # Mapa MM -> columna (si hay duplicados por año, toma la primera encontrada en mcols)
    mm2col = {c[-2:]: c for c in mcols}

    out = df[["FINCA"]].copy()
    out[mcols] = df[mcols].apply(pd.to_numeric, errors="coerce")

    # 0 en: ene, feb, may, jun, jul, sep, oct, nov
    for mm in ("01","02","05","06","07","09","10","11"):
        c = mm2col.get(mm)
        if c: out[c] = 0.0

    # marzo = 45000 * 39
    if (c := mm2col.get("03")): out[c] = float(45000 * 39)

    # abr, ago, dic = valor df × row_factor (por fila, posición)
    for mm in ("04","08","12"):
        if (c := mm2col.get(mm)):
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy() * rf

    return out
"----------------------------------------------------------------------------------------"
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import re

def parametro_4(
    df: pd.DataFrame,
    row_factor,                       # Serie/array (factor por FILA)
    month_factors_df: pd.DataFrame,   # DataFrame 1xN con factores por MES
    ceil: bool = False
) -> pd.DataFrame:
    out = df.copy()

    # Alinear row_factor por posición
    rf = row_factor if isinstance(row_factor, pd.Series) and row_factor.index.equals(out.index) \
         else pd.Series(np.asarray(row_factor), index=out.index)

    # Detectar columnas de meses: Period[M] o 'YYYY-MM'
    mcols, mcols_norm = [], []
    pat = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')
    for c in out.columns:
        if isinstance(c, pd.Period) and c.freqstr == 'M':
            mcols.append(c); mcols_norm.append(str(c))          # 'YYYY-MM'
        elif isinstance(c, str) and pat.match(c.strip()):
            mcols.append(c); mcols_norm.append(c.strip())
    if not mcols:
        raise ValueError("No se encontraron columnas de meses (Period[M] o 'YYYY-MM').")

    # Base numérica
    base = out[mcols].apply(pd.to_numeric, errors="coerce").to_numpy()

    # Factor por fila
    res = base * rf.to_numpy().reshape(-1, 1)

    # Factor por mes (alineado a mcols_norm)
    mf = month_factors_df.copy()
    mf.columns = [str(c).strip() for c in mf.columns]
    mf_vec = mf.reindex(columns=mcols_norm).iloc[0].to_numpy(dtype=float)
    if np.isnan(mf_vec).any():
        falt = [c for c, v in zip(mcols_norm, mf_vec) if pd.isna(v)]
        raise ValueError(f"month_factors_df no tiene valores para: {falt}")
    res = res * mf_vec.reshape(1, -1)

    # Ajuste adicional: julio–diciembre  (* 235/230)
    factor_jd = 235.0 / 230.0
    jd_idx = [i for i, col in enumerate(mcols_norm) if col.endswith(('-07','-08','-09','-10','-11','-12'))]
    if jd_idx:
        res[:, jd_idx] *= factor_jd

    # Ceil opcional
    if ceil:
        res = np.ceil(res)

    # Volcar respetando nombres originales (mcols)
    out[mcols] = res
    return out

"--------------------------------------------------------------------------------"

import numpy as np
import pandas as pd
import re

def parametro_5(
    df: pd.DataFrame,
    row_factor,             # Serie/array: factor por FILA
    factor_a: float,        # ajuste para JUNIO (06) -> base * (rf + A)
    factor_b: float,        # ajuste para SEPTIEMBRE (09) -> base * (rf + B)
    factor_c: float,        # ajuste para OCTUBRE (10) -> (base * rf) + C
    ceil: bool = False,
    round_to: int | None = None
) -> pd.DataFrame:
    out = df.copy()

    # Alinear row_factor por posición
    rf = row_factor if isinstance(row_factor, pd.Series) and row_factor.index.equals(out.index) \
         else pd.Series(np.asarray(row_factor), index=out.index)

    # Detectar columnas de meses (Period[M] o 'YYYY-MM')
    mcols, mm_codes = [], []
    pat = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')
    for c in out.columns:
        if isinstance(c, pd.Period) and getattr(c, "freqstr", None) == "M":
            mcols.append(c); mm_codes.append(f"{c.month:02d}")
        elif isinstance(c, str) and pat.match(c.strip()):
            s = c.strip(); mcols.append(c); mm_codes.append(s[-2:])
    if not mcols:
        raise ValueError("No se encontraron columnas de meses (Period[M] o 'YYYY-MM').")

    base = out[mcols].apply(pd.to_numeric, errors="coerce").to_numpy()
    rf_np = rf.to_numpy().reshape(-1, 1)

    # Por defecto: base * rf
    res = base * rf_np

    # Meses especiales
    for j, mm in enumerate(mm_codes):
        if mm == "06":      # junio: base * (rf + A)
            res[:, j] = base[:, j] * (rf_np.ravel() + factor_a)
        elif mm == "09":    # septiembre: base * (rf + B)
            res[:, j] = base[:, j] * (rf_np.ravel() + factor_b)
        elif mm == "10":    # octubre: (base * rf) + C
            res[:, j] = (base[:, j] * rf_np.ravel()) + factor_c

    if ceil:
        res = np.ceil(res)
    if round_to:
        res = np.round(res / round_to) * round_to

    out[mcols] = res
    return out

"--------------------------------------------------------------------------"
import numpy as np
import pandas as pd
import re

def parametro_6(df: pd.DataFrame, row_factor) -> pd.DataFrame:
    # Copia y normaliza row_factor por posición (acepta escalar o Serie)
    out = df.copy()
    if isinstance(row_factor, pd.Series) and row_factor.index.equals(out.index):
        rf = row_factor.to_numpy()
    else:
        rf = np.asarray(row_factor)
        if rf.ndim == 0:  # escalar
            rf = np.full(len(out), rf, dtype=float)
        elif rf.shape[0] != len(out):
            raise ValueError(f"row_factor debe tener longitud {len(out)}.")

    # Detectar columnas de meses (Period[M] o 'YYYY-MM')
    mcols, mm_codes = [], []
    pat = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')
    for c in out.columns:
        if isinstance(c, pd.Period) and getattr(c, "freqstr", None) == "M":
            mcols.append(c); mm_codes.append(f"{c.month:02d}")
        elif isinstance(c, str) and pat.match(c.strip()):
            s = c.strip(); mcols.append(c); mm_codes.append(s[-2:])
    if not mcols:
        raise ValueError("No se encontraron columnas de meses (Period[M] o 'YYYY-MM').")

    # Inicializa todos los meses en 0
    out[mcols] = 0.0

    # Factores constantes
    k_ene = 8 * 1.39 / 235.0
    k_dic = 8 * 1.39 / 230.0

    # Aplica enero y diciembre si existen
    for col, mm in zip(mcols, mm_codes):
        if mm == "01":  # enero
            base = pd.to_numeric(df[col], errors="coerce").to_numpy()
            out[col] = base * rf * k_ene
        elif mm == "12":  # diciembre
            base = pd.to_numeric(df[col], errors="coerce").to_numpy()
            out[col] = base * rf * k_dic

    # Conserva FINCA tal cual
    if "FINCA" in df.columns:
        out["FINCA"] = df["FINCA"]
    else:
        raise ValueError("Se requiere la columna 'FINCA'.")

    return out



def parametro_8(df: pd.DataFrame) -> pd.DataFrame:
    if "FINCA" not in df.columns:
        raise ValueError("Se requiere la columna 'FINCA'.")

    out = df.copy()

    # Detectar columnas de meses (YYYY-MM o Period[M])
    mcols, mm_codes = [], []
    pat = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')
    for c in out.columns:
        if isinstance(c, pd.Period) and getattr(c, "freqstr", None) == "M":
            mcols.append(c); mm_codes.append(f"{c.month:02d}")
        elif isinstance(c, str) and pat.match(c.strip()):
            s = c.strip(); mcols.append(c); mm_codes.append(s[-2:])
    if not mcols:
        raise ValueError("No se detectaron columnas de meses (YYYY-MM o Period[M]).")

    # Inicializa meses en 0
    out[mcols] = 0.0

    # Factores
    k_mayo = 3 * 50000 * 0.10     # 15000
    k_agos = 46000
    k_dic  = 147000
    const_sep = 30 * 170000       # 5_100_000

    # Aplica reglas por mes si existen
    for col, mm in zip(mcols, mm_codes):
        if mm == "05":  # mayo
            base = pd.to_numeric(df[col], errors="coerce").to_numpy()
            out[col] = base * k_mayo
        elif mm == "08":  # agosto
            base = pd.to_numeric(df[col], errors="coerce").to_numpy()
            out[col] = base * k_agos
        elif mm == "09":  # septiembre (constante)
            out[col] = float(const_sep)
        elif mm == "12":  # diciembre
            base = pd.to_numeric(df[col], errors="coerce").to_numpy()
            out[col] = base * k_dic

    # FINCA se conserva tal cual
    out["FINCA"] = df["FINCA"]
    return out

"------------------------------------------------------------------------"
import numpy as np
import pandas as pd
import re

def parametro_9(
    df: pd.DataFrame,
    row_factor,             # Serie/array: factor por FILA
    factor_a: float,        # MARZO (03) -> base * (rf + A)
    factor_b: float,        # ABRIL (04) -> base * (rf + B)
    factor_c: float,        # JUNIO (06) -> base * (rf + C)
    factor_d: float,        # SEPTIEMBRE (09) -> base * (rf + D)
    ceil: bool = False,
    round_to: int | None = None
) -> pd.DataFrame:
    out = df.copy()

    # Alinear row_factor por posición
    rf = row_factor if isinstance(row_factor, pd.Series) and row_factor.index.equals(out.index) \
         else pd.Series(np.asarray(row_factor), index=out.index)

    # Detectar columnas de meses (Period[M] o 'YYYY-MM')
    mcols, mm_codes = [], []
    pat = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')
    for c in out.columns:
        if isinstance(c, pd.Period) and getattr(c, "freqstr", None) == "M":
            mcols.append(c); mm_codes.append(f"{c.month:02d}")
        elif isinstance(c, str) and pat.match(c.strip()):
            s = c.strip(); mcols.append(c); mm_codes.append(s[-2:])
    if not mcols:
        raise ValueError("No se encontraron columnas de meses (Period[M] o 'YYYY-MM').")

    # Base numérica y factor fila
    base = out[mcols].apply(pd.to_numeric, errors="coerce").to_numpy()
    rf_np = rf.to_numpy().reshape(-1, 1)

    # Por defecto: base * rf
    res = base * rf_np

    # Meses especiales
    for j, mm in enumerate(mm_codes):
        if mm == "03":      # marzo
            res[:, j] = base[:, j] * (rf_np.ravel() + factor_a)
        elif mm == "04":    # abril
            res[:, j] = base[:, j] * (rf_np.ravel() + factor_b)
        elif mm == "06":    # junio
            res[:, j] = base[:, j] * (rf_np.ravel() + factor_c)
        elif mm == "09":    # septiembre
            res[:, j] = base[:, j] * (rf_np.ravel() + factor_d)
        elif mm == "12":    # diciembre: rf + 60000*3 + 20000
            res[:, j] = base[:, j] * (rf_np.ravel() + (60000*3 + 20000))

    # Redondeos opcionales
    if ceil:
        res = np.ceil(res)
    if round_to:
        res = np.round(res / round_to) * round_to

    out[mcols] = res
    return out

"------------------------------------------------------------------------"

import numpy as np
import pandas as pd
import re

def parametro_10(df: pd.DataFrame, row_factor) -> pd.DataFrame:
    out = df.copy()

    # Alinear row_factor por posición (acepta escalar o Serie)
    if isinstance(row_factor, pd.Series) and row_factor.index.equals(out.index):
        rf = row_factor.to_numpy(dtype=float)
    else:
        rf = np.asarray(row_factor, dtype=float)
        if rf.ndim == 0:
            rf = np.full(len(out), rf, dtype=float)
        elif rf.shape[0] != len(out):
            raise ValueError(f"row_factor debe tener longitud {len(out)}.")

    # Detectar columnas de meses (YYYY-MM o Period[M])
    mcols, mm_codes = [], []
    pat = re.compile(r'^\d{4}-(0[1-9]|1[0-2])$')
    for c in out.columns:
        if isinstance(c, pd.Period) and getattr(c, "freqstr", None) == "M":
            mcols.append(c); mm_codes.append(f"{c.month:02d}")
        elif isinstance(c, str) and pat.match(c.strip()):
            s = c.strip(); mcols.append(c); mm_codes.append(s[-2:])
    if not mcols:
        raise ValueError("No se encontraron columnas de meses (Period[M] o 'YYYY-MM').")
    if "FINCA" not in out.columns:
        raise ValueError("Se requiere la columna 'FINCA'.")

    base = out[mcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    rf_col = rf.reshape(-1, 1)

    # Por defecto: df_mes * row_factor
    res = base * rf_col

    # Marzo: df_mes * (row_factor + 1_300_000*0.20)
    add_mar = 1_300_000 * 0.20  # 260_000
    # Diciembre: df_mes * (row_factor + 180_000)
    add_dic = 180_000

    for j, mm in enumerate(mm_codes):
        if mm == "03":
            res[:, j] = base[:, j] * (rf + add_mar)
        elif mm == "12":
            res[:, j] = base[:, j] * (rf + add_dic)

    out[mcols] = res
    return out

    "--------------------------------------------------------------------"
   

def parametro_11(df: pd.DataFrame, row_factor, ceil: bool=False) -> pd.DataFrame:
    # Alinear row_factor al índice de df (por posición si no coincide)
    rf = row_factor if isinstance(row_factor, pd.Series) and row_factor.index.equals(df.index) \
         else pd.Series(np.asarray(row_factor), index=df.index)

    # Columnas de meses (YYYY-MM)
    mcols = df.filter(regex=r'^\d{4}-(0[1-9]|1[0-2])$').columns
    if mcols.empty:
        raise ValueError("No se encontraron columnas de meses con formato YYYY-MM.")

    out = df.copy()
    out[mcols] = out[mcols].apply(pd.to_numeric, errors="coerce").mul(rf.to_numpy().reshape(-1,1))

    # Redondeo opcional
    if ceil:
        out[mcols] = np.ceil(out[mcols].to_numpy())

    # 🔹 Override para FINCA = ENANO
    mask_enano = out["FINCA"].astype(str).str.strip().str.upper().eq("ENANO")
    override_vals = np.array(
        [531000, 531000, 531000, 531000, 531000, 531000,
         531000, 531000, 531000, 7058000, 531000, 531000],
        dtype=float
    )
    if len(mcols) != override_vals.size:
        raise ValueError(f"Se esperaban {len(mcols)} valores para override y llegaron {override_vals.size}.")
    out.loc[mask_enano, mcols] = override_vals

    return out
