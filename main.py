from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from kronos.modeler import Modeler
from pyspark.sql.functions import current_date
from pyspark.sql.types import StructType, StructField, StringType, DateType, FloatType

sc = SparkContext('local')
spark = SparkSession(sc)

df = spark.read.format("delta").load("dbfs:/mnt/dpc-datascience/ds_pdr_consumi_forecast_hera_calibro45")
df.createOrReplaceTempView("hera_45")

sql_df = spark.sql("""
    SELECT A.pdr, A.giorno_gas as date, A.volume_giorno--, B.quality_rank
    FROM hera_45 as A
    INNER JOIN (
        SELECT pdr, COUNT(*) as n_rows, ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS quality_rank
        FROM hera_45
        GROUP BY pdr
        ORDER BY quality_rank
        LIMIT 100 --Limit to 100 pdr
    ) AS B
    ON A.pdr = B.pdr
    ORDER BY A.pdr, A.giorno_gas
    """)

# Define output schema
result_schema =StructType([
  StructField('ds',DateType()),
  StructField('pdr',StringType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  ])

# Apply Model UDF
df_pred = sql_df.groupby("pdr").applyInPandas(Modeler().forecast, schema=result_schema).withColumn('training_date', current_date())

df_pred.head(10)
