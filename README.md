# kronos
Kronos package to manage time-series in Databricks.

This package provides a framework for working with time series in parallel using Spark, and managing ML workflow through MLflow.  
It is developed and tested in *Databricks 10.4 LTS ML (includes Apache Spark 3.2.1, Scala 2.12)*.  

Main steps are:
  1. Creation of MLflow **experiment** (if missing) and **run**.
  2. **Train/test** split.
  3. **Define models** to train and log their parameters in MLflow. 
  4. **Train models** and log them in MLflow. 
  5. **Predict** with all the models and compute the performance score. Finally log it in MLflow. 
  6. Retrieve the **current production model** (if present), **predict** and compute the performance score. 
  7. Find the **winning model** and, if different from the current production model, **register** it in MLflow Model Registry. 
  8. Take the registered model in the *Staging* are and perform a **unit test**. 
  9. If unit test succeed **promote** the model to the *Production* area. 
  10. Retrieve the new current production model and use it to provide the **final forecast**. 

## 1. Read data
Here the delta format is used, although recommended in Databricks it is not mandatory.

```python
df = spark.read.format("delta").load("dbfs:/delta_table")
```

## 2. Init MLflow Client
The client is used to track experiments and runs and to manage model versioning.  

The main interactions with the clients are the following: 

  1. If not present, an **experiment** for each time series will be created, with the name of the key that identifies the time series. 
  2. Under each experiment, every execution is tracked in a different **run** named with the current timestamp in iso format. 
  3. Resulting models, whether they perform "well enough", are first **registered** in the MLflow model registry and then are taken in the *Staging* area. 
  4. If a *unit test* is passed, they are finally **deployed** in the *production* area.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
```

## 3. Define output schema
Define the schema of the Spark DataFrame containing the output forecast.  
This is **strictly required** (as is) in order to subsequently apply the pandas User Defined Function.

```python
from pyspark.sql.types import StructType, StructField, DateType, FloatType, StringType, IntegerType

result_schema = StructType([
    StructField(key_col, StringType()),
    StructField(date_col, DateType()),
    StructField(fcst_col, FloatType()),
    StructField(dt_creation_col, DateType()),
    StructField(dt_reference_col, DateType()),
    StructField(days_from_last_obs_col, IntegerType()),
])
```

## 4. Force repartion
This step is **required** to enable Spark to apply the function in parallel on the DataFrame key.  

The reason why is required is that [Catalyst](https://databricks.com/glossary/catalyst-optimizer), the Databricks optimizer, groups together "small" partitions to avoid workers overhead.  
This, in the time series scenario, is a problem since usually a time series data is in the order of KBs, while Catalyst standards for the "right" partition size is in the order of MBs.
For this reason, Catalyst groups several time series together, avoiding their parallel execution on the cluster.

With the subsequent code rows, we are forcing the partition key:

```python
partition_key = 'id'
partition_number = df.select(partition_key).distinct().count()
df = df.repartition(partition_number, partition_key)
```

## 5. Generate pandas User Defined Function

```python
from kronos.forecast_udf import forecast_udf_gen

forecast_udf = forecast_udf_gen(
    client=client, 
    key_col=key_col, 
    date_col=date_col, 
    metric_col=metric_col,
    quality_col=quality_col,
    action_col=action_col,
    models_col=models_col,
    models_config=models_config,
    days_from_last_obs_col=days_from_last_obs_col, 
    current_date=current_date, 
    fcst_first_date=fcst_first_date, 
    n_test=n_test, 
    n_unit_test=n_unit_test, 
    forecast_horizon=forecast_horizon,
    dt_creation_col=dt_creation_col,
    dt_reference_col=dt_reference_col
)
```

## 6. Apply pandas User Defined Function

```python
df_pred = df.groupby(partition_key).applyInPandas(forecast_udf, schema=result_schema)
```
