# kronos
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

You got your forecast! Plus, all trainings are tracked in MLflow Tracking and all models have been versioned on MLflow Model Registry.

---

# Release management

## Azure
The required steps to release a new version on Azure DevOps are: 

  1. Update package version in **setup.py** (_consistent with the change made_).
  2. **Commit** changes.
  3. Create a **tag** called *release* (_required to trigger the azure release pipeline_).
  4. **Push** commit and tag.

The new version of the package is now available in the Artifact feed specified in the **azure-pipelines.yaml** file.

---

# Install on Databricks Cluster

## Azure
The required steps to install kronos on a Databricks cluster are:
  1. Generate a **Personal Access Token (PAT)** on Azure DevOps, setting at least *read* privilege on *Packaging* section. 
  This is required to correctly authenticate to the *Artifact feed*. 
  2. Add the PAT as a **secret** into the KeyVault.
  3. Add the following as **environment variable** of the Databricks cluster:  
  ```PYPI_TOKEN={{secrets/YourSecretScopeName/YourSecretName}}```  
  *(Don’t forget to replace the Secret Scope and Secret names by your own.)*
  4. Get the **URL** of your private PyPI repository.  
  You can find it going to: Azure DevOps -> Artifacts -> Connect to feed -> pip -> index-url
  5. Create **init script** for Databricks clusters, by running in a notebook the following code:
  ```
  script = r"""
  #!/bin/bash
  if [[ $PYPI_TOKEN ]]; then
    use $PYPI_TOKEN
  fi
  echo $PYPI_TOKEN
  printf "[global]\n" > /etc/pip.conf
  printf "extra-index-url =\n" >> /etc/pip.conf
  printf "\thttps://$PYPI_TOKEN@pkgs.dev.azure.com/organization/DataLab/_packaging/datalabartifacts/pypi/simple/\n" >> /etc/pip.conf
  """
  dbutils.fs.put("/databricks/scripts/init-scripts/set-private-pip-repositories.sh", script, True)
  ```
6. **Install** the package in your Databricks cluster as a PyPI package. 