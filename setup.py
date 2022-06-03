from setuptools import setup

setup(
    name='kronos',
    version='2.0.0',
    packages=['kronos', 'kronos.models', 'kronos.models.tensorflow'],
    author='zantedeschim',
    author_email='matteo.zantedeschi@sdggroup.com',
    description='Kronos package to manage time-series in Databricks',
)
