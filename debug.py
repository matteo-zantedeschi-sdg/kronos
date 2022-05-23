import pandas as pd
import datetime
import random

# Config
n_days = 365
min_val = 0
max_val = 100

# Creation
df = pd.DataFrame(data={
    'key': ['1' for x in range(n_days)],
    'ds': [datetime.date.today() - datetime.timedelta(days=x) for x in range(n_days)],
    'y': [random.randint(min_val, max_val) for x in range(n_days)]}
)
