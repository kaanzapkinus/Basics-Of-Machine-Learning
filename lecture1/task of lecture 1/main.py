import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

file = os.path.join(os.path.dirname(__file__), 'LifeExpectancy.csv')
df = pd.read_csv(file, header=0)

    for col in df.columns:
        print(col)

    print(df.head(5))