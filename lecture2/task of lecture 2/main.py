import numpy as np
import pandas as pd

if __name__ == '__main__':
    file = 'LifeExpectancy.csv' 
df = pd.read_csv(file, header=0)
print(df.head(5))
