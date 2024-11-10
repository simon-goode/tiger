from tiger.linear import OLS
import pandas as pd

df = pd.DataFrame()

df['income'] = [100000, 90000, 50000, 30000, 10000]
df['age'] = [45, 50, 30, 28, 18]
df['gen'] = [0, 1, 0, 1, 0]
df['yoe'] = [18, 16, 12, 12, 12]

model = OLS(df, ['income'], ['age', 'gen', 'yoe']).fit()
print(model.summary)