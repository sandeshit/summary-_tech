import pandas as pd
dfnew = pd.read_csv('output.csv')

for i in range(0,25):
    print(dfnew['json_out'][i])