import pandas as pd

url = "/Users/Artem/Downloads/NHANES Search Variables-A.html"
df_raw = pd.read_html(url, attrs = {'id' : 'GridView1'})[0]
df_useful = pd.DataFrame(feature = df_raw["Variable Name"])
print(df_useful.head())