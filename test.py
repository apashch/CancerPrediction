import pandas as pd
#df_tmp.set_index('SEQN', inplace=True)
s1 = pd.DataFrame({'val' : ['a', 'b', 'e'], 'ind' : [1.0,2.0,3.0]})
s2 = pd.DataFrame({'val' : ['c', 'd', 'f'], 'ind' : [2.0,3.0,4.0]})
s1.set_index('ind', inplace=True)
s2.set_index('ind', inplace=True)
print(s1.head())
print(pd.concat([s1, s2], axis= 1))