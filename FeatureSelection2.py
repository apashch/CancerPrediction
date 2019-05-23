import pandas as pd
import nhanes as nhanes

DATA_PATH = '/Users/Artem/Documents/CS 205/NHANES/'

features = pd.read_csv("/Users/Artem/Documents/GitHub/CancerPrediction/all_features.csv")
target = features[features.feature == 'MCQ220']
# drop target
features = features[features.feature != 'MCQ220']
# drop all closely related cols
features = features[~features.feature.str.contains("MCQ22|MCQ23|MCQ24", regex = True)]

discarded = []
kept = []

for i in range(features.shape[0]):
	thisrow = features.iloc[[i]]
	thisfeature = thisrow.feature
	loadingset = pd.concat([thisrow, target], axis = 0)
	ds = nhanes.Dataset(DATA_PATH)
	try:
		ds.load_custom(loadingset, verbose = False)
	except:
		continue
	
	pos = ds.df[ds.df["MCQ220"] == 1]
	neg = ds.df[ds.df["MCQ220"] == 0]

	try:
		if pos[thisfeature].isna().sum()[0] > 0.5 * pos.shape[0] and neg[thisfeature].isna().sum()[0] > 0.5 * neg.shape[0]:
			discarded.append(thisrow)
		else:
			kept.append(thisrow)
	except KeyError:
		continue
		
	if (i+1) % 10 == 0:
		disc_df = pd.concat(discarded)
		kept_df = pd.concat(kept)
		
		if ((i+1)/10)%2 == 0:
			disc_df.to_csv("all_features_dicarded.csv")
			kept_df.to_csv("all_features_kept.csv")
		else:
			disc_df.to_csv("all_features_dicarded2.csv")
			kept_df.to_csv("all_features_kept2.csv")