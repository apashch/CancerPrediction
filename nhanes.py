import pdb
import glob
import copy
import os
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.feature_selection

class FeatureColumn:
	def __init__(self, category, file, field, preprocessor, args=None, cost=None):
		self.category = category
		self.file = file
		self.field = field
		self.preprocessor = preprocessor
		self.args = args
		self.data = None
		self.cost = cost

	def __str__(self):
		return self.category+", "+self.file+", "+self.field

class NHANES:
	def __init__(self, db_path, columns):
		self.db_path = db_path
		self.columns = columns
		self.column_data = None
		self.column_info = None
		self.df_features = None
		self.df_targets = None
		self.costs = None


	def process_new(self, verbose):
		cache = {}
		df = pd.DataFrame()
		for col in self.columns:

			#  TO DO: construct lookup function
			if col.file == None or col.category == None:
				pass
				#col.file, col.category = lookup(col.field)
			
			# read file
			dfile = col.file
			folder = col.category

			dfilepath = self.db_path+folder+'/'+dfile+".XPT"
			print(dfilepath)

			if dfile in cache:
				df_tmp = cache[dfile]
			else:
				try:
					df_tmp = pd.read_sas(dfilepath)
					cache[dfile] = df_tmp
				except FileNotFoundError:
					if verbose:
						print("Not found in the DB ", col)
					continue

			# sanity check 1 - do we have an ID? 
			if 'SEQN' not in df_tmp.columns:
				if verbose:
					print("No SEQN found in ", col)
				continue

			# select only relevant feature
			df_col = df_tmp.filter(items=['SEQN', col.field])

			# sanity check 2 - now we should have 2 cols
			if df_col.shape[1] != 2:
				if verbose:
					print("Failed to select field in", col)
				continue

			df.join(df_col.set_index("SEQN"))
		
		proc_df = pd.DataFrame()


		# do preprocessing steps
		for col in self.columns:

			field = col.field

			#skip the ones we failed to load
			if field not in df.columns:
				if verbose:
					print("Preprocessing on failed on", col)
				continue

			# do preprocessing
			if col.preprocessor is not None:
				prepr_col = col.preprocessor(df[field], col.args)
			else:
				prepr_col = df[field]

		proc_df.join(prepr_col.set_index("SEQN"))
		return proc_df
			



				



	def process(self, verbose):
		df = None
		cache = {}
		# collect relevant data
		df = []
		for fe_col in self.columns:
			sheet = fe_col.category
			field = fe_col.field
			data_files = glob.glob(self.db_path+sheet+'/*.XPT')
			df_col = []
			for dfile in data_files:
				print(80*' ', end='\r')
				print('\rProcessing: ' + dfile.split('/')[-1], end='')
				# read the file
				if dfile in cache:
					df_tmp = cache[dfile]
				else:
					df_tmp = pd.read_sas(dfile)
					cache[dfile] = df_tmp
				# skip of there is no SEQN
				if 'SEQN' not in df_tmp.columns:
					continue
				# skip if there is nothing interseting there
				sel_cols = set(df_tmp.columns).intersection([field])
				if not sel_cols:
					continue
				else:
					df_tmp = df_tmp[['SEQN'] + list(sel_cols)]
					df_tmp.set_index('SEQN', inplace=True)
					df_col.append(df_tmp)

			try:
				df_col = pd.concat(df_col)
			except:
				#raise Error('Failed to process' + field)
				#raise Exception('Failed to process' + field)
				if verbose:
					print('Failed to process' + field)
				continue
			df.append(df_col)
		df = pd.concat(df, axis=1)
		#df = pd.merge(df, df_sel, how='outer')

		# do preprocessing steps
		df_proc = []#[df['SEQN']]
		for fe_col in self.columns:
			field = fe_col.field
			fe_col.data = df[field].copy()
			# do preprocessing
			if fe_col.preprocessor is not None:
				prepr_col = fe_col.preprocessor(df[field], fe_col.args)
			else:
				prepr_col = df[field]
			# handle the 1 to many
			if (len(prepr_col.shape) > 1):
				fe_col.cost = [fe_col.cost] * prepr_col.shape[1]
			else:
				fe_col.cost = [fe_col.cost]
			df_proc.append(prepr_col)
		self.dataset = pd.concat(df_proc, axis=1)
		return self.dataset
	
	
# Preprocessing functions
def preproc_onehot(df_col, args=None):
	return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_real(df_col, args=None):
	if args is None:
		args={'cutoff':np.inf}
	# other answers as nan
	df_col[df_col > args['cutoff']] = np.nan
	# nan replaced by mean
	df_col[df_col.isna()] = df_col.mean()
	# statistical normalization
	df_col = (df_col-df_col.mean()) / df_col.std()
	return df_col

def preproc_impute(df_col, args=None):
	# nan replaced by mean
	df_col[df_col.isna()] = df_col.mean()
	return df_col

def preproc_cut(df_col, bins):
	# limit values to the bins range
	df_col = df_col[df_col >= bins[0]]
	df_col = df_col[df_col <= bins[-1]]
	return pd.cut(df_col.iloc[:,0], bins, labels=False)

def preproc_dropna(df_col, args=None):
	df_col.dropna(axis=0, how='any', inplace=True)
	return df_col

#### Add your own preprocessing functions ####

# Dataset loader
class Dataset():
	""" 
	Dataset manager class
	"""
	def  __init__(self, data_path=None):
		"""
		Class intitializer.
		"""
		# set database path
		if data_path == None:
			self.data_path = './run_data/'
		else:
			self.data_path = data_path
		# feature and target vecotrs
		self.features = None
		self.targets = None
		self.costs = None
		
	def load_arthritis(self, opts=None):
		columns = [
			# TARGET: systolic BP average
			FeatureColumn('Questionnaire', 'MCQ160A', 
									None, None),
			# Gender
			FeatureColumn('Demographics', 'RIAGENDR', 
								 preproc_real, None),
			# Age at time of screening
			FeatureColumn('Demographics', 'RIDAGEYR', 
								 preproc_real, None),
			FeatureColumn('Demographics', 'RIDRETH3', 
								 preproc_onehot, None),
			# Race/ethnicity
			FeatureColumn('Demographics', 'RIDRETH1', 
								 preproc_onehot, None),
			# Annual household income
			FeatureColumn('Demographics', 'INDHHINC', 
								 preproc_real, {'cutoff':11}),
			# Education level
			FeatureColumn('Demographics', 'DMDEDUC2', 
								 preproc_real, {'cutoff':5}),
			# BMI
			FeatureColumn('Examination', 'BMXBMI', 
								 preproc_real, None),
			# Waist
			FeatureColumn('Examination', 'BMXWAIST', 
								 preproc_real, None),
			# Height
			FeatureColumn('Examination', 'BMXHT', 
								 preproc_real, None),
			# Upper Leg Length
			FeatureColumn('Examination', 'BMXLEG', 
								 preproc_real, None),
			# Weight
			FeatureColumn('Examination', 'BMXWT', 
								 preproc_real, None),
			# Total Cholesterol
			FeatureColumn('Laboratory', 'LBXTC', 
								 preproc_real, None),
			# Alcohol consumption
			FeatureColumn('Questionnaire', 'ALQ101', 
								 preproc_real, {'cutoff':2}),
			FeatureColumn('Questionnaire', 'ALQ120Q', 
								 preproc_real, {'cutoff':365}),
			# Vigorous work activity
			FeatureColumn('Questionnaire', 'PAQ605', 
								 preproc_real, {'cutoff':2}),
			FeatureColumn('Questionnaire', 'PAQ620', 
								 preproc_real, {'cutoff':2}),
			FeatureColumn('Questionnaire', 'PAQ180', 
								 preproc_real, {'cutoff':4}),
			FeatureColumn('Questionnaire', 'PAD615', 
								 preproc_real, {'cutoff':780}),
			# Doctor told overweight (risk factor)
			FeatureColumn('Questionnaire', 'MCQ160J', 
								 preproc_onehot, {'cutoff':2}),
			# Sleep
			FeatureColumn('Questionnaire', 'SLD010H', 
								 preproc_real, {'cutoff':12}),
			# Smoking
			FeatureColumn('Questionnaire', 'SMQ020', 
								 preproc_onehot, None),
			FeatureColumn('Questionnaire', 'SMD030', 
								 preproc_real, {'cutoff':72}),
			# Blood relatives with arthritis
			FeatureColumn('Questionnaire', 'MCQ250D',
								 preproc_onehot, {'cutoff':2}),
			# joint pain/aching/stiffness in past year
			FeatureColumn('Questionnaire', 'MPQ010',
								 preproc_onehot, {'cutoff':2}),
			# symptoms began only because of injury
			FeatureColumn('Questionnaire', 'MPQ030',
								 preproc_onehot, {'cutoff':2}),
			# how long experiencing pain
			FeatureColumn('Questionnaire', 'MPQ110',
								 preproc_real, {'cutoff':4}),
		]
		nhanes_dataset = NHANES(self.data_path, columns)
		df = nhanes_dataset.process()
		fe_cols = df.drop(['MCQ160A'], axis=1)
		features = fe_cols.values
		target = df['MCQ160A'].values
		# remove nan labeled samples
		inds_valid = ~ np.isnan(target)
		features = features[inds_valid]
		target = target[inds_valid]

		# Put each person in the corresponding bin
		targets = np.zeros(target.shape[0])
		targets[target == 1] = 0 # yes arthritis
		targets[target == 2] = 1 # no arthritis

	   # random permutation
		perm = np.random.permutation(targets.shape[0])
		self.features = features[perm]
		self.targets = targets[perm]
		self.costs = [c.cost for c in columns[1:]]
		self.costs = np.array(
			[item for sublist in self.costs for item in sublist])
		
		
	def load_custom(self, features, verbose = False):

		#assuming features were passed as a dataframe
		columns = [FeatureColumn(item[2], item[1], item[0], None) for item in features.values]
		nhanes_dataset = NHANES(self.data_path, columns)
		df = nhanes_dataset.process_new(verbose)

		#target column
		fe_cols = df.drop(['MCQ220'], axis=1)
		features = fe_cols.values
		target = df['MCQ220'].values
		# remove nan labeled samples
		inds_valid = ~ np.isnan(target)
		features = features[inds_valid]
		target = target[inds_valid]

		# Put each person in the corresponding bin
		targets = np.zeros(target.shape[0])
		targets[target == 1] = 1 # yes cancer
		targets[target == 2] = 0 # no cancer
		targets[target == 9] = 0 # don't know assume no cancer

	   # random permutation
		perm = np.random.permutation(targets.shape[0])
		self.features = features[perm]
		self.targets = targets[perm]
		self.costs = [c.cost for c in columns[1:]]
		self.costs = np.array(
			[item for sublist in self.costs for item in sublist])
