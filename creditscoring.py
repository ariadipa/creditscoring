
#import library
from matplotlib.bezier import split_bezier_intersecting_with_closedpath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import ignore warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

#mount drive
from google.colab import drive
drive.mount('/content/drive')

# read data
df = pd.read_csv('/content/drive/MyDrive/ID X Partners - Rakamin Virtual Internship/loan_data_2007_2014.csv')
df.head()

# cek distribusi data 
df.info()

# cek missing value data raw
(df.isna().mean()*100).sort_values(ascending=False).head(25)

# tampilkan missing value >40%
missing_values = df.isna().mean()*100
col_missingvalues = missing_values[missing_values > 40].index
col_missingvalues

# drop missing value >40%
df.drop(col_missingvalues, axis = 1, inplace = True)

# drop column index double
df.drop('Unnamed: 0', inplace=True, axis=1) #hapus variabel index doubel

# cek duplikasi data
print('id',df['id'].nunique())
print('member id',df['member_id'].nunique())

# drop irrelevant data
col = ['id','member_id','url','sub_grade','zip_code']
df.drop(col, axis=1, inplace= True)

# identifikasi target
df.loan_status.value_counts(normalize=True)*100
bad_loan = ['Charged Off', 'Default' , 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']
df['bad_loan'] = np.where(df['loan_status'].isin(bad_loan), 1, 0)

df.drop(columns = ['loan_status'], inplace = True) # Drop kolom 'loan_status'
df['bad_loan'].value_counts(normalize=True)*100

# data cleaning
df.iloc[:,0:16].head(3)
df.iloc[:,16:32].head(3)
df.iloc[:,32:48].head(3)

# cek column term
df['term'] = df['term'].str.replace(' months', '')
df['term'] = df['term'].astype(float)

# cek column emp_length
df['emp_length'].unique()
# ubah label emp_length ke numerik
df['emp_length_int'] = df['emp_length'].str.replace('\+ years', '')
df['emp_length_int'] = df['emp_length_int'].str.replace('< 1 year', str(0))
df['emp_length_int'] = df['emp_length_int'].str.replace(' years', '')
df['emp_length_int'] = df['emp_length_int'].str.replace(' year', '')
df['emp_length_int'] = df['emp_length_int'].astype(float)
df.drop('emp_length', axis=1, inplace=True)

# cek column issue_d
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y')
# selisih terhitung hingga Juli 2022
df['months_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2022-07-01') - df['issue_d']) / np.timedelta64(1, 'M')))
# cek nilai negatif
any(df['months_since_issue_d']<0)
# hapus variable awal (format tanggal)
df.drop('issue_d', axis=1, inplace=True)

# cek earliest_cr_line
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')
# selisih terhitung hingga Juli 2022
df['months_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2022-07-01') - df['earliest_cr_line']) / np.timedelta64(1, 'M')))
# cek nilai negatif
any(df['months_since_earliest_cr_line']<0)
df.loc[df['months_since_earliest_cr_line']<0, 'months_since_earliest_cr_line'] = df['months_since_earliest_cr_line'].max()
# hapus variable awal (format tanggal)
df.drop(['earliest_cr_line'], axis=1, inplace=True)

# cek last_pymnt_d
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%y')
# selisih terhitung hingga Juli 2022
df['months_since_last_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2022-07-01') - df['last_pymnt_d']) / np.timedelta64(1, 'M')))
# cek nilai negatif
any(df['months_since_last_pymnt_d']<0)
# hapus variable awal (format tanggal)
df.drop('last_pymnt_d', axis=1, inplace=True)

# cek last_credit_pull_d
df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], format='%b-%y')
# selisih terhitung hingga Juli 2022
df['months_since_last_credit_pull_d'] = round(pd.to_numeric((pd.to_datetime('2022-07-01') - df['last_credit_pull_d']) / np.timedelta64(1, 'M')))
# cek nilai negatif
any(df['months_since_last_credit_pull_d']<0)
# hapus variable awal (format tanggal)
df.drop('last_credit_pull_d', axis=1, inplace=True)

# cek label
numerik = df.select_dtypes(exclude='object')
kategorik = df.select_dtypes(include='object')
df.head()

col_kat = kategorik.columns
col_num = numerik.columns

for i in col_kat:
  data = df[i]
  print('nunique variabel {} sebanyak {}'.format(i, data.nunique()))

print('')

for j in col_num:
  data = df[j]
  print('nunique variabel {} sebanyak {}'.format(j, data.nunique()))

col_kat = kategorik.columns

for i in col_kat:
  data = df[i]
  print('distribusi label variabel {} antara lain: \n{}'.format(i,data.value_counts()))
  print('')

df.drop(['policy_code', 'application_type', 'emp_title', 'title', 'pymnt_plan'],axis = 1, inplace= True)

# handling missing value
numerik = df.select_dtypes(exclude='object')
missing_numerik = numerik.isna().mean()*100
col_mis_num = missing_numerik[missing_numerik > 0].index
print(len(col_mis_num))

kategorik = df.select_dtypes(include='object')
missing_kategorik = kategorik.isna().mean()*100
col_mis_kat = missing_kategorik[missing_kategorik > 0].index
print(len(col_mis_kat))
# cek distribusi
plt.figure(figsize=(30,10))
for i,j in enumerate(col_mis_num):
  plt.subplot(4,4,i+1)
  sns.distplot(numerik[j])
  plt.tight_layout()

treat_bymedian =  ['open_acc', 'revol_util', 'total_acc', 'tot_cur_bal', 'months_since_earliest_cr_line', 'months_since_last_pymnt_d', 'months_since_last_credit_pull_d']
treat_bymodus = ['annual_inc', 'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'acc_now_delinq', 'tot_coll_amt', 'total_rev_hi_lim', 'emp_length_int', 'collections_12_mths_ex_med']

for i in treat_bymedian:
  df.loc[df.loc[:,i].isnull(),i] = df.loc[:,i].median()

for j in treat_bymodus:
  df.loc[df.loc[:,j].isnull(),j] = df.loc[:,j].mode()[0]

#cek ulang missing value
df.isna().mean()*100

# handling outlier
from scipy import stats

numerik = df.select_dtypes(exclude='object')
kategorik = df.select_dtypes(include='object')

numerik = numerik[(np.abs(stats.zscore(numerik)) < 3).all(axis=1)]
df = numerik.join(kategorik)
df.shape

# cek ulang bobot value numerik
numerik = df.select_dtypes(exclude='object')
for j in numerik.columns:
  data = df[j]
  print('nunique variabel {} sebanyak {}'.format(j, data.nunique()))
     
col_convert = ['term','pub_rec','delinq_2yrs','inq_last_6mths']

for i in col_convert:
  df[i] = df[i].astype(str)

df.drop(['collections_12_mths_ex_med', 'acc_now_delinq'], axis = 1, inplace =True)

# EDA
df.loc[df['bad_loan']==0,'status']='Lunas'
df.loc[df['bad_loan']==1,'status']='Gagal Bayar'

numerik = df.select_dtypes(exclude='object')
kategorik = df.select_dtypes(include='object')
     
# cek distribusi plot numerik
col = numerik.columns

plt.figure(figsize=(30,15))
for i in range(0,len(col)):
  plt.subplot(6,6,i+1)
  sns.distplot(numerik[numerik.columns[i]])
  plt.tight_layout()

numerik.describe().T

df['total_rec_late_fee_label'] = np.where(df['total_rec_late_fee']==0, 'None', 'Paid')
df['collection_recovery_fee_label'] = np.where(df['collection_recovery_fee']==0, 'None', 'Paid')
df['tot_coll_amt_label'] = np.where(df['tot_coll_amt']==0, 'None', 'Paid')
df['recoveries_label'] = np.where(df['recoveries']==0, 'None', 'Paid')

df.drop(['total_rec_late_fee','recoveries','collection_recovery_fee','tot_coll_amt'], axis=1, inplace=True)

# cek distribusi plot kategorik
col = kategorik.columns

plt.figure(figsize=(25,10))
for i in range(0,len(col)):
  plt.subplot(3,4,i+1)
  sns.countplot(kategorik[kategorik.columns[i]])
  plt.tight_layout()

df['home_ownership'].replace({'NONE':'RENT', 'ANY':'RENT', 'OTHER':'RENT'},inplace=True)

df['purpose'].replace({'educational':'major_purchase',
                         'house':'major_purchase',
                         'medical':'major_purchase',
                         'moving':'major_purchase',
                         'vacation':'other',
                         'wedding':'other',
                         'renewable_energy':'home_improvement'},inplace=True)

df['addr_state'].replace({'IA':'OTHER', 'ID':'OTHER', 'NE':'OTHER', 'ME':'OTHER'},inplace=True)

# univariate

# feature numerik
numerik = df.select_dtypes(exclude='object').drop('bad_loan',axis=1)
col = numerik.columns
df["all"] = ""

plt.figure(figsize=(30,13))
for i,j in enumerate(col):
  plt.subplot(4,7,i+1)
  sns.violinplot(x="all", y=j, hue="status", data=df, split=True)
  plt.xlabel("")
  plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
  plt.tight_layout()

df.drop('all', axis = 1, inplace = True)

# feature kategorik
kategorik = df.select_dtypes(include='object').drop(['status','purpose','addr_state'],axis=1)
col = kategorik.columns

fig = plt.figure(figsize=(32, 20))
for i,j in enumerate(col,start=1):
  ax = plt.subplot(3,4,i)
  pd.crosstab(df[j],df['status']).sort_values(by=['Lunas']).plot.bar(stacked=True,ax=ax)
  plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
  plt.xticks(rotation='0')

df.drop('collection_recovery_fee_label', axis=1, inplace=True)

# perbandingan gagal bayar dan lunas
fig = plt.figure(figsize=(20,15))
ax = plt.subplot(121)
pd.crosstab(df['purpose'], df['status']).sort_values(by=['Lunas']).plot.barh(stacked=True,ax=ax)
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
plt.xticks(rotation='0')

ax = plt.subplot(122)
pd.crosstab(df['addr_state'], df['status']).sort_values(by=['Lunas']).plot.barh(stacked=True,ax=ax)
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
plt.xticks(rotation='0')

# bivariate

# anova test
x = df.select_dtypes(exclude='object').drop('bad_loan',axis=1)
y = df['status']
col_num = x.columns

from sklearn.feature_selection import f_classif
fval,pval = f_classif(x,y)
for i,j in enumerate(col_num): 
  print('pvalue variabel {} = {}'.format(j,round(pval[i],3)))
     
# chi-square test
x = df.select_dtypes(include='object').drop('status',axis=1)
y = df['status']
col_kat = x.columns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
for i in col_kat:
  x_cat = LabelEncoder().fit_transform(df[i]).reshape(-1,1)
  fval,pval = chi2(x_cat,y)
  print('pvalue variabel {} = {}'.format(i,pval))   

df.drop(['pub_rec'], axis=1, inplace=True)

# multivariate
data = df.select_dtypes(exclude='object').drop('bad_loan',axis=1)

corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(30, 20))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.9, center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# feature selection
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
high_corr = [column for column in upper.columns if any(upper[column] > 0.7)]
high_corr

df.drop(high_corr, axis=1, inplace=True)

# data prepocessing

# label encoding
df['grade'] = df['grade'].astype('category').cat.codes

# one-hot encoding
cat = df.select_dtypes(include='object').drop('status',axis=1)
kategorik_col = cat.columns.tolist()

onehot = pd.get_dummies(df[kategorik_col], drop_first=True)
kategorik = pd.concat([onehot,df['grade']],axis=1)

# data splitting
from sklearn.model_selection import train_test_split

numerik = df.select_dtypes(exclude='object')
dataset = pd.concat([numerik,kategorik],axis=1)

X = dataset.drop('bad_loan', axis = 1)
y = dataset['bad_loan']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# transformer
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

col_minmax = ['dti', 'open_acc', 'revol_util', 'total_acc', 'months_since_earliest_cr_line']
col_log = ['loan_amnt', 'int_rate', 'annual_inc', 'revol_bal', 'out_prncp', 'last_pymnt_amnt',
           'tot_cur_bal', 'total_rev_hi_lim', 'emp_length_int', 'months_since_issue_d',
           'months_since_last_pymnt_d', 'months_since_last_credit_pull_d']

#MinMax Transformation
minmaxSC = MinMaxScaler()
X_train.loc[:, col_minmax] = minmaxSC.fit_transform(X_train.loc[:, col_minmax])
X_test.loc[:, col_minmax] = minmaxSC.transform(X_test.loc[:, col_minmax])

#Log Transformation
X_train.loc[:, col_log] = np.log1p(X_train.loc[:, col_log])
X_test.loc[:, col_log] = np.log1p(X_test.loc[:, col_log])

# cek under/oversampling 
y_train.value_counts()

# undersampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

under_sampler = RandomOverSampler(random_state=0)

X_resampled, y_resampled = under_sampler.fit_resample(X_train.values, y_train.ravel())
Counter(y_resampled)

# return bentuk array ke dataframe
col = X_train.columns.to_list()

X_train = pd.DataFrame(X_resampled, 
             columns=col)

y_train = pd.Series(y_resampled)

# modelling

# komparasi algorithm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

RFclassifier = RandomForestClassifier()
RFclassifier.fit(X_train, y_train)
y_pred_rf = RFclassifier.predict(X_train)
y_pred_rf_test = RFclassifier.predict(X_test)

LRclassifier = LogisticRegression()
LRclassifier.fit(X_train, y_train)
y_pred_lr = LRclassifier.predict(X_train)
y_pred_lr_test = LRclassifier.predict(X_test)

XTclassifier = ExtraTreesClassifier()
XTclassifier.fit(X_train, y_train)
y_pred_xt = XTclassifier.predict(X_train)
y_pred_xt_test = XTclassifier.predict(X_test)

DTclassifier = DecisionTreeClassifier()
DTclassifier.fit(X_train, y_train)
y_pred_dt = DTclassifier.predict(X_train)
y_pred_dt_test = DTclassifier.predict(X_test)

GBclassifier = GradientBoostingClassifier()
GBclassifier.fit(X_train, y_train)
y_pred_gb = GBclassifier.predict(X_train)
y_pred_gb_test = GBclassifier.predict(X_test)

# cek akurasi test train
from sklearn.metrics import accuracy_score

algorithm = ['RandomForest','LogisticRegression','Xtree','DecisionTree','GradientBoost']
pred_train = [y_pred_rf, y_pred_lr, y_pred_xt, y_pred_dt, y_pred_gb]
pred_test = [y_pred_rf_test, y_pred_lr_test, y_pred_xt_test, y_pred_dt_test, y_pred_gb_test]

train_set_accuracy = []
test_set_accuracy = []

for i in pred_train:
  train_set_accuracy.append(accuracy_score(y_train, i))

for i in pred_test:
  test_set_accuracy.append(accuracy_score(y_test, i))

n = list(zip(algorithm, train_set_accuracy, test_set_accuracy))
pd.DataFrame(n, columns = ['Model','Akurasi Train', 'Akurasi Test']).sort_values(['Akurasi Train'],ascending=False)

# feature importance
feature_importance = pd.Series(RFclassifier.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(15,20))
feature_importance.plot(kind='barh', title='Feature Importance of Random Forest Model')

# feature selection
X_train.columns

col = ['months_since_last_pymnt_d','last_pymnt_amnt','recoveries_label_Paid','out_prncp','months_since_last_credit_pull_d','months_since_issue_d',
       'int_rate','annual_inc','loan_amnt','grade','tot_cur_bal','dti','total_rev_hi_lim','revol_bal','revol_util','months_since_earliest_cr_line',
       'total_acc','open_acc','emp_length_int','total_rec_late_fee_label_Paid','term_60.0','initial_list_status_w','home_ownership_RENT']

# cek overfitting pada feature selection

X_train2 = X_train[col]
X_test2 = X_test[col]

RFclassifier = RandomForestClassifier()
RFclassifier.fit(X_train2, y_train)
y_pred_rf = RFclassifier.predict(X_train2)
y_pred_rf_test = RFclassifier.predict(X_test2)

print('Akurasi Train',accuracy_score(y_train, y_pred_rf))
print('Akurasi Test',accuracy_score(y_test, y_pred_rf_test))

# model evaluation
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf_test)

print('\nTrue Positives(TP) = ', cm[0,0])
print('True Negatives(TN) = ', cm[1,1])
print('False Positives(FP) = ', cm[0,1])
print('False Negatives(FN) = ', cm[1,0])
print('\n')

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

TP = cm[0,0] #true positif
TN = cm[1,1] #true negatif
FP = cm[0,1] #false positif
FN = cm[1,0] #false negatif

# cek akurasi, presisi, dan sensitifitas
accuracy = (TP+TN) / float(TP+TN+FP+FN)
print('Akurasi Klasifikasi : {0:0.4f}'.format(accuracy))

class_error = (FP+FN) / float(TP+TN+FP+FN)
print('\nKesalahan Klasifikasi : {0:0.4f}'.format(class_error))

precision = TP / float(TP + FP)
print('\nPresisi : {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)
print('\nSensitifitas : {0:0.4f}'.format(recall))

# cek roc-auc score
from sklearn.metrics import roc_curve, precision_recall_curve,auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_rf_test)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

# plot roc-auc score
plt.figure(figsize=(8,8))

plt.title('ROC - AUC plot')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
