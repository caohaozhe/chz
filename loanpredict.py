import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
#读训练集、测试集
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
pd.set_option('display.max_columns',10)
#-------------------------------数据预处理------------------------------------

#拷贝原始数据
# 这样即使我们必须对这些数据集进行任何更改，我们也不会丢失原始数据集

train_original = train.copy()
test_original = test.copy()
#看一下 train set 的前 5 行，注意“Loan_Status”列
print(train.head())

#看一下测试集的前5行，注意没有我们将预测的“贷款状态”
print(test.head())
#看数据格式
print(train.shape, test.shape)
# 计算训练集测试集划分比例
print(train.shape[0]/(train.shape[0]+test.shape[0]),
      test.shape[0]/(train.shape[0]+test.shape[0]))
#查看数据集中的特征（即自变量）
print(train.columns, test.columns)
# 查看数据类型
print(train.dtypes)
#数据集的简明摘要，有关索引 dtype、列 dtype、非空值和内存使用情况的信息
print(train.info())

#-------------------------单变量分析------------------------------------------------
#变量的频率表将为我们提供该变量中每个类别的计数
print(train['Loan_Status'].value_counts())
#百分比分布可以通过设置 normalize=True 来显示比例而不是数字来计算
print(train['Loan_Status'].value_counts(normalize=True))
#条形图以可视化频率
print(train['Loan_Status'].value_counts().plot.bar())
plt.show()

#----------------------自变量（分类）----------------------------
#可视化分类特征
# plt.figure(1)
plt.subplot(231)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')

plt.subplot(232)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')

plt.subplot(233)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')

plt.subplot(234)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')

plt.subplot(235)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')

plt.show()

#----------------------------自变量（序数）--------------------------------
# 可视化剩余的分类特征
# plt.figure(1)
plt.subplot(121)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(12,4), title= 'Dependents')

plt.subplot(122)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')

plt.show()

#-----------------------------自变量（数值）----------------------------------
#可视化申请人收入
# plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))

plt.show()

# 按教育来区分它们：
train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")
plt.show()

# 其次，让我们看一下共同申请人的收入分配。
# plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);

plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))

plt.show()

#第三，让我们看一下 LoanAmount 变量的分布。
# plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['LoanAmount']);

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))

plt.show()
#变量的频率表将为我们提供该变量中每个类别的计数
print(train['Loan_Amount_Term'].value_counts())
# plot bar chart
train['Loan_Amount_Term'].value_counts(normalize=True).plot.bar(title= 'Loan_Amount_Term')
plt.show()

#------------------------双变量分析--------------------------------
#分类自变量与目标变量
#申请人的性别是否会对批准机会产生任何影响
print(pd.crosstab(train['Gender'],train['Loan_Status']))

Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis = 0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Gender')
p = plt.ylabel('Percentage')
plt.show()


print(pd.crosstab(train['Married'],train['Loan_Status']))

Married = pd.crosstab(train['Married'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Married')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Dependents'],train['Loan_Status']))

Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Dependents')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Education'],train['Loan_Status']))

Education=pd.crosstab(train['Education'],train['Loan_Status'])
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Education')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Self_Employed'],train['Loan_Status']))

Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Self_Employed')
p = plt.ylabel('Percentage')
plt.show()


print(pd.crosstab(train['Credit_History'],train['Loan_Status']))

Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.xlabel('Credit_History')
p = plt.ylabel('Percentage')
plt.show()

print(pd.crosstab(train['Property_Area'],train['Loan_Status']))

Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Property_Area')
P = plt.ylabel('Percentage')
plt.show()


#-------------------数值自变量与目标变量-----------------------------------
print(train.groupby('Loan_Status')['ApplicantIncome'].mean())

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
plt.show()
#划分收入变量
bins = [0,2500,4000,6000,81000]
group = ['Low','Average','High', 'Very high']
train['Income_bin'] = pd.cut(df['ApplicantIncome'],bins,labels=group)
# 看下训练集
print(train.head(8))
#重新分组后看收入是否有差异
print(pd.crosstab(train['Income_bin'],train['Loan_Status']))

Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')
plt.show()

#同样的方式分析CoapplicantIncome
bins = [0,1000,3000,42000]
group = ['Low','Average','High']
train['Coapplicant_Income_bin'] = pd.cut(df['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin = pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('CoapplicantIncome')
P = plt.ylabel('Percentage')
plt.show()

#构建新特征
train['Total_Income'] = train['ApplicantIncome'] + train['CoapplicantIncome']

#根据总收入分类
bins = [0,2500,4000,6000,81000]
group = ['Low','Average','High', 'Very high']
train['Total_Income_bin'] = pd.cut(train['Total_Income'],bins,labels=group)


Total_Income_bin = pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')
plt.show()

#删除 bins
train = train.drop(['Income_bin', 'Coapplicant_Income_bin',  'Total_Income_bin', 'Total_Income'], axis=1)

#  3+ 替换为3
train['Dependents'].replace('3+', 3, inplace=True)
test['Dependents'].replace('3+', 3, inplace=True)

# y=1,n=0
train['Loan_Status'].replace('N', 0, inplace=True)
train['Loan_Status'].replace('Y', 1, inplace=True)

# 检查下
print(train.head())
# train = train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
#计算相关矩阵 pearson
matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
#画个热图可视化
sns.heatmap(matrix, vmax=1, square=True, cmap="BuPu", annot=True)
print(matrix)
plt.show()

#--------------------------------数据预处理------------------------------------
#----------------------------------缺失值插补-------------------------
print(train.isnull().sum())

# mode 填补
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
# 连续型用中位数填补
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
# 看看
print(train.isnull().sum())
# replace missing values in Test set with mode/median from Training set
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# log变换
train['LoanAmount_log'] = np.log(train['LoanAmount'])
test['LoanAmount_log'] = np.log(test['LoanAmount'])
# after log transformation

ax1 = plt.subplot(121)
train['LoanAmount_log'].hist(bins=20, figsize=(12,4))
ax1.set_title("Train")

ax2 = plt.subplot(122)
test['LoanAmount_log'].hist(bins=20)
ax2.set_title("Test")
plt.show()


#------------------------logistic regression--------------------------------------------


# 删除 Loan_ID 没啥用
train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)
# Sklearn 需要单独数据集中的目标变量。 因此，从训练数据集中删除目标变量并将其保存在另一个数据集中。
X = train.drop('Loan_Status', 1)
y = train.Loan_Status
# adding dummies to the dataset
X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(X.shape, train.shape, test.shape)
print(X.head())

from sklearn.model_selection import train_test_split
# split the data into train and cross validation set
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=0)

# take a look at the dimension of the data
print(x_train.shape, x_cv.shape, y_train.shape, y_cv.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 拟合模型
model = LogisticRegression()
model.fit(x_train, y_train)
# 预测
pred_cv = model.predict(x_cv)
print('logistic预测准确率：',accuracy_score(y_cv, pred_cv))

#输出混淆矩阵
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_cv, pred_cv)
print(cm)
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_cv, pred_cv))
# #测试集做预测
pred_test = model.predict(test)
print(pred_test[:50])

# submission = pd.read_csv("Sample_Submission_ZAuTl8O_FK3zQHh.csv")
# submission['Loan_Status'] = pred_test
# submission['Loan_ID'] = test_original['Loan_ID']
# # "N" and "Y" 替换回去
# submission['Loan_Status'].replace(0, 'N', inplace=True)
# submission['Loan_Status'].replace(1, 'Y', inplace=True)
# print(submission.head())
# #转成csv
# submission.to_csv('logistic.csv', index=False)
#--------------------------分层 k 折交叉验证的逻辑回归---------------------------
from sklearn.model_selection import StratifiedKFold
#k=5，且对数据的每个分层进行shuffle

mean_accuracy = []
i = 1
kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X, y):
      print('\n{} of kfold {}'.format(i, kf.n_splits))
      xtr, xvl = X.loc[train_index], X.loc[test_index]
      ytr, yvl = y[train_index], y[test_index]

      model = LogisticRegression(random_state=1)
      model.fit(xtr, ytr)
      pred_test = model.predict(xvl)
      score = accuracy_score(yvl, pred_test)
      mean_accuracy.append(score)
      print('accuracy_score', score)
      i+=1
print("\n LR Mean validation accuracy: ", sum(mean_accuracy)/len(mean_accuracy))
# make prediction on test set
pred_test = model.predict(test)


# calculate probability estimates of loan approval
# column 0 is the probability for class 0 and column 1 is the probability for class 1
# probability of loan default = 1 - model.predict_proba(test)[:,1]
pred = model.predict_proba(xvl)[:,1]
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl,  pred)
auc = metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()

#-------------------------决策树--------------------------------------
from sklearn import tree
mean_accuracy = []
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
      print('\n{} of kfold {}'.format(i,kf.n_splits))
      xtr,xvl = X.loc[train_index],X.loc[test_index]
      ytr,yvl = y[train_index],y[test_index]

      model = tree.DecisionTreeClassifier(random_state=1)
      model.fit(xtr, ytr)
      pred_test = model.predict(xvl)
      score = accuracy_score(yvl,pred_test)
      mean_accuracy.append(score)
      print('accuracy_score',score)
      i+=1

print("\n DTMean validation accuracy: ", sum(mean_accuracy)/len(mean_accuracy))
# pred_test = model.predict(test)



#------------------- Random Forest--------------------------------
from sklearn.ensemble import RandomForestClassifier
mean_accuracy = []
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X, y):
      print('\n{} of kfold {}'.format(i,kf.n_splits))
      xtr,xvl = X.loc[train_index],X.loc[test_index]
      ytr,yvl = y[train_index],y[test_index]

      model = RandomForestClassifier(random_state=1, max_depth=10, n_estimators=10)
      model.fit(xtr, ytr)
      pred_test = model.predict(xvl)
      score = accuracy_score(yvl,pred_test)
      mean_accuracy.append(score)
      print('accuracy_score',score)
      i+=1

print("\n RF Mean validation accuracy: ", sum(mean_accuracy)/len(mean_accuracy))
pred_test = model.predict(test)

#---------------------网格搜索优化超参数------------------------------
# --------------------------GridSearchCV-------------------------
from sklearn.model_selection import GridSearchCV
paramgrid1= {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
#默认3折交叉验证cv=3
grid_search1 = GridSearchCV(RandomForestClassifier(random_state=1), paramgrid1)
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size =0.3, random_state=1)
grid_search1.fit(x_train, y_train)
print(grid_search1.best_estimator_)

mean_accuracy = []
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
      print('\n{} of kfold {}'.format(i,kf.n_splits))
      xtr,xvl = X.loc[train_index],X.loc[test_index]
      ytr,yvl = y[train_index],y[test_index]

      model = RandomForestClassifier(random_state=1, max_depth=7, n_estimators=41)
      model.fit(xtr, ytr)
      pred_test = model.predict(xvl)
      score = accuracy_score(yvl,pred_test)
      mean_accuracy.append(score)
      print('accuracy_score',score)
      i+=1

print("\n RF-CV Mean validation accuracy: ", sum(mean_accuracy)/len(mean_accuracy))
pred_test = model.predict(test)
pred2=model.predict_proba(test)[:,1]
# --------------------------特征重要性----------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(12,8))
plt.show()

#------------------------------XGBoost--------------------------------------
from xgboost import XGBClassifier
mean_accuracy = []
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
      print('\n{} of kfold {}'.format(i,kf.n_splits))
      xtr,xvl = X.loc[train_index],X.loc[test_index]
      ytr,yvl = y[train_index],y[test_index]

      model = XGBClassifier(random_state=1, n_estimators=50, max_depth=4,verbosity = 0)
      model.fit(xtr, ytr)
      pred_test = model.predict(xvl)
      score = accuracy_score(yvl,pred_test)
      mean_accuracy.append(score)
      print('accuracy_score',score)
      i+=1

print("\n XGBoost Mean validation accuracy: ", sum(mean_accuracy)/len(mean_accuracy))
pred_test = model.predict(test)
pred3=model.predict_proba(test)[:,1]

#优参
paramgrid2 = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}
grid_search2 = GridSearchCV(XGBClassifier(random_state=1), paramgrid2)
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size =0.3, random_state=1)
grid_search2.fit(x_train, y_train)
print(grid_search2.best_estimator_)
#优参后k折
mean_accuracy = []
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
      print('\n{} of kfold {}'.format(i,kf.n_splits))
      xtr,xvl = X.loc[train_index],X.loc[test_index]
      ytr,yvl = y[train_index],y[test_index]

      model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                            colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                            gamma=0, gpu_id=-1, importance_type=None,
                            interaction_constraints='', learning_rate=0.300000012,
                            max_delta_step=0, max_depth=1, min_child_weight=1,
                            monotone_constraints='()', n_estimators=121, n_jobs=16,
                            num_parallel_tree=1, predictor='auto', random_state=1,
                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                            tree_method='exact', validate_parameters=1, verbosity=None)
      model.fit(xtr, ytr)
      pred_test = model.predict(xvl)
      score = accuracy_score(yvl,pred_test)
      mean_accuracy.append(score)
      print('accuracy_score',score)
      i+=1

print("\n XG-CVMean validation accuracy: ", sum(mean_accuracy)/len(mean_accuracy))
pred_test = model.predict(test)
pred3 = model.predict_proba(test)[:,1]
# -------------------------catboost-----------------------------------
from catboost import CatBoostClassifier
mean_accuracy = []
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
      print('\n{} of kfold {}'.format(i,kf.n_splits))
      xtr,xvl = X.loc[train_index],X.loc[test_index]
      ytr,yvl = y[train_index],y[test_index]

      model = CatBoostClassifier(learning_rate=0.03)
      model.fit(xtr, ytr)
      pred_test = model.predict(xvl)
      score = accuracy_score(yvl,pred_test)
      mean_accuracy.append(score)
      print('accuracy_score',score)
      i+=1

print("\n CatBoost Mean validation accuracy: ", sum(mean_accuracy)/len(mean_accuracy))

x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size =0.3, random_state=1)
grid_search2.fit(x_train, y_train)

model = CatBoostClassifier()

grid = {'learning_rate': [0.01,0.02,0.03,0.07],
        'depth': [4, 6, 8,10,16],
        'l2_leaf_reg': [1, 3]}

randomized_search_result = model.randomized_search(grid,
                                                   X=x_train,
                                                   y=y_train,
                                                   plot=True)
