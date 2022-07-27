'''BP结果展示，归一化之后'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error
from sklearn.pipeline import Pipeline
from pickle import dump
from pickle import load

font1 = {'family': 'Times New Roman','weight': 'normal','size': 13,}
font2 = {'family': 'STSong','weight': 'normal','size': 13,}
fontsize1=13

# 设置字体，以作图显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
# 设置显示属性
pd.set_option('display.width',20000)
pd.set_option('display.max_columns',200000)
pd.set_option('display.max_rows',10)
# pd.set_option('display.width',100)           #宽度
np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True)
pd.set_option('precision',4)
np.set_printoptions(precision=4)


def MSE(y_pre_train,Y_train):#均方误差
    return sum((y_pre_train-Y_train)**2)/len(Y_train)

def R2(y_pre,y_real):#决定系数
    # return sum((y_pre - y_real.mean()) ** 2) / sum((y_real - y_real.mean()) ** 2)
    # print(y_real.mean())
    return 1-(sum((y_real-y_pre)**2)/sum((y_real-y_real.mean())**2))

def MAE(y_pre,y_real):#平均绝对误差
    return sum(abs(y_real-y_pre))/len(y_real)       #abs为绝对值

def PED(y_pre,y_real,detal = 0.05): #PED是误差分布直方图
    error = abs(y_real - y_pre)
    num = 0
    for i in error:
        if i <= detal:
            num = num+1
    ped = num / len(error) * 100

    return ped


def figure_plot(data1,data2,key_label,figure_property):
    # 折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data1,'-*',label = '训练集')
    ax.plot(data2,'-s',label = '测试集')
    # x_ticks = ax.set_xticks([i for i in range(len(key_label))])
    # x_labels = ax.set_xticklabels(key_label,rotation=45,fontdict=font1)
    ax.set_title(figure_property['title'],fontdict=font2)
    ax.set_xlabel(figure_property['X_label'],fontdict=font2)
    ax.set_ylabel(figure_property['Y_label'],fontdict=font2)
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # #     y_ticks=ax.set_yticks([])
    # #     y_labels=ax.set_yticklabels([-20+i for i in range(20)],rotation=0,fontsize=14)
    plt.grid()
    plt.savefig('../fig/{}.jpg'.format(figure_property['title']),dpi=500,bbox_inches = 'tight')#保存图片
    plt.legend()
    plt.show()

# 导入数据

filename = '../data/Molecular_Descriptor.xlsx'
train_data = pd.read_excel(filename,sheet_name= 'training',header=0)
x = ['MDEO-12' , 'SHBint10' , 'minssO' , 'ATSc3' , 'TopoPSA' , 'VC-5' , 'MLFER_A' , 'minHBint5' , 'MLogP' , 'minsOH' , 'nHBAcc' , 'nC' , 'BCUTc-1l' , 'minHsOH' , 'C1SP2' , 'maxssO' , 'minsssN' , 'maxHsOH' , 'LipoaffinityIndex' , 'MDEC-23']
y = 'pIC50'

Input_data = train_data[x]
Out_data = train_data[y]

# 分离数据集转数组
X  = np.array(Input_data)
Y = np.array(Out_data)

print('X:{},Y:{}'.format(X.shape,Y.shape))
#
#
validation_size = 0.2  #测试集划分


'''=================划分数据集，80%训练，20%测试 ====================='''
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size,random_state=42)
print(X_train.shape, X_validation.shape, Y_train.shape, Y_validation.shape)

# pipeline算法：https://blog.csdn.net/weixin_40161254/article/details/89446293
#mlpregressor多层分类器：https://blog.csdn.net/xspyzm/article/details/102832206
#  激活函数选项：activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
model = Pipeline([('Scaler',StandardScaler()),('MLP',MLPRegressor(hidden_layer_sizes=1000,
                                                                 activation='tanh',solver='adam',
                                                                 batch_size='auto',
                                                                 learning_rate='constant',
                                                                 learning_rate_init=0.01,
                                                                 power_t=0.5, max_iter=1000,
                                                                 shuffle=True,
                                                                 random_state=None,
                                                                 tol=0.0001,
                                                                 verbose=False,
                                                                 warm_start=False,
                                                                 momentum=0.9,
                                                                 nesterovs_momentum=True,
                                                                 early_stopping=False,
                                                                 validation_fraction=0.1,
                                                                 beta_1=0.9, beta_2=0.999,
                                                                 epsilon=1e-08,
                                                                 n_iter_no_change=10))])


clf = model.fit(X_train,Y_train)

# 保存模型
model_file = '../code/final_MLP_model.sav'
with open(model_file, 'wb') as model_f:
    dump(clf, model_f)

predict_train1 = clf.predict(X_train)
predict_test1 = clf.predict(X_validation)

'''结果对比'''
# 训练集输出
print('训练集结果')
mse_train = MSE(predict_train1,Y_train)
mae_train = MAE(predict_train1,Y_train)
'''R21'''
error1 = Y_train - predict_train1
error2 = Y_train - mean(Y_train)
R2_train  = 1-(error1.T @ error1)/(error2.T @ error2)


'''测试集输出'''
print('测试集预测结果')
mse_test = MSE(predict_test1,Y_validation)
mae_test = MAE(predict_test1,Y_validation)

'''R2'''
error1 = Y_validation - predict_test1
error2 = Y_validation-mean(Y_validation)
R2_test = 1-(error1.T @ error1)/(error2.T @ error2)

train_result = pd.DataFrame([mse_train,mae_train,R2_train,],columns=['train'],index=['mse','mae','R2']).T
test_result = pd.DataFrame([mse_test,mae_test,R2_test],columns=['test'],index=['mse','mae','R2']).T
result  = pd.concat([train_result,test_result],axis=0)
print(result)

# 作图展示
# 训练集
key_label = np.arange(len(predict_train1))
figure_property={'title':'BP训练集结果对比','X_label':'X','Y_label':'Y'}
figure_plot(predict_train1,Y_train,key_label,figure_property)

# 测试集
key_label = np.arange(len(predict_test1))
figure_property={'title':'BP测试集结果对比','X_label':'X','Y_label':'Y'}
figure_plot(predict_test1,Y_validation,key_label,figure_property)

