'''MLR1模型使用'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error,mean_squared_error
from pickle import dump
from pickle import load


font1 = {'family': 'Times New Roman','weight': 'normal','size': 13,}
font2 = {'family': 'STSong','weight': 'normal','size': 13,}
fontsize1=13

# 设置字体，以作图显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
# 设置显示属性
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',100)
pd.set_option('display.width',100)           #宽度
np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True)
pd.set_option('precision',4)
np.set_printoptions(precision=4)

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
train_data = pd.read_excel(filename,sheet_name= 'test',header=0)
x = ['MDEO-12' , 'SHBint10' , 'minssO' , 'ATSc3' , 'TopoPSA' , 'VC-5' , 'MLFER_A' , 'minHBint5' , 'MLogP' , 'minsOH' , 'nHBAcc' , 'nC' , 'BCUTc-1l' , 'minHsOH' , 'C1SP2' , 'maxssO' , 'minsssN' , 'maxHsOH' , 'LipoaffinityIndex' , 'MDEC-23']

Input_data = train_data[x]

# 分离数据集
X  = np.array(Input_data)


print('X:{}'.format(X.shape))


# 加载模型
model_file = 'final_MLP_model.sav'
with open(model_file, 'rb') as model_f:
    loaded_model = load(model_f)
    Y_predict =  loaded_model.predict(X)

# 保存结果到EXCEl
data = pd.DataFrame(Y_predict,columns=['预测值'])
data.to_excel('../data/BP预测值.xlsx')




