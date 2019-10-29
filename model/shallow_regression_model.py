#coding:utf-8
'''
本文完成SVR以及LR的特征抽取、模型构建、训练、效果评测

'''
import sys
sys.path.extend(['..','.'])
from paths import PATH
from basic_config import *

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


## 首先抽取特征,根据数据集构建训练集，测试集
def construct_datasets(pathObj,m,n):

    testing_ids = set(pathObj.read_file(pathObj._testing_pid_path))
    validing_ids = set(pathObj.read_file(pathObj._validing_pid_path))

    pid_features = pathObj.loads_json(pathObj.dataset_feature_path(m,n))

    ## 抽取特征
    train_X = []
    train_Y = []

    test_X = []
    test_Y = []

    valid_X = []
    valid_Y = []

    for pid in pid_features.keys():

        ## 将所有的特征串联起来
        feature = pid_features[pid]

        X=[]
        Y=feature['Y']

        ##文章被引用的历史
        X.extend(feature['hist_cits'])
        ## 作者hindex
        X.extend(feature['a-first-hix'])
        X.extend(feature['a-avg-hix'])
        ## 作者文章数量
        X.extend(feature['a-first-pnum'])
        X.extend(feature['a-avg-pnum'])
        ## 作者数量
        X.append(feature['a-num'])
        X.append(feature['a-career-length'])
        ## 机构影响力 
        X.extend(feature['i-avg-if'])
        ## 期刊影响力
        X.extend(feature['v-if'])
        ## 背景
        X.extend(features['b-num'])

        if pid in testing_ids:
            test_X.append(X)
            test_Y.append(Y)
        elif pid in validing_ids:
            valid_X.append(X)
            valid_Y.append(Y)
        else:
            train_X.append(X)
            train_Y.append(Y)

    logging.info('{} of training dataset, {} of testing dataset, {} of valid dataset.'.format(len(train_X),len(test_X),len(valid_X)))

    return train_X,train_Y,test_X,test_Y,valid_X,valid_Y

## 训练模型
def train_SVR(train_X,train_Y):

    ## 对于序列预测来讲，每一个时间点需要训练一个模型，因此需要根据Y的宽度确定模型的数量
    size = len(train_Y[0])
    logging.info('length of y is {}.'.format(size))
    ## 每一列都是对应时间序列的对应的Y值
    train_Ys = zip(*train_Y)
    train_X = np.array(train_X)

    models = []
    ## 每一列训练一个model
    for i,train_y in enumerate(train_Ys):
        logging.info('train model for position {} ...'.format(i))

        train_y = np.array(train_y)

        lr = LinearRegression().fit(train_X, train_y)

        models.append(lr)


    return models



##评测模型
def evaluate_model(models,test_X,test_Y):

    predict_Y = []
    for model in models:
        predict_Y.append(model.predict(test_X))

    predict_Y = zip(predict_Y)

    ## 衡量predict_Y和test_Y之间的关系

    return r2_score(test_Y, predict_Y, multioutput='variance_weighted'),mean_absolute_error(test_Y, predict_Y),mean_squared_error(test_Y, predict_Y)


def train_and_evaluate(pathObj,mn_list):
    ## m n list
    for m,n in mn_list:
        logging.info('train dataset sip-m{}n{} ..'.format(m,n))

        train_X,train_Y,test_X,test_Y,valid_X,valid_Y = construct_datasets(pathObj,m,n)

        print(train_X[:2])
        print(train_Y[:2])

        models = train_SVR(train_X,train_Y)

        r2,mae,mse =evaluate_model(models,test_X,test_Y)

        print('R^2:{},MAE:{},MSE:{}'.format(r2,mae,mse))


if __name__ == '__main__':
    
    

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    mn_list = [(3,1)]

    train_and_evaluate(pathObj,mn_list)




