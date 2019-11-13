#coding:utf-8
'''
本文完成RANDOM FOREST RE以及LR的特征抽取、模型构建、训练、效果评测

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
from sklearn.ensemble import RandomForestRegressor


## 首先抽取特征,根据数据集构建训练集，测试集
def construct_datasets(pathObj,m,n,scale=True):

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

    test_sorted_ids = []

    for pid in pid_features.keys():

        ## 将所有的特征串联起来
        feature = pid_features[pid]

        X=[]
        Y=[float(y) for y in feature['Y']]

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
        X.extend(feature['b-num'])

        X = [float(x) for x in X]

        if pid in testing_ids:
            test_sorted_ids.append(pid)
            test_X.append(X)
            test_Y.append(Y)
        elif pid in validing_ids:
            valid_X.append(X)
            valid_Y.append(Y)
        else:
            train_X.append(X)
            train_Y.append(Y)

    logging.info('{} of training dataset, {} of testing dataset, {} of valid dataset.'.format(len(train_X),len(test_X),len(valid_X)))
    
    train_X,test_X,valid_X,train_X_mean,train_X_std = scale_dataset(train_X,test_X,valid_X,scale)
    train_Y,test_Y,valid_Y,train_Y_mean,train_Y_std = scale_dataset(train_Y,test_Y,valid_Y,scale)

    return train_X,train_Y,test_X,test_Y,valid_X,valid_Y,test_sorted_ids


def scale_dataset(train,test,valid,scale=True):

    train = np.array(train)
    test = np.array(test)
    valid = np.array(valid)

    if scale:

        logging.info('as array done. shape of array is {}'.format(train.shape))
        mean = np.mean(train,axis=0)
        std = np.std(train,axis=0)

    else:

        mean = 0
        std = 1

    return (train-mean)/std,(test-mean)/std,(valid-mean)/std,mean,std
    # return train,test,valid

def unscale_dataset(train,result):

    train = np.array(train)

    mean = np.mean(train,axis=0)
    std = np.std(train,axis = 0)

    return result*std+mean


## 训练模型
def train_Linear(train_X,train_Y):

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

## 训练模型
def train_RFR(train_X,train_Y):

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

        # svr = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=.1)

        regr = RandomForestRegressor(max_depth=3, random_state=0,n_estimators=100,n_jobs=-1)
        regr.fit(train_X, train_y)

        models.append(regr)

    return models



##评测模型
def evaluate_model(models,test_X,test_Y):

    predict_Y = []
    for model in models:
        predict_Y.append(list(model.predict(test_X)))

    predict_Y = list(zip(*predict_Y))

    # print(test_Y[:2])
    # print(predict_Y[:2])

    ## 衡量predict_Y和test_Y之间的关系

    return r2_score(test_Y, predict_Y, multioutput='variance_weighted'),mean_absolute_error(test_Y, predict_Y),mean_squared_error(test_Y, predict_Y),predict_Y


def train_and_evaluate(pathObj,mn_list,scale=False):

    ## m n list
    lines = ['dataset,model,r2,mae,mse']
    shallow_result = defaultdict(dict)
    for m,n in mn_list:
        dataset = 'sip-m{}n{}'.format(m,n)
        logging.info('train dataset sip-m{}n{} ..'.format(m,n))
        train_X,train_Y,test_X,test_Y,valid_X,valid_Y,test_sorted_ids = construct_datasets(pathObj,m,n,scale=scale)
        
        shallow_result[dataset]['IDS'] = test_sorted_ids

        # print(train_X[:2])
        # print(train_Y[:2])

        models = train_RFR(train_X,train_Y)
        r2,mae,mse,predict_Y =evaluate_model(models,test_X,test_Y)

        shallow_result[dataset]['RFR']=predict_Y

        print('RFR====R^2:{},MAE:{},MSE:{}'.format(r2,mae,mse))

        lines.append('sip-m{}n{},{},{},{},{}'.format(m,n,'RFR',r2,mae,mse))

        models = train_Linear(train_X,train_Y)
        r2,mae,mse,predict_Y =evaluate_model(models,test_X,test_Y)
        shallow_result[dataset]['LR']=predict_Y

        lines.append('sip-m{}n{},{},{},{},{}'.format(m,n,'LR',r2,mae,mse))

        print('Linear====R^2:{},MAE:{},MSE:{}'.format(r2,mae,mse))

    open(pathObj._shallow_result_summary,'w').write('\n'.join(lines))

    logging.info('result summary saved.')

    open(pathObj._shallow_testing_prediction_result,'w').write(json.dumps(shallow_result))

    logging.info('result saved.')

## 查看一下全部预测为0的效果
def test_zero(pathObj,m,n):

    train_X,train_Y,test_X,test_Y,valid_X,valid_Y,test_sorted_ids = construct_datasets(pathObj,m,n,False)



    test_Y = np.array(test_Y)

    fake_Y = np.zeros((test_Y.shape[0],test_Y.shape[1]))

    train,test,fake = scale_dataset(train_Y,test_Y,fake_Y)


    print(m,n)

    print('r2',r2_score(test,fake))
    print('mae',mean_absolute_error(test,fake))

    print('mse',mean_squared_error(test,fake))




if __name__ == '__main__':
    

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    mn_list=[(3,1),(3,3),(3,5),(3,10),(5,1),(5,3),(5,5),(5,10)]

    train_and_evaluate(pathObj,mn_list)

    # for m,n in mn_list:
        # test_zero(pathObj,m,n)







