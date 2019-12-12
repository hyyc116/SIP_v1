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

from dataset.datasets_construction import construct_shallow_datasets as construct_datasets

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


def train_and_evaluate(pathObj,mn_list,scale=False,feature_set='basic'):

    ## m n list
    lines = ['dataset,model,r2,mae,mse']
    shallow_result = defaultdict(dict)
    for m,n in mn_list:
        dataset = 'sip-m{}n{}'.format(m,n)
        logging.info('train dataset sip-m{}n{} ..'.format(m,n))
        train_X,train_Y,test_X,test_Y,valid_X,valid_Y,test_sorted_ids,_,_,_ = construct_datasets(pathObj,m,n,scale=scale,feature_set=feature_set)
        
        shallow_result[dataset]['IDS'] = test_sorted_ids

        # print(train_X[:2])
        # print(train_Y[:2])

        model_name = 'RFR-{}'.format(feature_set)

        models = train_RFR(train_X,train_Y)
        r2,mae,mse,predict_Y =evaluate_model(models,test_X,test_Y)

        shallow_result[dataset][model_name]=predict_Y

        print('{}====R^2:{},MAE:{},MSE:{}'.format(model_name,r2,mae,mse))

        lines.append('sip-m{}n{},{},{},{},{}'.format(m,n,model_name,r2,mae,mse))

        models = train_Linear(train_X,train_Y)
        model_name = 'LR-{}'.format(feature_set)
        r2,mae,mse,predict_Y =evaluate_model(models,test_X,test_Y)
        shallow_result[dataset][model_name]=predict_Y

        lines.append('sip-m{}n{},{},{},{},{}'.format(m,n,model_name,r2,mae,mse))

        print('{}====R^2:{},MAE:{},MSE:{}'.format(model_name,r2,mae,mse))

    open(pathObj._shallow_result_summary,'a').write('\n'.join(lines))

    logging.info('result summary saved.')

    open(pathObj._shallow_testing_prediction_result,'a').write(json.dumps(shallow_result))

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

    mn_list=[(10,1),(10,3),(10,5),(10,10),(5,1),(5,3),(5,5),(5,10),(3,1),(3,3),(3,5),(3,10)]

    train_and_evaluate(pathObj,mn_list)

    train_and_evaluate(pathObj,mn_list,feature_set='basic-author')

    train_and_evaluate(pathObj,mn_list,feature_set='basic-structure')

    train_and_evaluate(pathObj,mn_list,feature_set='basic-author-structure')



    # for m,n in mn_list:
        # test_zero(pathObj,m,n)







