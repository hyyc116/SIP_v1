#coding:utf-8
import sys
sys.path.extend(['..','.'])
from basic_config import *
from paths import PATH

'''
    本文件完成数据集的抽取
    
    basic-structure-author

'''


## 首先抽取特征,根据数据集构建训练集，测试集
def construct_RNN_datasets(pathObj,m,n,scale=True,feature_set = 'basic'):

    testing_ids = set(pathObj.read_file(pathObj._testing_pid_path))
    validing_ids = set(pathObj.read_file(pathObj._validing_pid_path))
    pid_features = pathObj.loads_json(pathObj.dataset_feature_path(m,n))

    train_X = []
    train_Y = []

    test_X = []
    test_Y = []

    valid_X = []
    valid_Y = []

    test_sorted_ids = []

    for pid in pid_features.keys():

        feature = pid_features[pid]

        X = []
        
        Y = [float(y) for y in feature['Y']]

        X.append([float(f) for f in feature['hist_cits']])

        ## 背景
        X.append([float(f) for f in feature['b-num']])

        if 'author' in feature_set: 
            ## 作者hindex
            X.append([float(f) for f in feature['a-first-hix']])
            X.append([float(f) for f in feature['a-avg-hix']])

            ## 作者文章数量
            X.append([float(f) for f in feature['a-first-pnum']])
            X.append([float(f) for f in feature['a-avg-pnum']])
            
            ## 机构影响力 
            X.append([float(f) for f in feature['i-avg-if']])
            ## 期刊影响力
            X.append([float(f) for f in feature['v-if']])
            
            ## 作者数量,静态特征也用动态表示，每年不变
            X.append([float(feature['a-num'])]*m)
            X.append([float(feature['a-career-length'])]*m)

        elif 'structure' in feature_set:

            X.append([float(f) for f in feature['disrupt']])
            X.append([float(f) for f in feature['depth']])
            X.append([float(f) for f in feature['dependence']])
            X.append([float(f) for f in feature['anlec']])



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

    ## 需要将X中的维度进行翻转
    ## 目前是 (num,10,m) 需要转换为 (num,m,10)
    train_X = np.array(train_X).swapaxes(1,2)
    test_X = np.array(test_X).swapaxes(1,2)
    valid_X = np.array(valid_X).swapaxes(1,2)

    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)
    valid_Y = np.array(valid_Y)

    train_X,test_X,valid_X,dx_mean,dx_std = scale_dataset(train_X,test_X,valid_X,scale)
    train_Y,test_Y,valid_Y,y_mean,y_std = scale_dataset(train_Y,test_Y,valid_Y,scale)

    return train_X,test_X,valid_X,dx_mean,dx_std,\
            train_Y,test_Y,valid_Y,y_mean,y_std,\
            test_sorted_ids


## 归一化数据
def scale_dataset(train,test,valid,scale=True):

    if scale:
        mean = np.mean(train,axis=0)
        std = np.std(train,axis=0)
    else:
        mean = 0
        std = 1
    return (train-mean)/std,(test-mean)/std,(valid-mean)/std,mean,std

def unscale_dataset(data,mean,std):
    return data*std+mean


if __name__ == '__main__':
    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    construct_RNN_datasets(pathObj,3,10)


