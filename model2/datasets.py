#coding:utf-8
import sys
sys.path.extend(['..','.'])
from basic_config import *
from paths import PATH

'''
    本文件完成数据集的抽取
    
    basic-structure-author

'''

### feature-set:basic-stucture-author
def num_tolabel(num):

    if num>=341: ## 0.01
        return 0
    elif num >=106: ## 0.05
        return 1
    elif num >=56: ## 0.1
        return 2
    elif num >=26: ## 0.2
        return 3
    elif num >=9: ## 0.4
        return 4
    else:
        return 5

## 首先抽取特征,根据数据集构建训练集，测试集
def construct_RNN_datasets(pathObj,m,n,scale=True,feature_set = 'basic',return_label=False,seperate_static=False,only_all=False):

    testing_ids = set(pathObj.read_file(pathObj._testing_pid_path))
    validing_ids = set(pathObj.read_file(pathObj._validing_pid_path))
    pid_features = pathObj.loads_json(pathObj.dataset_feature_path(m,n))

    train_X = []
    train_Y = []
    train_L = []
    train_SX = []


    test_X = []
    test_Y = []
    test_L = []
    test_SX = []

    valid_X = []
    valid_Y = []
    valid_L = []
    valid_SX = []

    test_sorted_ids = []

    for pid in pid_features.keys():

        feature = pid_features[pid]

        X = []
        SX = []

        Y = [float(y) for y in feature['Y']]

        m = len(feature['hist_cits'])

        L = num_tolabel(np.sum(Y)+np.sum(feature['hist_cits']))

        X.append([float(f) for f in feature['hist_cits']])

        ## 背景
        X.append([float(f) for f in feature['b-num']])

        if 'author' in feature_set: 

            if only_all:
                ## 作者hindex, 只保留含有这些特征的样本
                if feature.get('a-first-hix',None) is None:
                    continue

                if feature.get('i-avg-if', None) is None:
                    continue

                if feature.get('v-if',None) is None:
                    continue


            X.append([float(f) for f in feature.get('a-first-hix',[0]*m)])
            X.append([float(f) for f in feature.get('a-avg-hix',[0]*m)])

            ## 作者文章数量
            X.append([float(f) for f in feature.get('a-first-pnum',[0]*m)])
            X.append([float(f) for f in feature.get('a-avg-pnum',[0]*m)])
            
            ## 机构影响力 
            X.append([float(f) for f in feature.get('i-avg-if',[0]*m)])
            ## 期刊影响力
            X.append([float(f) for f in feature.get('v-if',[0]*m)])
            
            ## 作者数量,静态特征也用动态表示，每年不变
            if not seperate_static:
                X.append([float(feature.get('a-num',0))]*m)
                X.append([float(feature.get('a-career-length',0))]*m)
            else:
                SX.append(feature.get('a-num',0))
                SX.append(feature.get('a-career-length',0))

        elif 'structure' in feature_set:

            X.append([float(f) for f in feature['disrupt']])
            X.append([float(f) for f in feature['depth']])
            X.append([float(f) for f in feature['dependence']])
            X.append([float(f) for f in feature['anlec']])

        if pid in testing_ids:
            test_sorted_ids.append(pid)
            test_X.append(X)
            test_Y.append(Y)
            test_L.append(L)
            test_SX.append(SX)


        elif pid in validing_ids:
            valid_X.append(X)
            valid_Y.append(Y)
            valid_L.append(L)
            valid_SX.append(SX)

        else:
            train_X.append(X)
            train_Y.append(Y)
            train_L.append(L)
            train_SX.append(SX)


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
    train_SX,test_SX,valid_SX,sx_mean,sx_std = scale_dataset(train_SX,test_SX,valid_SX,scale)

    if return_label:

        if seperate_static:
            return train_X,test_X,valid_X,dx_mean,dx_std,\
                train_SX,test_SX,valid_SX,sx_mean,sx_std,\
                train_Y,test_Y,valid_Y,y_mean,y_std,\
                train_L,test_L,valid_L,\
                test_sorted_ids

        return train_X,test_X,valid_X,dx_mean,dx_std,\
            train_Y,test_Y,valid_Y,y_mean,y_std,\
            train_L,test_L,valid_L,\
            test_sorted_ids

    if seperate_static:

        return train_X,test_X,valid_X,dx_mean,dx_std,\
            train_SX,test_SX,valid_SX,sx_mean,sx_std,\
            train_Y,test_Y,valid_Y,y_mean,y_std,\
            test_sorted_ids

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


