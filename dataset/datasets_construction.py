#coding:utf-8
import sys
sys.path.extend(['..','.'])
from basic_config import *
from paths import PATH

'''
    本文件完成数据集的抽取，
    分别构建sip-m5n1,sip-m5n5,sip-m5n10, from short-term prediction to long-term prediction.

'''

### 每一个数据集的文章ID至少比2016年早m+n年发表
def construct_datasets(pathObj,mn_list):

    logging.info('Loading data ...')
    reserved_pids = [line.strip() for line in open(pathObj._reserved_papers_path)]

    logging.info('{} selected paper ids loaded, start to load paper year dict ...'.format(len(reserved_pids)))

    paper_year = json.loads(open(pathObj._field_paper_year_path).read())

    logging.info('paper year dict loaded.')

    for m,n in mn_list:

        mn_pids = [pid for pid in reserved_pids if (2016-int(paper_year[pid]))>=(m+n)]

        open(pathObj.dataset_id_path(m,n),'w').write('\n'.join(mn_pids))

        logging.info('{} papers in dataset sip-m{}n{}.'.format(len(mn_pids),m,n))

    logging.info('Done')


## 统一所有数据集的测试集以及验证集，从sip-m5n10中随机抽取1000篇作为验证集，10000篇作为测试集。
def get_test_valid_set_ids(pathObj):
    pid_features = json.loads(open(pathObj.dataset_feature_path(5,10)).read())

    pids = list(pid_features.keys())

    print(len(pids)*0.6)

    test_percent = 0.4

    test_vs_valid = 3

    test_num = int(len(pids)*test_percent)
    valid_num = int(test_num/3)

    ## 选择11000个作为测试验证机和
    selected_pids =  np.random.choice(pids,test_num,replace=False)

    valid_pids = np.random.choice(selected_pids,valid_num,replace=False)

    test_pids = list(set(selected_pids)-set(valid_pids))

    logging.info('{} test_pids selected, {} valid pids selected.'.format(len(test_pids),len(valid_pids)))

    open(pathObj._testing_pid_path,'w').write('\n'.join(test_pids))
    open(pathObj._validing_pid_path,'w').write('\n'.join(valid_pids))


## 首先抽取特征,根据数据集构建训练集，测试集
def construct_RNN_datasets(pathObj,m,n,scale=True):

    testing_ids = set(pathObj.read_file(pathObj._testing_pid_path))
    validing_ids = set(pathObj.read_file(pathObj._validing_pid_path))

    pid_features = pathObj.loads_json(pathObj.dataset_feature_path(m,n))

    ## 抽取特征
    train_dynamic_X = []
    train_static_X = []
    train_Y = []

    test_dynamic_X = []
    test_static_X = []
    test_Y = []

    valid_dynamic_X = []
    valid_static_X = []
    valid_Y = []

    test_sorted_ids = []

    for pid in pid_features.keys():

        ## 将所有的特征串联起来
        feature = pid_features[pid]

        dynamic_X = []
        static_X = []
        Y=[float(y) for y in feature['Y']]

        dynamic_X.append([float(f) for f in feature['hist_cits']])
        ## 作者hindex
        dynamic_X.append([float(f) for f in feature['a-first-hix']])
        dynamic_X.append([float(f) for f in feature['a-avg-hix']])
        ## 作者文章数量
        dynamic_X.append([float(f) for f in feature['a-first-pnum']])
        dynamic_X.append([float(f) for f in feature['a-avg-pnum']])
        ## 作者数量
        static_X.append(float(feature['a-num']))
        static_X.append(float(feature['a-career-length']))
        ## 机构影响力 
        dynamic_X.append([float(f) for f in feature['i-avg-if']])
        ## 期刊影响力
        dynamic_X.append([float(f) for f in feature['v-if']])
        ## 背景
        dynamic_X.append([float(f) for f in feature['b-num']])

        if pid in testing_ids:
            test_sorted_ids.append(pid)
            test_dynamic_X.append(list(zip(*dynamic_X)))
            test_static_X.append(static_X)
            test_Y.append(Y)
        elif pid in validing_ids:
            valid_dynamic_X.append(list(zip(*dynamic_X)))
            valid_static_X.append(static_X)
            valid_Y.append(Y)
        else:
            train_dynamic_X.append(list(zip(*dynamic_X)))
            train_static_X.append(static_X)
            train_Y.append(Y)


    logging.info('{} of training dataset, {} of testing dataset, {} of valid dataset.'.format(len(train_Y),len(test_Y),len(valid_Y)))

    # print(train_Y[:64])
    train_dynamic_X,test_dynamic_X,valid_dynamic_X,dx_mean,dx_std = scale_dataset(train_dynamic_X,test_dynamic_X,valid_dynamic_X,True)

    train_static_X,test_static_X,valid_static_X,sx_mean,sx_std = scale_dataset(train_static_X,test_static_X,valid_static_X,True)

    train_Y,test_Y,valid_Y,y_mean,y_std = scale_dataset(train_Y,test_Y,valid_Y,scale)

    logging.info('scale done')

    return train_dynamic_X,train_static_X,train_Y,\
            test_dynamic_X,test_static_X,test_Y,\
            valid_dynamic_X,valid_static_X,valid_Y,\
            test_sorted_ids,dx_mean,dx_std,\
            sx_mean,sx_std,y_mean,y_std



def construct_RNN_cat_datasets(pathObj,m,n,scale=False):

    testing_ids = set(pathObj.read_file(pathObj._testing_pid_path))
    validing_ids = set(pathObj.read_file(pathObj._validing_pid_path))

    pid_features = pathObj.loads_json(pathObj.dataset_feature_path(m,n))

    ## 抽取特征
    train_dynamic_X = []
    train_static_X = []
    train_Y = []
    train_ins_Y = []
    train_last_Y = []

    test_dynamic_X = []
    test_static_X = []
    test_Y = []
    test_ins_Y = []
    test_last_Y = []

    valid_dynamic_X = []
    valid_static_X = []
    valid_Y = []
    valid_ins_Y = []
    valid_last_Y = []

    y_id = {}
    id_y = {}

    test_sorted_ids = []

    y_dis = defaultdict(int)

    for pid in pid_features.keys():

        ## 将所有的特征串联起来
        feature = pid_features[pid]

        dynamic_X = []
        static_X = []
        last_y = int(feature['hist_cits'][-1])
        Y=[int(y) for y in feature['Y']]
        ins_Y = []
        _last_y = last_y
        for y in feature['Y']:
            increase_Y = y-last_y

            if increase_Y<-100:
                increase_Y = -100

            if increase_Y>200:
                increase_Y = 200

            y_dis[increase_Y]+=1

            _id = y_id.get(increase_Y,len(y_id))

            y_id[increase_Y] =  _id
            id_y[_id] = increase_Y


            ins_Y.append(_id)
            # _last_y = y


        dynamic_X.append([float(f) for f in feature['hist_cits']])
        ## 作者hindex
        dynamic_X.append([float(f) for f in feature['a-first-hix']])
        dynamic_X.append([float(f) for f in feature['a-avg-hix']])
        ## 作者文章数量
        dynamic_X.append([float(f) for f in feature['a-first-pnum']])
        dynamic_X.append([float(f) for f in feature['a-avg-pnum']])
        ## 作者数量
        static_X.append(float(feature['a-num']))
        static_X.append(float(feature['a-career-length']))
        ## 机构影响力 
        dynamic_X.append([float(f) for f in feature['i-avg-if']])
        ## 期刊影响力
        dynamic_X.append([float(f) for f in feature['v-if']])
        ## 背景
        dynamic_X.append([float(f) for f in feature['b-num']])

        if pid in testing_ids:
            test_sorted_ids.append(pid)
            test_dynamic_X.append(list(zip(*dynamic_X)))
            test_static_X.append(static_X)
            test_Y.append(Y)
            test_ins_Y.append(ins_Y)
            test_last_Y.append(last_y)
        elif pid in validing_ids:
            valid_dynamic_X.append(list(zip(*dynamic_X)))
            valid_static_X.append(static_X)
            valid_Y.append(Y)
            valid_last_Y.append(last_y)
            valid_ins_Y.append(ins_Y)

        else:
            train_dynamic_X.append(list(zip(*dynamic_X)))
            train_static_X.append(static_X)
            train_Y.append(Y)
            train_ins_Y.append(ins_Y)
            train_last_Y.append(last_y)


    logging.info('{} of training dataset, {} of testing dataset, {} of valid dataset.'.format(len(train_Y),len(test_Y),len(valid_Y)))

    # print(train_Y[:64])
    train_dynamic_X,test_dynamic_X,valid_dynamic_X,dx_mean,dx_std = scale_dataset(train_dynamic_X,test_dynamic_X,valid_dynamic_X,True)

    train_static_X,test_static_X,valid_static_X,sx_mean,sx_std = scale_dataset(train_static_X,test_static_X,valid_static_X,True)

    train_Y,test_Y,valid_Y,y_mean,y_std = scale_dataset(train_Y,test_Y,valid_Y,scale)

    logging.info('scale done')


    ## y_dis的分布
    plot_ydis(y_dis)


    return train_dynamic_X,train_static_X,train_Y,\
            test_dynamic_X,test_static_X,test_Y,\
            valid_dynamic_X,valid_static_X,valid_Y,\
            test_sorted_ids,dx_mean,dx_std,\
            sx_mean,sx_std,y_mean,y_std,\
            train_ins_Y,test_ins_Y,valid_ins_Y,\
            y_id,id_y,train_last_Y,test_last_Y,valid_last_Y


def plot_ydis(y_dis):


    xs = []
    ys = []
    for x in sorted(y_dis.keys()):
        xs.append(x)
        ys.append(y_dis[x])

    plt.figure(figsize=(5,4))

    plt.plot(xs,np.array(ys)/float(np.sum(ys)))


    plt.xlabel('y increase')
    plt.ylabel('percentage')
    plt.yscale('log')

    plt.tight_layout()

    plt.savefig('data/y_dis.png',dpi=300)

## 首先抽取特征,根据数据集构建训练集，测试集
def construct_shallow_datasets(pathObj,m,n,scale=True):

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

## 将数据scale scale的方法是>10的数字取log
def scale_Y(y):
    y = int(y)
    if y<=100:
        return y
    else:
        # print('y:',y)
        # print('scale y',int(y+np.log(100)-1))
        return int(100+np.log(y)/np.log(4)-1)





def unscale_dataset(data,mean,std):

    return data*std+mean


if __name__ == '__main__':
    
    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    mn_list=[(3,1),(3,3),(3,5),(3,10),(5,1),(5,3),(5,5),(5,10)]

    # construct_datasets(pathObj,mn_list)

    get_test_valid_set_ids(pathObj)







