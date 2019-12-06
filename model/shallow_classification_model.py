#coding:utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import sys
sys.path.extend(['..','.'])
from paths import PATH
from basic_config import *

from dataset.datasets_construction import construct_shallow_datasets as construct_datasets


def train_RFC(X,y):

    clf = RandomForestClassifier(n_estimators=100, max_depth=3,
                              random_state=0,n_jobs=8)
    return clf.fit(X,y)

##评测模型
def evaluate_model(model,test_X,test_Y):

    predict_Y = model.predict(test_X)

    return accuracy_score(test_Y,predict_Y),predict_Y


def train_and_evaluate(pathObj,mn_list,scale=False):

    ## m n list
    lines = ['dataset,model,r2,mae,mse']
    shallow_result = defaultdict(dict)
    for m,n in mn_list:
        dataset = 'sip-m{}n{}'.format(m,n)
        logging.info('train dataset sip-m{}n{} ..'.format(m,n))
        train_X,train_Y,test_X,test_Y,valid_X,valid_Y,test_sorted_ids,train_L,test_L,valid_L = construct_datasets(pathObj,m,n,scale=scale)
        
        # shallow_result[dataset]['IDS'] = list(test_sorted_ids)

        # print(train_X[:2])
        # print(train_Y[:2])

        model = train_RFC(train_X,train_L)
        ACC,predict_Y =evaluate_model(model,test_X,test_L)

        # shallow_result[dataset]['RFC']=predict_Y.toList()

        print('RFC====ACC:{}'.format(ACC))

        lines.append('sip-m{}n{},{},{}'.format(m,n,'RFC',ACC))


    open(pathObj._shallow_result_clc_summary,'w').write('\n'.join(lines))

    logging.info('result summary saved.')

    # open(pathObj._shallow_testing_prediction_result_clc,'w').write(json.dumps(shallow_result))

    # logging.info('result saved.')


if __name__ == '__main__':
    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    mn_list=[(3,1),(3,3),(3,5),(3,10)]

    train_and_evaluate(pathObj,mn_list)