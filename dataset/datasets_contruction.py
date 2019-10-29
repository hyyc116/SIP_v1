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

    print(pids[:10])

    ## 选择11000个作为测试验证机和
    selected_pids =  np.random.choice(pids,11000,replace=False)

    test_pids = np.random.choice(selected_pids,10000,replace=False)

    valid_pids = list(set(selected_pids)-set(test_pids))

    logging.info('{} test_pids selected, {} valid pids selected.'.format(len(test_pids),len(valid_pids)))

    open(pathObj._testing_pid_path,'w').write('\n'.join(test_pids))
    open(pathObj._validing_pid_path,'w').write('\n'.join(valid_pids))


if __name__ == '__main__':
    
    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    mn_list=[(3,1),(3,3),(3,5),(3,10),(5,1),(5,3),(5,5),(5,10)]

    # construct_datasets(pathObj,mn_list)

    get_test_valid_set_ids(pathObj)







