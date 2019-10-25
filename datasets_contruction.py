#coding:utf-8
from basic_config import *
from paths import PATH

'''
    本文件完成数据集的抽取，
    分别构建sip-m5n1,sip-m5n5,sip-m5n10, from short-term prediction to long-term prediction.

'''


### 每一个数据集的文章ID至少比2016年早m+n年发表
def construct_datasets(pathObj.mn_list):

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


if __name__ == '__main__':
    
    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    mn_list=[(3,1),(3,3),(3,5),(3,10),(5,1),(5,3),(5,5),(5,10)]

    construct_datasets(pathObj,mn_list)







