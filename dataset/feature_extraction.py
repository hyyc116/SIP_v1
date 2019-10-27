#coding:utf-8
import sys
sys.path.extend(['..','.'])
from basic_config import *
from paths import PATH
'''
    抽取所有特征，并保存到dict中

'''

def extract_hindex_features(pid,history_years,pid_seq_authors,author_year_hindex):

    ## 发表年份
    published_year = pid_year[pid]

    ## 第一作者的hindex变化
    first_a_hindex_year_hindex = author_year_hindex[pid_seq_authors[pid][1]]

    for year in history_years:

        pass




    ## average hindex
    for seq,author in pid_seq_authors[pid]:

        pass


def extract_citations(years,year_citnum):

    citations = []
    for year in years:
        citations.append(year_citnum.get(str(year),0))

    return citations



## 抽取m年作为输入，n年的citationlist作为Y
def extract_features(pathObj,m,n):

    pid_seq_authors = defaultdict(dict)
    pid_affs = defaultdict(list)
    author_year_papers = defaultdict(list)
    pid_year = {}
    for line in open(pathObj._paper_author_aff_path):
        paper_id,author_id,author_name,aff_id,aff_name,author_sequence_number,year = line.strip().split(',')

        if paper_id =='paper_id':
            continue

        pid_seq_authors[paper_id][int(author_sequence_number)] = author_id

        pid_affs[paper_id].append(aff_id)

        author_year_papers[author_id][int(year)].append(paper_id)

        pid_year[paper_id] = int(year)

    ## 加载作者hindex
    author_year_hindex = json.loads(pathObj._author_year_hix_path)

    ## paper ids in datasets
    dataset_ids = [line.strip() for line in open(pathObj.dataset_id_path(m,n))]
    logging.info('{} papers in datasets reserved loaded.'.format(len(resrved_ids)))

    ## 加载引用次数字典
    pid_year_citnum = json.loads(open(pathObj._paper_year_citations_path).read())

    ## 每一篇论文抽取特征
    for pid in dataset_ids:

        year = int(pid_year[pid])

        history_years = [year+d for d in range(m)]

        predict_years = [year+m+d for d in range(n)]

        ## 引用次数
        his_citations = extract_citations(history_years,pid_year_citnum[pid])

        ## 预测的引用次数
        predict_citations = extract_citations(predict_years,pid_year_citnum[pid])

        ## h index 相关特征
        extract_hindex_features(pid,history_years,pid_seq_authors,author_year_hindex)





    ## 作者特征

    pass

if __name__ == '__main__':
    year = 2000
    m=5
    n=10

    history_years = [year+d for d in range(m)]

    predict_years = [year+m+d for d in range(n)]

    print(history_years)
    print(predict_years)








