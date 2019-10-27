#coding:utf-8
import sys
sys.path.extend(['..','.'])
from basic_config import *
from paths import PATH
'''
    抽取所有特征，并保存到dict中

'''

def extract_hindex_features(pid,pid_seq_authors,author_year_hindex):

    ## 发表年份
    published_year = pid_year[pid]

    ## 第一作者的hindex变化
    first_a_hindex_year_hindex = author_year_hindex[pid_seq_authors[pid][1]]

    for year in sorted(first_a_hindex_year_hindex.keys(),key = lambda x:int(x)):

        if int(year)<published_year:

            continue



    ## average hindex
    for seq,author in pid_seq_authors[pid]:

        pass








def extract_features(pathObj):

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
    reserved_ids = [line.strip() for line in open(pathObj._reserved_papers_path)]
    logging.info('{} papers in datasets reserved loaded.'.format(len(resrved_ids)))

    ## 每一篇论文抽取特征
    for pid in reserved_ids:

        ## h index 相关特征
        extract_hindex_features(pid,pid_year,pid_seq_authors,author_year_hindex)





    ## 作者特征

    pass






