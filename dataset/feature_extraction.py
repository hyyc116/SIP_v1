#coding:utf-8
import sys
sys.path.extend(['..','.'])
from basic_config import *
from paths import PATH
'''
    抽取所有特征，并保存到dict中

'''

def extract_hindex_features(history_years,seq_authors,author_year_hindex):

    if len(seq_authors)==0:
        return None,None
    ## 所有作者的平均hindex
    all_hindex = []
    ## average hindex
    for i,seq in enumerate(sorted(seq_authors.keys())):

        author = seq_authors[seq]

        year_hindex = author_year_hindex[author]

        author_hindex= []
        for year in history_years:
            author_hindex.append(year_hindex[year])

        ## 第一个就是第一作者，防止出现数据库中seq缺失的问题
        if i==0:
            first_hindex = author_hindex

        all_hindex.append(author_hindex)

    ## 求平均值
    avg_hindex = [np.mean(hixs) for hixs in zip(*all_hindex)]

    return first_hindex,avg_hindex

def extract_citations(years,year_citnum):

    citations = []
    for year in years:
        citations.append(year_citnum.get(str(year),0))

    return citations

def extract_author_pnum(history_years,seq_authors,author_year_pnum):

    if len(seq_authors)==0:
        return None,None

    ## 遍历所有作者在history_years内每年发表论文的数量
    all_pnums = []
    for i,seq in enumerate(seq_authors.keys()):

        author = seq_authors[seq]

        year_pnum = author_year_pnum[author]

        pnums = []

        for year in history_years:

            pnums.append(year_pnum.get(year,0))

        if i==0:

            first_author_pnum = pnums

        all_pnums.append(pnums)

    avg_pnums = [np.mean(pnums) for pnums in zip(*all_pnums)]

    return first_author_pnum,avg_pnums


def avg_author_career_length(year,seq_authors,author_starty):

    if len(seq_authors)==0:
        return None

    delta_years = []

    for author in seq_authors.values():

        delta_years.append(int(year) - int(author_starty[author]))

    return np.mean(delta_years)

def avg_ins_if(history_years,inses,ins_year_if):

    all_ifs = []

    for ins in inses:

        year_if = ins_year_if[ins]

        ifs = []
        for year in history_years:
            ifs.append(year_if.get(year,0))

        all_ifs.append(ifs)

    return [np.mean(ifs) for ifs in zip(*all_ifs)]

def cal_venue_if(history_years,year_if):

    ifs = []
    for year in history_years:
        ifs.append(year_if.get(year,0))

    return ifs


def extract_structural(history_years,year_attr):

    attrs = []


    for year in history_years:

        attrs.append(year_attr.get(year,0))

    return attrs





## 抽取m年作为输入，n年的citationlist作为Y
def extract_features(pathObj,mnlist):

    logging.info("loading data ...")
    pid_seq_authors = defaultdict(dict)
    pid_affs = defaultdict(list)

    

    pid_year = json.loads(open(pathObj._field_paper_year_path).read())

    for line in open(pathObj._paper_author_aff_path):
        paper_id,author_id,author_name,aff_id,aff_name,author_sequence_number,year = line.strip().split(',')

        if paper_id =='paper_id':
            continue

        if pid_year[paper_id]<1970:
            continue

        pid_seq_authors[paper_id][int(author_sequence_number)] = author_id
        pid_affs[paper_id].append(aff_id)

    ## 加载引用次数字典
    pid_year_citnum = json.loads(open(pathObj._paper_year_citations_path).read())

    ## 加载作者hindex
    author_year_hindex = json.loads(open(pathObj._author_year_hix_path).read())

    ## 加载作者文章数量
    author_year_pnum = json.loads(open(pathObj._author_year_papernum_path).read())

    ## 作者研究开始的年份
    author_starty = json.loads(open(pathObj._author_start_time_path).read())

    ## 机构随着年份的impact factor
    ins_year_if = json.loads(open(pathObj._ins_year_if_path).read())

    ## 文章对应的机构列表
    pid_inses = json.loads(open(pathObj._paper_ins_path).read())

    ## 会议随着年份的impact factor
    venue_year_if = json.loads(open(pathObj._venue_year_if_path).read())

    ## 文章对应的venue id
    pid_vid = json.loads(open(pathObj._paper_venueid_path).read())

    ## 每年论文总数量变化
    year_pnum_t = json.loads(open(pathObj._field_paper_num_dis_path).read())

    ## 论文的结构信息
    pid_year_disruptives = json.loads(open(pathObj._paper_year_disruptive_path).read())
    pid_year_depths = json.loads(open(pathObj._paper_year_depth_path).read())
    pid_year_dependence = json.loads(open(pathObj._paper_year_dependence_path).read())
    pid_year_anlec = json.loads(open(pathObj._paper_year_anlec_path).read())

    pidset_with_vid = set(pid_vid.keys())
    pidset_with_aff = set(pid_seq_authors.keys())

    for m,n in mnlist:
        ## paper ids in datasets
        dataset_ids = list(set([line.strip() for line in open(pathObj.dataset_id_path(m,n))]))

        datawith_author_venue = pidset_with_aff&pidset_with_vid&set(dataset_ids)

        logging.info('{} papers in datasets reserved loaded,{} with venue and aff.'.format(len(dataset_ids),len(datawith_author_venue)))

        ## 论文ID对应特征值
        pid_features = {}

        ## 每一篇论文抽取特征
        for progress,pid in enumerate(dataset_ids):

            if progress%100000==0:
                logging.info('progress {}/{},{} pid features extracted ...'.format(progress,len(dataset_ids),len(pid_features)))

            if pid_year.get(pid,None) is None:
                continue

            year = int(pid_year[pid])

            history_years = [str(year+d) for d in range(m)]

            predict_years = [str(year+m+d) for d in range(n)]


            ## 所有数据中必有的特征
            s_features = {}

            ## 1. 引用次数
            his_citations = extract_citations(history_years,pid_year_citnum[pid])

            s_features['hist_cits'] = his_citations

            ## 0. 预测的引用次数
            predict_citations = extract_citations(predict_years,pid_year_citnum[pid])

            s_features['Y'] = predict_citations

            ## 结构特征
            # print(pid_year_disruptives[pid],history_years)
            discruptive_scores = extract_structural(history_years,pid_year_disruptives[pid])

            s_features['disrupt'] = discruptive_scores

            depth_scores = extract_structural(history_years,pid_year_depths[pid])

            s_features['depth'] = depth_scores

            dependence_scores = extract_structural(history_years,pid_year_dependence[pid])

            s_features['dependence'] = dependence_scores

            anlec_score = extract_structural(history_years,pid_year_anlec[pid])

            s_features['anlec'] = anlec_score

            ## 每年的论文数量
            bak_t_pnums = [year_pnum_t[year] for year in history_years]
            s_features['b-num'] = bak_t_pnums

            # 其他特征,可能为空，如果不存在我们不进行设置值
            ## 作者
            seq_authors = pid_seq_authors[pid]
            ## ins的列表
            inses = pid_inses.get(pid,None)

            ## venue id
            vid = pid_vid.get(pid,None)

            ## h index 相关特征
            his_first_hix,his_avg_hix =  extract_hindex_features(history_years,seq_authors,author_year_hindex)
            if his_first_hix is not None and his_avg_hix is not None:
                s_features['a-first-hix'] = his_first_hix
                s_features['a-avg-hix'] = his_avg_hix

            ## 文章数量相关特征
            his_first_pnum,his_avg_pnum = extract_author_pnum(history_years,seq_authors,author_year_pnum)
            if his_first_pnum is not None and his_avg_pnum is not None:
                s_features['a-first-pnum'] = his_first_pnum
                s_features['a-avg-pnum'] = his_avg_pnum

            ## 作者数量
            au_num = len(seq_authors)
            if au_num!=0:
                s_features['a-num'] = au_num

            ## 作者的平均研究年龄
            avg_career_length = avg_author_career_length(year,seq_authors,author_starty)

            if avg_career_length is not None:
                s_features['a-career-length'] = avg_career_length

            ## 机构的平均impact facor
            if inses is not None:
                ins_avg_if = avg_ins_if(history_years,inses,ins_year_if)

                s_features['i-avg-if'] = ins_avg_if

            ## venue的if
            if vid is not None:
                venue_if = cal_venue_if(history_years,venue_year_if[vid])

                s_features['v-if'] = venue_if

            pid_features[pid] = s_features

        ##保存特征json文件
        open(pathObj.dataset_feature_path(m,n),'w').write(json.dumps(pid_features))
        logging.info('{} dataset features saved to {}.'.format(len(pid_features),pathObj.dataset_feature_path(m,n)))


### 每一个数据集的文章ID至少比2016年早m+n年发表
def construct_datasets(pathObj,mn_list):

    logging.info('Loading data ...')
    reserved_pids = [line.strip() for line in open(pathObj._reserved_papers_path)]

    logging.info('{} selected paper ids loaded, start to load paper year dict ...'.format(len(reserved_pids)))

    paper_year = json.loads(open(pathObj._field_paper_year_path).read())

    logging.info('paper year dict loaded.')

    for m,n in mn_list:

        mn_pids = [pid for pid in reserved_pids if (2018-int(paper_year[pid]))>(m+n) and int(paper_year[pid])>=1970]

        open(pathObj.dataset_id_path(m,n),'w').write('\n'.join(list(set(mn_pids))))

        logging.info('{} papers in dataset sip-m{}n{}.'.format(len(mn_pids),m,n))

    logging.info('Done')


if __name__ == '__main__':

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    mn_list=[(3,1),(3,3),(3,5),(3,10),(5,1),(5,3),(5,5),(5,10),(10,1),(10,3),(10,5),(10,10)]

    construct_datasets(pathObj,mn_list)
    extract_features(pathObj,mn_list)










