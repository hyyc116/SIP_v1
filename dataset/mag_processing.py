#coding:utf-8
'''
对MAG的数据进行数据预处理

MAG的数据包括作者名消歧，本程序从所有的数据里面抽取某一个领域的论文数据，创建训练数据。

SELECT * FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog'
AND schemaname != 'information_schema';

 mag_core   | affiliations                         |
 mag_core   | journals                             |
 mag_core   | conference_series                    |
 mag_core   | conference_instances                 |
 mag_core   | papers                               |
 mag_core   | paper_resources                      |
 mag_core   | fields_of_study                      |
 mag_core   | related_field_of_study               |
 mag_core   | paper_urls                           |
 mag_core   | paper_abstract_inverted_index        |
 mag_core   | paper_author_affiliations            |
 mag_core   | authors                              |
 mag_core   | paper_citation_contexts              |
 mag_core   | paper_fields_of_study                |
 mag_core   | paper_languages                      |
 mag_core   | paper_recommendations                |
 mag_core   | paper_references                     |
 mag_core   | fields_of_study_children             |

------------------------------------
field,id,level=0
 (Art,142362112)
 (Biology,86803240)
 (Business,144133560)
 (Chemistry,185592680)
 ("Computer science",41008148)
 (Economics,162324750)
 (Engineering,127413603)
 ("Environmental science",39432304)
 (Geography,205649164)
 (Geology,127313418)
 (History,95457728)
 ("Materials science",192562407)
 (Mathematics,33923547)
 (Medicine,71924100)
 (Philosophy,138885662)
 (Physics,121332964)
 ("Political science",17744445)
 (Psychology,15744967)
 (Sociology,144024400)


'''
import sys
sys.path.extend(['..','.'])
from basic_config import *
from paths import PATH
## 读出计算机领域的所有论文ID
def read_paper_ids(pathObj):

    field = pathObj._field_name

    query_op = dbop()

    sql = "select mag_core.paper_fields_of_study.paper_id from mag_core.paper_fields_of_study, mag_core.fields_of_study where mag_core.paper_fields_of_study.field_of_study_id = mag_core.fields_of_study.field_of_study_id and mag_core.fields_of_study.field_of_study_id='41008148' "
    progress = 0

    paper_ids = []
    for paper_id in query_op.query_database(sql):

        progress+=1

        if progress%10000000==0:
            logging.info('read paper ids of {}, progress {} ...'.format(field,progress))

        paper_ids.append(paper_id[0])

    logging.info('Filed {} has {} papers.'.format(field,len(paper_ids)))

    open(pathObj._field_paper_ids_path,'w').write('\n'.join(paper_ids))

    logging.info('Paper ids of field {} saved to {}'.format(field,pathObj._field_paper_ids_path))

    paper_ids_set = set(paper_ids)
    sql = 'select paper_id,year from mag_core.papers'
    paper_year = {}
    progress = 0
    year_dis = defaultdict(int)
    logging.info('starting to read paper years ...')
    for paper_id,year in query_op.query_database(sql):

        progress+=1

        if progress%10000000==0:
            logging.info('Read paper year， progress {}, {} paper has year ...'.format(progress,len(paper_year)))

        if paper_id in paper_ids_set:

            paper_year[paper_id] = int(year)

            year_dis[int(year)]+=1

    logging.info('Done, {}/{} paper has year ...'.format(len(paper_year),len(paper_ids)))
    open(pathObj._field_paper_year_path,'w').write(json.dumps(paper_year))
    logging.info('Data saved to data/mag_{}_paper_year.json'.format(pathObj._field_paper_year_path))

    open(pathObj._field_paper_num_dis_path,'w').write(json.dumps(year_dis))


## 读取选取的文章的venue id
def read_paper_venue(pathObj):

    paper_year = json.loads(open(pathObj._field_paper_year_path).read())
    logging.info('paper year dict loaded ..')
    progress = 0

    f = open(pathObj._paper_venue_path,'w')

    query_op = dbop()
    sql = 'select paper_id,journal_id,conference_series_id,conference_instance_id from mag_core.papers'
    lines = ['pid,journal_id,conference_series_id,conference_instance_id']
    for paper_id,journal_id,conference_series_id,conference_instance_id in query_op.query_database(sql):

        progress+=1

        if progress%10000000==0:
            logging.info('progress {} ...'.format(progress))

        if paper_year.get(paper_id,None) is None:
            continue

        lines.append('{},{},{},{}'.format(paper_id,journal_id,conference_series_id,conference_instance_id))

        if len(lines)%100000==0:
            f.write('\n'.join(lines)+'\n')

            lines = []

    if len(lines)!=0:
        f.write('\n'.join(lines)+'\n')

    f.close()

    logging.info('paper venue saved to {}.'.format(pathObj._paper_venue_path))


## 读取引用关系，所有引用关系必须在上述id的范围内,并且控制时间在2016年之前
def red_ref_relations(pathObj,cut_year):

    ##目标ID列表
    paper_year = pathObj.paper_year
    ## 参考关系存放文件
    ref_relation_file = open(pathObj._paper_ref_relation_path,'a+')

    sql = 'select paper_id,paper_reference_id from mag_core.paper_references'
    cit_relations = []
    query_op = dbop()
    total_num = 0

    progress = 0
    for paper_id,paper_reference_id in query_op.query_database(sql):

        progress+=1

        if progress%100000000==0:
            logging.info('progress {:}, {} ref realtions saved.'.format(progress,total_num))

        if int(paper_year.get(paper_id,9999))<1970:
            continue

        if int(paper_year.get(paper_reference_id,9999))<1970:
            continue

        if int(paper_year.get(paper_id,9999))<=cut_year and int(paper_year.get(paper_reference_id,9999))<=cut_year:
            cit_relation = '{},{}'.format(paper_id,paper_reference_id)
            cit_relations.append(cit_relation)

            ## 每100万条存储一次
            if len(cit_relations)%10000000==0:
                ref_relation_file.write('\n'.join(cit_relations)+'\n')
                total_num+=len(cit_relations)
                cit_relations = []

    if len(cit_relations)>0:
        total_num+=len(cit_relations)
        ref_relation_file.write('\n'.join(cit_relations)+'\n')

    ref_relation_file.close()
    logging.info('{} ref relations saved to {}'.format(total_num,pathObj._paper_ref_relation_path))

def plot_citation_distribution(pathObj):

    paper_year = json.loads(open(pathObj._field_paper_year_path).read())

    pid_citnum = defaultdict(int)
    pids = []

    pid_intervals = defaultdict(list)
    for line in open(pathObj._paper_ref_relation_path):

        line = line.strip()

        citing_pid,cited_pid = line.split(',')

        pid_citnum[cited_pid]+=1

        citing_year = int(paper_year[citing_pid])
        cited_year = int(paper_year[cited_pid])

        interval = citing_year-cited_year

        pid_intervals[cited_pid].append(interval)

        pids.append(citing_pid)
        pids.append(cited_pid)

    logging.info('{} papers published till 2016,{} papers has citations.'.format(len(set(pids)),len(pid_citnum)))

    xs = []
    ys = []

    ## 根据文章的论文被引次数，来获取分界值
    cits_list = sorted(pid_citnum.values(),reverse=True)
    total = float(len(cits_list))
    ps = [0.01,0.05,0.1,0.2,0.4,0.7]

    for p in ps:
        print('p:',p,',cit:',cits_list[int(total*p)])

    citnum_counter = Counter(pid_citnum.values())
    for num in sorted(citnum_counter.keys()):

        xs.append(num)
        ys.append(citnum_counter[num]/float(len(pid_citnum)))


    plt.figure(figsize=(4,3))

    plt.plot(xs,ys,'o',fillstyle='none')

    plt.xscale('log')

    plt.yscale('log')

    plt.xlabel('number of citations')

    plt.ylabel('percentage of papers')

    plt.tight_layout()

    plt.savefig(pathObj._field_citation_dis_fig,dpi=300)

    logging.info('citation distribution saved to {}'.format(pathObj._field_citation_dis_fig))

    ## get id of papers not filter out
    reserved_pids = []
    for pid in pid_intervals.keys():

        max_interval = np.max(pid_intervals[pid])

        num = pid_citnum[pid]

        if max_interval<5 or num<5:
            continue

        reserved_pids.append(pid)

    logging.info('{} papers reserved.'.format(len(reserved_pids)))

    open(pathObj._reserved_papers_path,'w').write('\n'.join(reserved_pids))

    logging.info('reserved papers saved to {}.'.format(pathObj._reserved_papers_path))


## 获得文章的作者信息
def read_paper_authors(pathObj):
    ## 文章 年份
    paper_year = pathObj.paper_year
    progress = 0
    author_papers = defaultdict(list)
    paper_authors = defaultdict(list)

    query_op = dbop()

    lines = ['paper_id,author_id,author_name,aff_id,aff_name,author_sequence_number,year']
    sql = 'select paper_id,mag_core.authors.author_id,mag_core.authors.normalized_name,mag_core.affiliations.affiliation_id,mag_core.affiliations.normalized_name,author_sequence_number from mag_core.paper_author_affiliations,mag_core.authors,mag_core.affiliations where mag_core.paper_author_affiliations.author_id=mag_core.authors.author_id and mag_core.paper_author_affiliations.affiliation_id=mag_core.affiliations.affiliation_id'
    for paper_id,author_id,author_name,aff_id,aff_name,author_sequence_number in query_op.query_database(sql):

        progress+=1
        if progress%10000000==0:
            logging.info('read author id {} ...'.format(progress))

        pub_year = int(paper_year.get(paper_id,9999))
        # if pub_year > 2016 or pub_year < 1970:
        #     continue

        ## 在这里需要的是对该作者在该领域的所有论文信息,需要保留作者1970年发表论文的记录，用于记录作者的career的长度
        if pub_year>2016:
            continue


        line = '{},{},{},{},{},{},{}'.format(paper_id,author_id,author_name,aff_id,aff_name,author_sequence_number,pub_year)

        lines.append(line)

    open(pathObj._paper_author_aff_path,'w').write('\n'.join(lines))
    logging.info("paper author affiliations saved to {}.".format(pathObj._paper_author_aff_path))


# 画出随着时间的文章数量变化曲线
def plot_paper_year_dis(year_dis_path,outfig):

    year_dis = json.loads(open(year_dis_path).read())

    xs = []
    ys = []

    for x in sorted(year_dis.keys(),key= lambda x:int(x)):

        if int(x)<1970:
            continue

        if int(x)>2016:
            continue

        xs.append(int(x))
        ys.append(year_dis[x])

    plt.figure(figsize=(4,3))

    plt.plot(xs,ys,linewidth=2)

    plt.xlabel("year")
    plt.ylabel("number of papers")

    plt.yscale('log')

    plt.title('Number of computer science papers over years')


    plt.tight_layout()

    plt.savefig(outfig,dpi=400)

    logging.info('{} papers,Fig saved to {}'.format(np.sum(ys),outfig))


def Hindex(indexList):
    indexSet = sorted(list(set(indexList)), reverse = True)
    for index in indexSet:
        #clist为大于等于指定引用次数index的文章列表
        clist = [i for i in indexList if i >=index ]
        #由于引用次数index逆序排列，当index<=文章数量len(clist)时，得到H指数
        if index <=len(clist):
            break
    return index


## 根据论文-作者,论文-机构，论文-期刊 关系生成每一位作者、机构、venue的h-index，impact factor，作者的career的长度
def hindex_of_au_ins(pathObj):

    paper_year = json.loads(open(pathObj._field_paper_year_path).read())

    ## 加载论文随着时间的引用次数
    pid_year_citnum = defaultdict(lambda: defaultdict(int))
    for line in open(pathObj._paper_ref_relation_path):
        line = line.strip()
        citing_pid,cited_pid = line.split(',')

        pid_year_citnum[cited_pid][int(paper_year[citing_pid])]+=1

    open(pathObj._paper_year_citations_path,'w').write(json.dumps(pid_year_citnum))
    logging.info('paper yearly citation number saved.')

    pid_year_totalcit = defaultdict(lambda:defaultdict(int))

    for pid in pid_year_citnum.keys():
        total = 0
        year_citnum = pid_year_citnum[pid]

        years = sorted(year_citnum.keys())
        cits = []
        for year in range(years[0],2017):
            cits.append(year_citnum.get(year,0))

        total = 0
        for i,year in enumerate(range(years[0],2017)):
            total+=cits[i]
            pid_year_totalcit[pid][year] = total

    author_year_paper = defaultdict(lambda:defaultdict(list))
    ins_year_paper = defaultdict(lambda:defaultdict(list))

    author_year_papernum = defaultdict(lambda:defaultdict(int))

    pid_inses = defaultdict(list)

    ## 论文与作者关系
    for line in open(pathObj._paper_author_aff_path):
        paper_id,author_id,author_name,aff_id,aff_name,author_sequence_number,year = line.strip().split(',')

        if paper_id =='paper_id':
            continue

        if author_id!='':
            author_year_paper[author_id][int(year)].append(paper_id)

            author_year_papernum[author_id][int(year)]+=1

        if aff_id!='':
            ins_year_paper[aff_id][int(year)].append(paper_id)

            pid_inses[paper_id].append(aff_id)


    open(pathObj._paper_ins_path,'w').write(json.dumps(pid_inses))
    logging.info('paper inses saved to {}.'.format(pathObj._paper_ins_path))

    open(pathObj._author_year_papernum_path,'w').write(json.dumps(author_year_papernum))

    logging.info('author paper num saved to {} , Stat author hindex ...'.format(pathObj._author_year_papernum_path))


    author_starty = {}
    ## 作者的h-index 以及 career的长度进行计算
    author_year_hix = defaultdict(dict)
    for author_id in author_year_paper.keys():

        year_papers = author_year_paper[author_id]

        years = sorted(year_papers.keys())

        ## 作者第一篇论文的发表时间
        author_starty[author_id] = np.min(years)

        pids = []
        ## 这里的年份应该是从第一年到2016年
        for year in range(years[0],2017):

            ## 对所有作者1970年开始计算h-index

            if year<1970:
                continue

            ## 该年及之前发表所有论文
            pids.extend(year_papers[year])

            cits = []
            for pid in pids:

                ## 对于每一篇论文，获得该论文该年被引用的总次数,如果篇论文没有被引用 返回0
                cits.append(pid_year_totalcit[pid][year])

            ## 计算该作者当年的h-index
            if len(cits)==0:
                hix = 0
            else:
                hix = Hindex(cits)
            author_year_hix[author_id][year] = hix

    open(pathObj._author_start_time_path,'w').write(json.dumps(author_starty))
    logging.info('author start year saved.')

    open(pathObj._author_year_hix_path,'w').write(json.dumps(author_year_hix))
    logging.info('author yearly h-index saved.')

    ## impact facotr是当前对前两年发表的期刊的平均引用次数
    ins_year_if = defaultdict(dict)
    for ins_id in ins_year_paper.keys():

        year_papers = ins_year_paper[ins_id]
        years = sorted([y for y in year_papers.keys() if y>=1970])

        if len(years)==0:
            continue

        ## 时间同样从发表年份到2016年
        for year in range(years[0],2017):
            ## 获得前两年发表的论文
            pids = []

            if year-1>=1970:
                pids.extend(ins_year_paper[ins_id][year-1])

            if year-2>=1970:
                pids.extend(ins_year_paper[ins_id][year-2])

            ## 获得前两年发表的论文在今年的引用次数
            cits = []
            for pid in pids:
                ## 对于每一篇论文，获得去年以及前年的被引用次数
                cits.append(pid_year_citnum[pid][year])

            if len(cits)==0:
                IF = 0
            else:
                IF = np.mean(cits)

            ins_year_if[ins_id][year] = IF

    open(pathObj._ins_year_if_path,'w').write(json.dumps(ins_year_if))
    logging.info('institutes yearly if saved.')

def venue_if(pathObj):
    paper_year = json.loads(open(pathObj._field_paper_year_path).read())

    pid_year_citnum = json.loads(open(pathObj._paper_year_citations_path).read())

    venue_year_paper = defaultdict(lambda:defaultdict(list))
    count = 0

    paper_venue_id = {}

    total = 0
    for line in open(pathObj._paper_venue_path):

        paper_id,journal_id,conf_series_id,conf_inst_id = line.strip().split(',')

        if paper_id=='pid':
            continue

        year = int(paper_year[paper_id])
        venue_id=None
        if journal_id!='':
            venue_id = 'J_'+journal_id
        if conf_series_id!='':
            venue_id = 'C_'+conf_series_id

        if conf_inst_id!='':
            venue_id = 'I_'+conf_inst_id

        if venue_id is None:
            count+=1
            continue

        total+=1

        paper_venue_id[paper_id] = venue_id

        venue_year_paper[venue_id][year].append(paper_id)

    logging.info('{}/{} papers do not have venue info.'.format(count,total))

    open(pathObj._paper_venueid_path,'w').write(json.dumps(paper_venue_id))

    ## ins的hindex
    progress=0
    totalnum = len(venue_year_paper.keys())
    venue_year_if = defaultdict(dict)
    for venue_id in venue_year_paper.keys():

        progress +=1
        if progress%10000==0:
            logging.info('progress {}/{} ...'.format(progress,totalnum))


        year_papers = venue_year_paper[venue_id]
        years = sorted([y for y in year_papers.keys() if y>=1970])

        if len(years)==0:
            continue

        for year in range(years[0],2017):

            pids= []

            if year-1>=1970:
                pids.extend(venue_year_paper[venue_id][year-1])

            if year-2>=1970:
                pids.extend(venue_year_paper[venue_id][year-2])

            ## 获得前两年发表的论文在今年的引用次数
            cits = []
            for pid in pids:

                year_num = pid_year_citnum.get(pid,None)

                if year_num is None:
                    cits.append(0)
                else:
                    num = year_num.get(str(year),0)
                    ## 对于每一篇论文，获得去年以及前年的被引用词素
                    cits.append(num)

            if len(cits)==0:
                IF = 0
            else:
                IF = np.mean(cits)

            venue_year_if[venue_id][year] = IF

    open(pathObj._venue_year_if_path,'w').write(json.dumps(venue_year_if))
    logging.info('venue yearly if saved.')



if __name__ == '__main__':
    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    read_paper_ids(pathObj)

    read_paper_venue(pathObj)

     ## 画出数量随时间变化曲线
    plot_paper_year_dis(pathObj._field_paper_num_dis_path,pathObj._field_paper_num_dis_over_time_fig)

    red_ref_relations(pathObj,2016)

    plot_citation_distribution(pathObj)

    read_paper_authors(pathObj)

    hindex_of_au_ins(pathObj)
    venue_if(pathObj)

    logging.info('done')


