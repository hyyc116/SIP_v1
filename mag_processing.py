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

'''
from basic_config import *
from paths import PATH
## 读出计算机领域的所有论文ID
def read_paper_ids(pathObj):

    field = pathObj._field_name

    query_op = dbop()

    sql = "select mag_core.paper_fields_of_study.paper_id from mag_core.paper_fields_of_study, mag_core.fields_of_study,mag_core.paper_languages where mag_core.paper_fields_of_study.field_of_study_id = mag_core.fields_of_study.field_of_study_id and mag_core.paper_fields_of_study.paper_id = mag_core.paper_languages.paper_id and mag_core.paper_languages.language_code='en' and mag_core.fields_of_study.normalized_name='{:}'".format(field)
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

    pid_citnum = defaultdict(int)
    pids = []
    for line in open(pathObj._paper_ref_relation_path):

        line = line.strip()

        citing_pid,cited_pid = line.split(',')

        pid_citnum[cited_pid]+=1

        pids.append(citing_pid)
        pids.append(cited_pid)

    logging.info('{} papers published till 2016,{} papers has citations.'.format(len(set(pids)),len(pid_citnum)))

    xs = []
    ys = []

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
            print('read author id {} ...'.format(progress))

        pub_year = int(paper_year.get(paper_id,9999))
        if pub_year > 2016 or pub_year < 1970:
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

    for x in sorted(year_dis.keys(),key= x:int(x)):

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



if __name__ == '__main__':
    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    # read_paper_ids(pathObj)

     ## 画出数量随时间变化曲线
    plot_paper_year_dis(pathObj._field_paper_num_dis_path,pathObj._field_paper_num_dis_over_time_fig)
    
    # red_ref_relations(pathObj,2016)

    # plot_citation_distribution(pathObj)

    # read_paper_authors(pathObj)
   

