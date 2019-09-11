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

    sql = "select paper_id from mag_core.paper_fields_of_study, mag_core.fields_of_study where mag_core.paper_fields_of_study.field_of_study_id = mag_core.fields_of_study.field_of_study_id and mag_core.fields_of_study.normalized_name='{:}'".format(field)
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
    for paper_id,year in query_op.query_database(sql):

        progress+=1

        if progress%10000000==0:
            logging.info('Read paper year， progress {}, {} paper has year ...'.format(progress,len(paper_year)))

        if paper_id in paper_ids:

            paper_year[paper_id] = year

            year_dis[int(year)]+=1

    logging.info('Done, {}/{} paper has year ...'.format(len(paper_year),len(paper_ids)))
    open(pathObj._field_paper_year_path,'w').write(json.dumps(paper_year))
    logging.info('Data saved to data/mag_{}_paper_year.json'.format(pathObj._field_paper_year_path))

    ## 画出数量随时间变化曲线
    plot_paper_year_dis(year_dis,pathObj._field_paper_num_dis_over_time_fig)


## 读取引用关系，所有引用关系必须在上述id的范围内,并且控制时间在2016年之前
def red_ref_relations(pathObj,cut_year):

    ##目标ID列表
    paper_year = pathObj.paper_year
    ## 参考关系存放文件
    ref_relation_file = open(pathObj._paper_ref_relation_path,'a')

    sql = 'select paper_id,paper_reference_id from mag_core.paper_references'
    cit_relations = []
    query_op = dbop()
    total_num = 0

    progress = 0
    for paper_id,paper_reference_id in query_op.query_database(sql):

        progress+=1

        if progress&100000000==0:
            logging.info('progress {:}, {} ref realtions saved.'.format(progress,total_num))

        if paper_year.get(paper_id,9999)<=cut_year and paper_year.get(paper_reference_id,9999)<=cut_year:
            cit_relation = '{},{}'.format(paper_id,paper_reference_id)
            cit_relation.append(cit_relation)

            ## 每100万条存储一次
            if len(cit_relations)%10000000==0:
                ref_relation_file.write('\n'.join(cit_relations)+'\n')
                total_num+=len(cit_relations)
                cit_relations = []

    if len(cit_relations)>0:
        ref_relation_file.write('\n'.join(cit_relations)+'\n')

    ref_relation_file.close()
    logging.info('{} ref relations saved to {}'.format(total_num,pathObj._paper_ref_relation_path))


def read_paper_authors():
    ## 根据得到的id列表，从mag_core.paper_author_affiliations 存储的是每一篇论文对应的作者的id以及作者顺序
    progress = 0

    author_papers = defaultdict(list)
    paper_authors = defaultdict(list)
    sql = 'select paper_id,author_id,author_sequence_number from mag_core.paper_author_affiliations'
    for paper_id,author_id,author_sequence_number in query_op.query_database(sql):

        progress+=1

        if progress%10000000==0:
            print('read author id {} ...'.format(progress))

        if paper_fields.get(paper_id,None) is None:
            continue

        author_papers[author_id].append([paper_id,author_sequence_number])
        paper_authors[paper_id].append([author_id,author_sequence_number])

    print('There are {} authors in this field..'.format(len(author_papers)))
    open('data/mag_{}_author_papers.json'.format(tag),'w').write(json.dumps(author_papers))
    print('author papers json saved to data/mag_{}_author_papers.json'.format(tag))
    open('data/mag_{}_paper_authors.json'.format(tag),'w').write(json.dumps(paper_authors))
    print('author papers json saved to data/mag_{}_paper_authors.json'.format(tag))
    print('Done')


# 画出随着时间的文章数量变化曲线
def plot_paper_year_dis(year_dis,outfig):
    xs = []
    ys = []

    for x in sorted(year_dis.keys()):
        xs.append(x)
        ys.append(year_dis[x])

    plt.figure(figsize=(4,3))

    plt.plot(xs,ys,linewidth=2)

    plt.xlabel("year")
    plt.ylabel("number of papers")

    plt.yscale('log')


    plt.tight_layout()

    plt.savefig(outfig,dpi=400)

    print('Fig saved to {}'.format(outfig))



if __name__ == '__main__':
    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    read_paper_ids(pathObj)
    red_ref_relations(pathObj)
   

