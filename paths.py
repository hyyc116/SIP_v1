#coding:utf-8
import json
class PATH:


    def __init__(self,field_name,field_tag):

        self._field_tag = field_tag

        self._field_name = field_name

        ''' ==========文件列表=============='''
        ##存储文章id列表文件地址
        self._field_paper_ids_path = 'data/paper_ids_{}.txt'.format(self._field_tag)
        ## 文章对应的发表年份
        self._field_paper_year_path = 'data/paper_year_{}.json'.format(self._field_tag)
        ## 存储参考关系的文件地址
        self._paper_ref_relation_path = 'data/paper_ref_relations_{}.txt'.format(self._field_tag)

        ## 文章数量随时间的变化数据统计
        self._field_paper_num_dis_path = 'data/paper_num_dis_{}.txt'.format(self._field_tag)

        ##文章，作者，机构关系地址
        self._paper_author_aff_path = 'data/paper_author_aff_{}.txt'.format(self._field_tag)

        ## 文章与venue的对应关系
        self._paper_venue_path = 'data/paper_venue_{}.csv'.format(self._field_tag)

        ### 过滤掉引用长度小于5 以及 引用次数小于5的论文之后的论文分布
        self._reserved_papers_path = 'data/reserved_papers_{}.txt'.format(self._field_tag)

        ## 论文对应的venue id
        self._paper_venueid_path = 'data/paper_vid_{}.json'.format(self._field_tag)


        ### 论文每年的引用次数
        self._paper_year_citations_path = 'data/paper_year_citation_{}.txt'.format(self._field_tag)

        ### 论文对应的inses
        self._paper_ins_path = 'data/paper_ins_{}.json'.format(self._field_tag)

        ### 作者h-index随着时间的变化
        self._author_year_hix_path = 'data/author_year_hix_{}.json'.format(self._field_tag)

        ### 作者每年发表的文章数量
        self._author_year_papernum_path = 'data/author_year_pnum_{}.json'.format(self._field_tag)

        ### 作者研究开始的年份
        self._author_start_time_path = 'data/_author_start_time_{}.json'.format(self._field_tag)

        ## 机构的impact factor随着时间的变化
        self._ins_year_if_path = 'data/ins_year_if_{}.json'.format(self._field_tag)

        ##  venue的IF随时间的变化
        self._venue_year_if_path = 'data/venue_year_if_{}.json'.format(self._field_tag)




        ''' ============== 图片列表 =========='''
        ### 随时间的文章数量变化曲线
        self._field_paper_num_dis_over_time_fig = 'fig/paper_num_dis_over_time_{}.png'.format(self._field_tag)

        ### 2016年前论文引用次数分布
        self._field_citation_dis_fig = 'fig/citation_dis_{}.png'.format(self._field_tag)



    def dataset_id_path(self,m,n):
        return 'data/sip_m{}_n{}_ids_{}.txt'.format(m,n,self._field_tag)

    def dataset_feature_path(self,m,n):
        return 'data/sip_m{}_n{}_ids_{}.json'.format(m,n,self._field_tag)



    @property
    def field_papers(self):
        paper_ids = []
        for paper_id in open(self._field_paper_ids_path):
            paper_ids.append(paper_id.strip())

        return paper_ids

    @property
    def paper_year(self):
        return json.loads(open(self._field_paper_year_path).read())



