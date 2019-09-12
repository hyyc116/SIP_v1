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


        ''' ============== 图片列表 =========='''
        ### 随时间的文章数量变化曲线
        self._field_paper_num_dis_over_time_fig = 'fig/paper_num_dis_over_time_{}.png'.format(self._field_tag)

        ### 2016年前论文引用次数分布
        self._field_citation_dis_fig = 'fig/citation_dis_{}.png'.format(self._field_tag)



    @property
    def field_papers(self):
        paper_ids = []
        for paper_id in open(self._field_paper_ids_path):
            paper_ids.append(paper_id.strip())

        return paper_ids

    @property
    def paper_year(self):
        return json.loads(open(self._field_paper_year_path).read())
    


