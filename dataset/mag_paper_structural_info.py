#coding:utf-8
'''
论文的结构化信息抽取

1. disruptive score = \frac{n_i - n_j}{ n_i + n_j + n_k}

2. depth 

3. dependence

4. (edges-n)/n

'''
import sys
sys.path.extend(['..','.'])
from basic_config import *
from paths import PATH

def extract_structual_info(pathObj):

    paper_year = json.loads(open(pathObj._field_paper_year_path).read())
    pid_year_cits = defaultdict(lambda:defaultdict(set))
    pid_refs = defaultdict(set)
    logging.info('stating number of papers ...')
    for line in open(pathObj._paper_ref_relation_path):

        line = line.strip()

        citing_pid,cited_pid = line.split(',')

        citing_year = int(paper_year[citing_pid])
        cited_year = int(paper_year[cited_pid])

        pid_refs[citing_pid].add(cited_pid)
        pid_year_cits[cited_pid][citing_year].add(citing_pid)

    logging.info('year total cits ...')
    pid_year_tcits = defaultdict(lambda:defaultdict(set))
    for pid in pid_year_cits:

        year_cits = pid_year_cits[pid]
        cits = set()
        for year in sorted(year_cits.keys()):
            cits = cits | year_cits[year]

            pid_year_tcits[pid][year]  = cits

    ## 构建cascade然后计算
    pid_year_disruptives = defaultdict(dict)
    pid_year_depths = defaultdict(dict)
    pid_year_dependence = defaultdict(dict)
    pid_year_anlec = defaultdict(dict)

    logging.info('stating pid year cits ...')
    for pid in pid_year_tcits:

        refs = pid_refs[pid]
        ## 对于每篇论文来讲
        year_tcits = pid_year_tcits[pid]
        for year in sorted(year_tcits.keys()):

            cits = year_tcits[year]

            num_cc = len(cits)

            if num_cc==0:
                continue

            ## 计算n_k
            ref_citings  = []
            for ref in refs:
                ## 只引用其参考文献
                if len(pid_year_tcits[ref][year] & cits)==0:
                    ref_citings.extend(pid_year_tcits[ref][year])

            n_k = len(set(ref_citings))


            n_i = 0
            n_j = 0

            TR_citings = 0
            TR_citeds = 0

            ## 对于cits中每一篇引证文献
            for cit in cits:

                ##其参考文献集合
                cit_refs = pid_refs[cit]

                ## R_citing是引证文献与其参考文献共引本文的篇数
                R_citing = len(cit_refs&cits)
                ## R_cited 是本文与引证文献共同参考文献的数量
                R_cited = len(cit_refs&refs)

                TR_citings += R_citing
                TR_citeds += R_cited

                ## 如果可R_citings>0 说明都进行了引用,否者只引用了本文
                if R_citing>0:
                    n_j +=1
                else:
                    n_i +=1
            
            ## disruptive_score
            disruptive_score = (n_i-n_j)/float(n_k+n_j+n_i)
            pid_year_disruptives[pid][year] = disruptive_score

            ## depth
            depth = TR_citings/float(num_cc)
            dependence = TR_citeds/float(num_cc)
            pid_year_depths[pid][year] = depth
            pid_year_dependence[pid][year] = dependence

            ## anlec就是TR_citings
            pid_year_anlec[pid][year] = TR_citings
            
    logging.info('save info.')

    open(pathObj._paper_year_disruptive_path,'w').write(json.dumps(pid_year_disruptives))
    logging.info('data saved to {}'.format(pathObj._paper_year_disruptive_path))

    open(pathObj._paper_year_depth_path,'w').write(json.dumps(pid_year_depths))
    logging.info('data saved to {}'.format(pathObj._paper_year_depth_path))

    open(pathObj._paper_year_dependence_path,'w').write(json.dumps(pid_year_dependence))
    logging.info('data saved to {}'.format(pathObj._paper_year_dependence_path))

    open(pathObj._paper_year_anlec_path,'w').write(json.dumps(pid_year_anlec))
    logging.info('data saved to {}'.format(pathObj._paper_year_anlec_path))


if __name__ == '__main__':

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    extract_structual_info(pathObj)


