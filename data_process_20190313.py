import pymysql
import re, csv
from string import punctuation as punc
from gensim import corpora, models, similarities


class Process(object):
    def __init__(self):
        self.data = self.read_data()

    # 从数据库中获取所有数据
    def read_data(self):
        conn = pymysql.connect(host='localhost', user='root', passwd='123456', db='thesis', port=3306, charset='utf8')
        cur = conn.cursor()
        sql = 'SELECT * FROM patents_data_v2'
        cur.execute(sql)
        data = cur.fetchall()
        cur.close()
        conn.close()
        return data

    # 从数据库中删除数据
    def del_data(self, sql):
        conn = pymysql.connect(host='localhost', user='root', passwd='123456', db='thesis', port=3306, charset='utf8')
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.commit()
        conn.close()
        print('SQL执行完成！')

    # 计算摘要文本之间的相似度
    def cal_abstract_similarity(self):
        data = self.data
        total_abstract = []
        for item in data:  # 针对数据库中的每一行，分词，加入total_abstract
            abstract = item[-1]
            abstract = re.sub(r"[{}]+".format(punc), " ", abstract)
            abstract_list = abstract.split()
            total_abstract.append(abstract_list)
        dictionary = corpora.Dictionary(total_abstract)
        corpus = [dictionary.doc2bow(abstract) for abstract in total_abstract]
        tfidf = models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
        with open('sim_abstract.csv', 'w', newline='') as sim_result:
            csv_write = csv.writer(sim_result)
            for i in range(len(corpus)):
                sim = index[tfidf[corpus[i]]]
                sim = [round(s, 4) for s in sim]
                csv_write.writerow(sim[:i+1])
                print(i, sim[:i+1])
        print('摘要相似度写入完成!')

    # 计算标题文本之间的相似度
    def cal_title_similarity(self):
        data = self.data
        total_title = []
        for item in data:  # 针对数据库中的每一行，分词，加入total_abstract
            title = item[2]
            title = re.sub(r"[{}]+".format(punc), " ", title)
            title_list = title.split()
            total_title.append(title_list)
        dictionary = corpora.Dictionary(total_title)
        corpus = [dictionary.doc2bow(title) for title in total_title]
        tfidf = models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
        with open('sim_title.csv', 'w', newline='') as sim_result:
            csv_write = csv.writer(sim_result)
            for i in range(len(corpus)):
                sim = index[tfidf[corpus[i]]]
                sim = [round(s, 4) for s in sim]
                csv_write.writerow(sim[:i+1])
                print(i, sim[:i+1])
        print('标题相似度写入完成!')

    # 打印，删除相似度超过阈值的两文本中之一
    def del_similar_abstract(self, threshold):
        data = self.data
        sim_abstract_result = open('sim_abstract.csv', 'r')
        sim_title_result = open('sim_title.csv', 'r')
        csv_read1 = sim_abstract_result.readlines()
        csv_read2 = sim_title_result.readlines()
        with open('similar_patent_num_set_' + str(threshold*10) + '.csv', 'w', encoding='utf-8', newline='') as similar_patent:
            csv_write = csv.writer(similar_patent)
            count = 0
            total_similar_patent_list = set()
            for i in range(len(csv_read1)):
                sim_abs_list = csv_read1[i].split(',')
                sim_tit_list = csv_read2[i].split(',')
                similar_patent_list = [j+1 for j in range(len(sim_abs_list))
                                       if (float(sim_abs_list[j]) == threshold or float(sim_tit_list[j]) == threshold) and i != j
                                       and (j+1 not in similar_patent_list)]
                if len(similar_patent_list) > 0:
                    count += 1
                    total_similar_patent_list = set(list(total_similar_patent_list) + similar_patent_list)
                csv_write.writerow(tuple(similar_patent_list))
                print(similar_patent_list)
            csv_write.writerow(('总计行数', count))
            csv_write.writerow(('专利数量', len(total_similar_patent_list)))
        sim_abstract_result.close()
        sim_title_result.close()
        del_sql = 'DELETE FROM patents_data_v2 WHERE num in ('+','.join([str(i) for i in total_similar_patent_list])+')'
        self.del_data(del_sql)


if __name__ == '__main__':
    A = Process()
    # A.cal_abstract_similarity()
    # A.cal_title_similarity()
    # A.del_similar_abstract()
    # for i in range(5, 10):
        # A.del_similar_abstract(round(i/10, 1))