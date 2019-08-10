import re
import math
import pandas as pd
import numpy as np
import pyLDAvis.gensim
import datetime
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from string import punctuation as punc
from Thesis.Part1_20190321 import Part1


class Part3:
    def __init__(self):
        sql_1 = 'SELECT abstract ' \
              'FROM patents_data_v2 a ' \
              'INNER JOIN top75perc_patents b ' \
              'ON a.patent_id = b.patent_id'
        sql_2 = 'SELECT AB ' \
              'FROM savedrecs核心合集'
        self.data = Part1().select_data(sql_2)  ###关系整体LDA的数据是专利还是论文
        self.stop_word = Part1().stop_word

    # 计算困惑度
    def perplexity(self, ldamodel, testset, dictionary, size_dictionary, size_topics):
        prep = 0
        prob_doc_sum = 0
        topic_word_list = []  # 每行为一个主题，每个主题中存在词和该词在该主题中的对应概率
        for topic_id in range(size_topics):
            topic_word = ldamodel.show_topic(topic_id, size_dictionary)
            dic = {}
            for word, probability in topic_word:
                dic[word] = probability
            topic_word_list.append(dic)
        doc_topic_list = []
        for doc in testset:
            doc_topic_list.append(ldamodel.get_document_topics(doc, minimum_probability=0))
        testset_word_num = 0
        for i in range(len(testset)):  # 对于每个文档
            prob_doc = 0
            doc_word_num = 0
            for word_id, num in testset[i]:  # 对于文档中的每个词
                prob_word = 0
                doc_word_num = doc_word_num + num  # N_i = [j = 1, 各文档词总数]ΣN_j， N_j是序号为j的词的频数
                word = dictionary[word_id]
                for topic_id in range(size_topics):  # 对于文档中的每个主题
                    prob_topic = doc_topic_list[i][topic_id][1]  # p(z_t|d_i)
                    prob_topic_word = topic_word_list[topic_id][word]  # p(w_j|z_t)
                    prob_word += prob_topic*prob_topic_word  # p(w_j) = [t=1, 主题数]Σp(z_t|d_i)*p(w_j|z_t)
                prob_doc += math.log(prob_word)  # p(W|d_i) = [j=1, 各文档词总数]Σlog(p(w_j|d_i))
            prob_doc_sum += prob_doc  # [i=1, 文档总数]Σp(W|d_i)
            testset_word_num += doc_word_num  # [i=1, 文档总数]ΣN_i
        prep = math.exp(-prob_doc_sum/testset_word_num)
        return prep

    # 模型的困惑度探索
    def LDA_pre(self):
        import random
        data = self.data
        total_abstract = []
        wnl = WordNetLemmatizer()
        for item in data:  # 针对数据库中的每一行，分词，加入total_abstract
            # 分词
            abstract = str(item[0])
            abstract = re.sub(r"[{}]+".format(punc), " ", abstract)  # 去标点符号
            abstract = re.sub("\d+", " ", abstract)
            abstract_list = [word.lower() for word in abstract.split() if word.lower() in abstract.split()]  # 分词
            clean_abstract_list = [word for word in abstract_list if word not in self.stop_word]  # 去停用词
            # 进行词形还原，不进行词干提取
            clean_abstract_list = [wnl.lemmatize(word) for word in clean_abstract_list]
            total_abstract.append(clean_abstract_list)
        # 创建字典、语料库
        dictionary = corpora.Dictionary(total_abstract)
        corpus = [dictionary.doc2bow(abstract) for abstract in total_abstract]
        # 划分训练集、测试集
        train_size = int(round(len(corpus)*0.7))
        train_index = sorted(random.sample(range(len(corpus)), train_size))
        test_index = sorted(set(range(len(corpus)))-set(train_index))
        train_corpus = [corpus[i] for i in train_index]
        test_corpus = [corpus[j] for j in test_index]
        # 选择最优主题数
        topic_count = []
        topic_perp = []
        for topic_num in range(1, 151, 10):
            print('测试主题数为：', topic_num)
            ldamodel = models.ldamodel.LdaModel(train_corpus, num_topics=topic_num, id2word=dictionary, passes=20)
            prep = self.perplexity(ldamodel, train_corpus, dictionary, len(dictionary.keys()), topic_num)
            topic_count.append(topic_num)
            topic_perp.append(prep)
        # 绘制主题数-困惑度图像
        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig = plt.figure()
        plt.plot(topic_count, topic_perp)
        plt.xticks(topic_count)
        plt.xlabel('主题数')
        plt.ylabel('困惑度')
        plt.title('5-60主题数下的困惑度')
        fig.tight_layout()
        plt.savefig('LDA_topic_perplexity', dpi=300)
        plt.show()

    # LDA模型
    def LDA(self, sql, model_save_path, save_word_path, save_num_path):
        data = Part1().select_data(sql)
        total_abstract = []
        wnl = WordNetLemmatizer()
        for item in data:  # 针对数据库中的每一行，分词，加入total_abstract
            # 分词
            abstract = str(item[0])
            abstract = re.sub(r"[{}]+".format(punc), " ", abstract)  # 去标点符号
            abstract = re.sub("\d+", " ", abstract)
            abstract_list = [word.lower() for word in abstract.split() if word.lower() in abstract.split()]  # 分词
            clean_abstract_list = [word for word in abstract_list if word not in self.stop_word]  # 去停用词
            # 进行词形还原，不进行词干提取
            clean_abstract_list = [wnl.lemmatize(word) for word in clean_abstract_list]
            total_abstract.append(clean_abstract_list)
        # 创建字典、语料库
        dictionary = corpora.Dictionary(total_abstract)
        corpus = [dictionary.doc2bow(abstract) for abstract in total_abstract]
        ldamodel = models.ldamodel.LdaModel(corpus, num_topics=11, id2word=dictionary, passes=20)
        # 保存模型
        ldamodel.save(model_save_path)
        # 分配各个主题的文献
        total_topic_dis = [0]*11
        doc_num_dict = dict()
        for i in range(len(corpus)):
            num = data[i][1]
            # 识别各文献的主要主题
            doc = corpus[i]
            doc_topic = ldamodel.get_document_topics(doc)
            topic = -1
            pro = 0
            for pro_item in doc_topic:
                if pro_item[1] > pro:
                    topic = pro_item[0]
                    pro = pro_item[1]
            total_topic_dis[topic] = total_topic_dis[topic] + 1
            # 将不同主题的专利划分存入各主题
            if topic not in list(doc_num_dict.keys()):
                doc_num_dict[topic] = [num]
            else:
                doc_num_dict[topic] = doc_num_dict[topic] + [num]

        doc_num_dict = dict([(k, pd.Series(v)) for k, v in doc_num_dict.items()]) ###填充不等长的列表，使其可以转化为dataframe
        doc_num_df = pd.DataFrame(doc_num_dict)
        doc_num_df = doc_num_df[[i for i in range(11)]]
        doc_num_df.columns = ['主题' + str(i) for i in range(1, 12)]
        doc_num_df.to_csv(save_num_path, index=False)

        # 将11个主题，各主题8个词结果存表
        topic_word = ldamodel.print_topics(num_topics=11, num_words=8)
        topic_word_dict = dict()
        for item in topic_word:
            pre = [word for word in re.findall('\"(\w+)\"', item[1])] + [total_topic_dis[item[0]]]
            topic_word_dict['主题' + str(item[0]+1)] = pre
        topic_word_df = pd.DataFrame(topic_word_dict)
        topic_word_df = topic_word_df[['主题' + str(i) for i in range(1, 12)]]
        topic_word_df.to_csv(save_word_path, index=False)
        # 可视化
        # vis_data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
        # pyLDAvis.show(vis_data)

    # 按年份进行主题提取
    def LDA_by_year(self, sql, prefix):
        data = Part1().select_data(sql)

        # 对整体文本进行处理，针对数据库中的每一行，年份+分词，加入total_abstract
        total_abstract = []
        wnl = WordNetLemmatizer()
        for item in data:
            year = item[0][:4]
            # 分词
            abstract = str(item[1])
            abstract = re.sub(r"[{}]+".format(punc), " ", abstract)  # 去标点符号
            abstract = re.sub("\d+", " ", abstract)
            abstract_list = [word.lower() for word in abstract.split() if word.lower() in abstract.split()]  # 分词
            clean_abstract_list = [word for word in abstract_list if word not in self.stop_word]  # 去停用词
            # 进行词形还原，不进行词干提取
            clean_abstract_list = [year] + [wnl.lemmatize(word) for word in clean_abstract_list]
            total_abstract.append(clean_abstract_list)

        # 按年份创建字典、语料库和LDA模型
        for year in ['2014', '2015', '2016', '2017', '2018', '2019']:
            total_year_abstract = [item[1:] for item in total_abstract if item[0] == year]
            if total_year_abstract:
                # 创建字典、语料库
                dictionary = corpora.Dictionary(total_year_abstract)
                corpus = [dictionary.doc2bow(abstract) for abstract in total_year_abstract]
                ldamodel = models.ldamodel.LdaModel(corpus, num_topics=11, id2word=dictionary, passes=20)
                ldamodel.save('Part3/' + prefix + '_LDA_model/' + prefix + '_lda_' + year + '.model')
                # 计算各个主题的文献数量
                total_topic_dis = [0 for i in range(11)]
                for doc in corpus:
                    doc_topic = ldamodel.get_document_topics(doc)
                    topic = -1
                    pro = 0
                    for pro_item in doc_topic:
                        if pro_item[1] > pro:
                            topic = pro_item[0]
                            pro = pro_item[1]
                    total_topic_dis[topic] = total_topic_dis[topic]+1

                # 将11个主题，各主题8个词结果存表
                topic_word = ldamodel.print_topics(num_topics=11, num_words=8)
                topic_word_dict = dict()
                for item in topic_word:
                    pre = [word for word in re.findall('\"(\w+)\"', item[1])] + [total_topic_dis[item[0]]]
                    topic_word_dict['主题' + str(item[0]+1)] = pre
                topic_word_df = pd.DataFrame(topic_word_dict)
                topic_word_df = topic_word_df[['主题'+str(i) for i in range(1, 12)]]
                topic_word_df.to_csv('Part3/' + prefix + '_LDA_model/' + prefix + '_topic_word_' + year + '.csv', index=False)

    # 计算不同年份之间主题词的相似度
    def cal_topic_similarity(self, prefix):
        from gensim import corpora, models, similarities
        if prefix == 'patent':
            year_max = 2018
        elif prefix == 'thesis':
            year_max = 2019
        topic_sim_trend = pd.DataFrame()  # 将相邻年份的相似主题进行匹配的结果
        write = pd.ExcelWriter('Part3/' + prefix + '_LDA_model/sim.xlsx')
        for year in range(2014, year_max):
            topic_year_1 = pd.read_csv('Part3/' + prefix + '_LDA_model/' + prefix + '_topic_word_' + str(year) + '.csv', encoding='gbk').T
            topic_year_2 = pd.read_csv('Part3/' + prefix + '_LDA_model/' + prefix + '_topic_word_' + str(year+1) + '.csv', encoding='gbk').T
            topic_year_1_count = topic_year_1[topic_year_1.columns[-1]]
            topic_year_2_count = topic_year_2[topic_year_2.columns[-1]]
            topic_year_1 = topic_year_1.loc[:, topic_year_1.columns[:-1]].values.tolist()
            topic_year_2 = topic_year_2.loc[:, topic_year_2.columns[:-1]].values.tolist()
            dictionary = corpora.Dictionary(topic_year_1)  # df.values.tolist()可将dataframe转换为二维矩阵
            dictionary.add_documents(topic_year_2)
            corpus_1 = [dictionary.doc2bow(topic) for topic in topic_year_1]
            corpus_2 = [dictionary.doc2bow(topic) for topic in topic_year_2]
            index = similarities.SparseMatrixSimilarity(corpus_1, num_features=len(dictionary.keys()))
            sim = dict()
            for i in range(11):
                sim[str(year+1) + '_主题' + str(i+1)] = index[corpus_2[i]]
            sim_pd = pd.DataFrame(sim)
            sim_pd.index = [str(year) + '_主题' + str(j) for j in range(1, 12)]
            sim_pd.to_excel(write, sheet_name=str(year) + '_' + str(year + 1))

            # 将相邻年份的相似主题进行匹配
            compare_result = []
            year_topic_word_1 = pd.read_csv('Part3/' + prefix + '_LDA_model/' + prefix + '_topic_word_' + str(year) + '.csv', encoding='gbk', nrows=8)
            year_topic_word_2 = pd.read_csv('Part3/' + prefix + '_LDA_model/' + prefix + '_topic_word_' + str(year+1) + '.csv', encoding='gbk', nrows=8)
            for i in range(11):
                array = sim_pd.values
                sim_pd['max_index'] = np.argmax(array,axis=1)
                sim_pd['max_value'] = array.max(axis=1)
                idx = sim_pd['max_value'].idxmax()
                col = sim_pd.columns[sim_pd.loc[idx, 'max_index']]
                # 该年该主题的关键词
                idx_topic_word = ';'.join(list(year_topic_word_1[idx[5:]]))
                col_topic_word = ';'.join(list(year_topic_word_2[col[5:]]))
                compare_result.append([idx, idx_topic_word, topic_year_1_count[idx[5:]], col, col_topic_word, topic_year_2_count.loc[col[5:]]])
                sim_pd = sim_pd.drop([col, 'max_index', 'max_value'], axis=1)
                sim_pd = sim_pd.drop([idx], axis=0)

            topic_sim_trend_tmp = pd.DataFrame(compare_result, columns=[str(year), str(year)+'关键词', str(year)+'数量',
                                                                        str(year+1), str(year+1)+'关键词', str(year+1)+'数量'])
            if year == 2014:
                topic_sim_trend = topic_sim_trend_tmp
            else:
                topic_sim_trend_tmp = topic_sim_trend_tmp.drop([str(year)+'关键词', str(year) + '数量'], axis=1)
                topic_sim_trend = pd.merge(topic_sim_trend, topic_sim_trend_tmp, how='inner',
                                           left_on=topic_sim_trend.columns[-3], right_on=topic_sim_trend_tmp.columns[0])
        topic_sim_trend.to_csv('Part3/' + prefix + '_LDA_model/topic_sim_trend.csv')
        write.save()

    # 专利结果和论文结果之间的对比
    def sort_topic_word_one_table(self):
        sim_1 = pd.read_csv('Part3/patent_LDA_model/topic_sim_trend.csv', encoding='gbk')
        sim_2 = pd.read_csv('Part3/thesis_LDA_model/topic_sim_trend.csv', encoding='gbk')
        co_list =\
            []
        year_word_2019 = ';'.join(list(sim_2['2019关键词']))
        year_word_2019 = year_word_2019.split(';')
        print(year_word_2019)
        for year in range(2014, 2019):
            year_word_1 = ';'.join(list(sim_1[str(year)+'关键词']))
            year_word_2 = ';'.join(list(sim_2[str(year)+'关键词']))
            year_word_1_list = list(set(year_word_1.split(';')))
            year_word_2_list = list(set(year_word_2.split(';')))

            year_word_2019 = [word for word in year_word_2019 if word not in year_word_1_list+year_word_2_list]

            co_list_tmp = []
            for word in year_word_2_list:
                if word in year_word_1_list:
                    co_list_tmp.append(word)
                    year_word_1_list.remove(word)
                    year_word_2_list.remove(word)
            co_list.append([';'.join(co_list_tmp), ';'.join(year_word_1_list), ';'.join(year_word_2_list)])
        co_list.append(['', '', ';'.join(year_word_2019)])
        co_pd = pd.DataFrame(co_list, columns=['共同的关键词', '专利独有的关键词', '论文独有的关键词'], index=[i for i in range(2014, 2020)])
        co_pd.to_csv('Part3/co_result.csv')

    ###补充的内容###
    # 获取最大主题的专利数据,保存到数据库
    def get_max_topic_patent_data(self, sql, max_topic_num, max_topic_table_name):
        patent_data = Part1().read_data_by_Pandas(sql)
        print(patent_data)
        max_topic_patent = patent_data[patent_data['num'].isin(max_topic_num)]
        max_topic_patent.index = max_topic_patent['patent_id']
        del max_topic_patent['num']
        engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/thesis?charset=utf8')
        conn = engine.connect()
        max_topic_patent.to_sql(max_topic_table_name, conn, index=False, if_exists='replace',
                                dtype={'patent_id': VARCHAR(max_topic_patent['patent_id'].str.len().max())})
        conn.close()

    # 获得最大主题的各年份的专利质量得分列表，TOPSIS按年份计算
    def cal_TOPSIS_score(self, patent_data, table_name_prefix):
        patent_data['year'] = [date[:4] for date in list(patent_data.priority_date)]
        for year in range(2014, 2019):
            data = patent_data[patent_data['year'] == str(year)].copy()
            data_dt = pd.to_datetime(data.priority_date, format='%Y-%m-%d')
            date_seg = datetime.datetime(2019, 3, 8) - data_dt
            date_seg = pd.Series([i.days for i in date_seg])  # 按天数计的时间间隔
            data['cited_by_per_year'] = data.cited_by*365/date_seg
            del data['year']
            del data['priority_date']
            del data['cited_by']
            data_value = data[data.columns[1:]]  # 截取有数据的一部分
            data_value.index = data['patent_id']
            data_value = (data_value - data_value.min())/(data_value.max() - data_value.min())  # 归一化
            weight = pd.Series([0.15, 0.15, 0.15, 0.15, 0.1, 0.15, 0.15], index=data_value.columns)
            data_value = data_value*weight
            best_vector = data_value.max()  # 理想解
            worst_vector = data_value.min()  # 负理想解
            best_dist = (data_value-best_vector)**2
            best_dist = best_dist.sum(axis=1)
            worst_dist = (data_value-worst_vector)**2
            worst_dist = worst_dist.sum(axis=1)
            score = worst_dist**0.5/(best_dist**0.5+worst_dist**0.5)
            result = pd.DataFrame(score, columns=['score'])
            # 将数据添加到数据库中
            engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/thesis?charset=utf8')
            conn = engine.connect()
            result.to_sql(table_name_prefix + str(year), conn, index=True, if_exists='replace',
                          dtype={'patent_id': VARCHAR(result.index.get_level_values('patent_id').str.len().max())})
            conn.close()
            print('存储成功！')

    # 获取最大主题的各年份的词频
    def get_top75perc_patent_key_word(self, read_score_table_name_prefix, save_keyword_count_table_name_prefix):
        for year in range(2014, 2019):
            sql = 'SELECT a.patent_id, score, month, key_word, num ' \
                  'FROM ' \
                  '(SELECT patent_id, score ' \
                  'FROM ' + read_score_table_name_prefix + str(year) + ') a ' \
                  'INNER JOIN key_word_count_by_month_allpatents b ' \
                  'ON a.patent_id = b.patent_id'
            total_patent_key_word = Part1().read_data_by_Pandas(sql)
            last25perc_score = total_patent_key_word['score'].quantile(0.25)
            top75perc_patent_key_word = total_patent_key_word[total_patent_key_word['score'] > last25perc_score]
            top75perc_patent_key_word_sum = top75perc_patent_key_word.groupby('key_word')['num'].sum()
            top75perc_patent_key_word_sum = pd.DataFrame(top75perc_patent_key_word_sum, columns=['num'])
            top75perc_patent_key_word_sum = top75perc_patent_key_word_sum.sort_values(by='num', ascending=False)
            engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/thesis?charset=utf8')
            conn = engine.connect()
            top75perc_patent_key_word_sum.to_sql(save_keyword_count_table_name_prefix + str(year), conn, index=True,
            if_exists='replace', dtype={'key_word': VARCHAR(top75perc_patent_key_word_sum.index.get_level_values('key_word').str.len().max())})
            conn.close()

    # 补充部分的main
    def add_cal(self, first_year):
        # 划分所有专利的主题
        sql = 'SELECT abstract, num ' \
              'FROM patents_data_v2'
        save_model_path = 'Part3/add_cal_model'
        save_word_path = 'Part3/add_cal_topic_word.csv'
        save_num_path = 'Part3/add_cal_num.csv'
        # self.LDA(sql, save_model_path, save_word_path, save_num_path)
        # 确定最大的主题
        topic_word_df = pd.read_csv(save_word_path, encoding='gbk')
        topic_patent_count = [int(count) for count in list(topic_word_df.iloc[-1, :])]
        max_topic_index = topic_patent_count.index(max(topic_patent_count)) + 1  # 记录的序号以1开始
        # 确定最大的主题的相关专利序号
        topic_num_df = pd.read_csv(save_num_path, encoding='gbk')
        max_topic_num = list(topic_num_df['主题'+str(max_topic_index)])
        # 获取最大主题的专利数据,保存到数据库
        sql = 'SELECT num, patent_id, priority_date, family_num, patent_citations, non_patent_citations,' \
              'cited_by, claim_num, description_num, CPC_num ' \
              'FROM patents_data_v2'
        max_topic_table_name = 'max_topic_patent_data'
        # self.get_max_topic_patent_data(sql, max_topic_num, max_topic_table_name)
        # 计算各年的的专利得分,保存到数据库
        sql = 'SELECT patent_id, priority_date, family_num, patent_citations, non_patent_citations,' \
              'cited_by, claim_num, description_num, CPC_num ' \
              'FROM ' + max_topic_table_name
        per_year_quality_score_prefix = 'patent_quality_score_'
        topic_patent_data = Part1().read_data_by_Pandas(sql)
        # self.cal_TOPSIS_score(topic_patent_data, per_year_quality_score_prefix)
        # 筛选出各年份TOP75%的高质量专利中频次最高的关键词
        per_year_top75perc_keyword_count_prefix = 'top25perc_key_word_count_'
        self.get_top75perc_patent_key_word(per_year_quality_score_prefix, per_year_top75perc_keyword_count_prefix)
        # 计算各年份的专利总量
        # total_year_patent_count = pd.DataFrame()
        # for year in range(2014, 2019):
        #     sql = 'SELECT COUNT(patent_id) '\
        #           'FROM ' + per_year_quality_score_prefix + str(year)
        #     year_patent_count = Part1().read_data_by_Pandas(sql)
        #     year_patent_count.columns = [year]
        #     if year == 2014:
        #         total_year_patent_count = year_patent_count
        #     else:
        #         total_year_patent_count = pd.merge(total_year_patent_count, year_patent_count,
        #                                            left_index=True, right_index=True)

        # 对于某年筛选前20个关键词，查看在今后n年的TOP25%中的占比，以及LAST25%中的占比
        for year in range(2014, 2017):
            sql = 'SELECT key_word ' \
                  'FROM ' + per_year_top75perc_keyword_count_prefix + str(year) + ' '\
                  'ORDER BY num DESC LIMIT 20'
            top20keyword = Part1().read_data_by_Pandas(sql)
            top20keyword = list(top20keyword['key_word'])
            # total_year_patent_count = total_year_patent_count[[i for i in range(year, 2019)]]
            # print(total_year_patent_count)

            total_intop25_inlast25_prop = pd.DataFrame()
            for later_year in range(year, 2019):
                sql = 'SELECT a.patent_id, score, month, key_word, num ' \
                      'FROM ' \
                      '(SELECT patent_id, score ' \
                      'FROM ' + per_year_quality_score_prefix + str(later_year) + ') a ' \
                      'INNER JOIN key_word_count_by_month_allpatents b ' \
                      'ON a.patent_id = b.patent_id'
                total_patent_key_word = Part1().read_data_by_Pandas(sql)
                top25perc_score = total_patent_key_word['score'].quantile(0.75)
                last25perc_score = total_patent_key_word['score'].quantile(0.25)
                # 前后25%的专利关键词数据
                top25perc_patent_key_word = total_patent_key_word[(total_patent_key_word['score'] > top25perc_score)]
                last25perc_patent_key_word = total_patent_key_word[(total_patent_key_word['score'] < last25perc_score)]
                # 前后25%的专利关键词总和
                top25perc_key_word_count = top25perc_patent_key_word['num'].sum()
                last25perc_key_word_count = last25perc_patent_key_word['num'].sum()

                # 前后25%中属于高质量的专利关键词数据
                top20intop25perc_patent_key_word = top25perc_patent_key_word[(top25perc_patent_key_word['key_word'].isin(list(top20keyword)))]
                top20inlast25perc_patent_key_word = last25perc_patent_key_word[(last25perc_patent_key_word['key_word'].isin(list(top20keyword)))]
                top20intop25perc_patent_key_word = top20intop25perc_patent_key_word.groupby('key_word')['num'].sum().to_frame()
                top20inlast25perc_patent_key_word = top20inlast25perc_patent_key_word.groupby('key_word')['num'].sum().to_frame()
                top20intop25perc_patent_key_word.columns = [str(later_year) + '_top25_count']
                top20inlast25perc_patent_key_word.columns = [str(later_year) + '_last25_count']

                # 每年的高质量关键词占比
                top20intop25perc_patent_key_word = top20intop25perc_patent_key_word/top25perc_key_word_count
                print(top20intop25perc_patent_key_word)
                top20inlast25perc_patent_key_word = top20inlast25perc_patent_key_word/last25perc_key_word_count
                year_top_intop25_inlast25_prop = pd.merge(top20intop25perc_patent_key_word,
                                                          top20inlast25perc_patent_key_word,
                                                          how='outer', left_index=True, right_index=True)
                if later_year == year:
                    total_intop25_inlast25_prop = year_top_intop25_inlast25_prop
                else:
                    total_intop25_inlast25_prop = pd.merge(total_intop25_inlast25_prop,
                                                            year_top_intop25_inlast25_prop,
                                                            how='outer', left_index=True, right_index=True)
            # total_intop25_inlast25_count = total_intop25_inlast25_count.fillna(0)
            col = [str(year) + '_top25_count' for year in range(year, 2019)] + [str(year) + '_last25_count' for year in range(year, 2019)]
            total_intop25_inlast25_prop = total_intop25_inlast25_prop[col]
            # quanitle_col = pd.merge(total_year_patent_count, total_year_patent_count, left_index=True, right_index=True)
            # quanitle_col.columns = col
            # quanitle_col.index = ['patent_count']
            # quanitle_col = quanitle_col.T['patent_count']
            # total_intop25_inlast25_prop = total_intop25_inlast25_count.div(quanitle_col*0.25, axis=1)
            total_intop25_inlast25_prop.to_csv('Part3/v3_' + str(year) + 'top20keyword_inlater_prop.csv')


if __name__ == '__main__':
    C = Part3()
    # C.LDA_pre()
    # C.LDA()
    ###选择要分析的文献类型：专利OR论文###
    # paper_type = 'thesis'
    # if paper_type == 'patent':
    #     sql = 'SELECT filing_date, abstract ' \
    #            'FROM patents_data_v2 a ' \
    #            'INNER JOIN top75perc_patents b ' \
    #            'ON a.patent_id = b.patent_id'
    # elif paper_type == 'thesis':
    #    sql = 'SELECT PY, AB ' \
    #          'FROM savedrecs核心合集'
    # C.LDA_by_year(sql, paper_type)
    # C.cal_topic_similarity(paper_type)
    # C.sort_topic_word_one_table()

    ###补充方法###
    C.add_cal('2014')


