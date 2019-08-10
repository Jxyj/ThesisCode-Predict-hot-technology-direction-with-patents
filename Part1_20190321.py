import pymysql
import pandas as pd
import datetime
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import re
from string import punctuation as punc


class Part1:
    def __init__(self):
        sql = 'SELECT patent_id, priority_date, family_num, patent_citations, non_patent_citations,' \
              'cited_by, claim_num, description_num, CPC_num ' \
              'FROM patents_data_v2'
        self.data = self.read_data_by_Pandas(sql)
        self.font = 'C:/Windows/Fonts/STSONG.TTF'
        self.stop_word = ['a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again',
                          'against', 'all', 'almost', 'alone', 'along', 'already', 'also','although',
                          'always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 'another',
                          'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',
                          'at', 'back','be','became', 'because','become','becomes', 'becoming', 'been',
                          'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                          'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can',
                          'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe',
                          'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight',
                          'either', 'eleven','else', 'elsewhere', 'empty', 'enough', 'etc', 'even',
                          'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few',
                          'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former',
                          'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get',
                          'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here',
                          'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him',
                          'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc',
                          'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last',
                          'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me',
                          'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
                          'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never',
                          'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not',
                          'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only',
                          'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out',
                          'over', 'own','part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same',
                          'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she',
                          'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some',
                          'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere',
                          'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their',
                          'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby',
                          'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third',
                          'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus',
                          'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two',
                          'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well',
                          'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
                          'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which',
                          'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
                          'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
                          'yourselves', 'the', '∑', 'π', 'φ', 'ϕ', 'αi', 'ηr', 'μs', 'jϕ', '，', '；',
                          '·', '·', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i','j', 'k', 'l', 'm', 'n', 'o', 'p',
                          'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def read_data_by_Pandas(self, sql):
        conn = pymysql.connect(host='localhost', user='root', passwd='123456', db='thesis', port=3306, charset='utf8')
        data_df = pd.read_sql(sql, conn)
        conn.close()
        return data_df

    def select_data(self, sql):
        conn = pymysql.connect(host='localhost', user='root', passwd='123456', db='thesis', port=3306, charset='utf8')
        cur = conn.cursor()
        cur.execute(sql)
        data = cur.fetchall()
        cur.close()
        conn.close()
        return data

    # 1.TOPSIS方法，筛选出TOP75%的专利
    def TOPSIS(self, patent_data, table_name):
        data = patent_data.copy()
        data_dt = pd.to_datetime(data.priority_date, format='%Y-%m-%d')
        date_seg = datetime.datetime(2019, 3, 8) - data_dt
        date_seg = pd.Series([i.days for i in date_seg])  # 按天数计的时间间隔
        data['cited_by_per_year'] = data.cited_by*365/date_seg
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
        result = score[score > score.quantile(0.25)]
        result = pd.DataFrame(result, columns=['score'])
        # 将数据添加到数据库中
        engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/thesis?charset=utf8')
        conn = engine.connect()
        result.to_sql(table_name, conn, index=True, if_exists='replace',
                      dtype={'patent_id': VARCHAR(result.index.get_level_values('patent_id').str.len().max())})
        conn.close()
        print('存储成功！')

    # 2.制作授让人词云
    def paint_word_cloud(self, sql, file_name):
        word_frequency = self.select_data(sql)
        word_dict = {}
        for item in word_frequency:
            word_dict[item[0]] = item[1]
        word_cloud = WordCloud(
            scale=64,
            background_color='white',
            max_words=500,
            font_path=self.font
        ).generate_from_frequencies(word_dict)
        plt.figure()
        plt.imshow(word_cloud)
        plt.axis('off')
        plt.show()
        word_cloud.to_file(file_name)

    # 3.1.获取关键词CPC分类
    def get_CPC(self):
        sql = 'SELECT a.patent_id, SUBSTR(a.filing_date, 1, 7), a.CPC ' \
               'FROM patents_data_v2 a ' \
               'INNER JOIN top75perc_patents b ' \
               'ON a.patent_id = b.patent_id'
        # sql = 'SELECT patent_id, SUBSTR(filing_date, 1, 7), abstract ' \
        #       'FROM patents_data_v2'
        data = self.select_data(sql)
        result = []
        for item in data:  # 针对数据库中的每一行，分词，加入total_abstract
            CPC = item[2]
            CPC = CPC.split(',')
            CPC_list = [code[:4] for code in CPC]
            # 计算频率
            freq = nltk.FreqDist(CPC_list)
            for key, val in freq.items():
                result.append([item[0], item[1], key, val])
        result_df = pd.DataFrame(result, columns=['patent_id', 'month', 'CPC', 'num'])
        # 将数据添加到数据库中
        engine = create_engine('mysql+pymysql://root:123456@localhost:3306/thesis?charset=utf8')
        conn = engine.connect()
        result_df.to_sql('CPC_code_count_by_month', conn, index=False, if_exists='replace')
        conn.close()
        print('存储成功！')

    # 3.2.绘制CPC数量图
    def paint_CPC_count(self):
        sql = 'SELECT CPC ' \
              'FROM cpc_code_count_by_month'
        CPC = self.select_data(sql)
        CPC_dict = {}
        for CPC in CPC:
            if CPC in CPC_dict.keys():
                CPC_dict[CPC] = CPC_dict[CPC] + 1
            else:
                CPC_dict[CPC] = 1
        CPC_df = pd.DataFrame(CPC_dict, index=['数量']).T
        CPC_df.plot(kind='bar')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.legend()
        plt.xlabel('CPC代码')
        plt.ylabel('数量')
        plt.title('CPC的专利申请数量')
        plt.tight_layout()  # 调整整体空白
        plt.savefig('CPC的专利申请数量' + '.jpg')
        plt.show()

    # 3.3.CPC申请量变化图
    def paint_CPC_count_by_year(self):
        sql1 = 'SELECT month, COUNT(DISTINCT patent_id) ' \
               'FROM CPC_code_count_by_month ' \
               'WHERE CPC = "%s" AND month > "2014-01" ' \
               'GROUP BY month ' \
               'ORDER BY month'
        sql2 = 'SELECT month, COUNT(DISTINCT patent_id) ' \
               'FROM CPC_code_count_by_month ' \
               'GROUP BY month ' \
               'ORDER BY month'
        sql3 = 'SELECT CPC ' \
               'FROM cpc_code_count_by_month ' \
               'GROUP BY CPC ' \
               'ORDER BY COUNT(patent_id) DESC'

        data2 = self.select_data(sql2)
        data3 = self.select_data(sql3)
        x_list = [item[0] for item in data2 if item[0] >= '2014-01']
        x_num_list = [i for i in range(len(x_list))]
        CPC_code_list = [item[0] for item in data3]
        plt.rcParams['font.sans-serif'] = ['SimHei']
        for cpc in CPC_code_list[:6]:
            patent_count_by_cpc_month = self.select_data(sql1 % cpc)
            y_list = []
            i = 0
            j = 0
            while j < len(x_list):
                if i < len(patent_count_by_cpc_month) and patent_count_by_cpc_month[i][0] == x_list[j]:
                    y_list.append(patent_count_by_cpc_month[i][1])
                    i += 1
                    j += 1
                else:
                    y_list.append(0)
                    j += 1
            plt.plot(x_num_list, y_list, label=cpc)
        plt.xticks(x_num_list, x_list, rotation=90)
        plt.legend()
        plt.grid()
        plt.xlabel('月份')
        plt.ylabel('数量')
        plt.tight_layout()  # 调整整体空白
        plt.savefig('CPC申请量变化图.jpg', dpi=300)
        plt.show()

    # 4.绘制各年份的专利数量变化图
    def paint_patent_count(self, sql, title_name):
        filing_date = self.select_data(sql)
        filing_date_dict = {}
        for date in filing_date:
            year = date[0][:7]
            if year in filing_date_dict.keys():
                filing_date_dict[year] = filing_date_dict[year] + 1
            else:
                filing_date_dict[year] = 1
        filing_date_df = pd.DataFrame(filing_date_dict, index=['数量']).T
        filing_date_df.plot(kind='bar')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.legend()
        plt.xlabel('年份')
        plt.ylabel('数量')
        plt.title(title_name)
        plt.tight_layout()  # 调整整体空白
        plt.savefig(title_name + '.jpg')
        plt.show()

    # 5.1.分词，获取关键词
    def get_key_word(self, sql):
        # sql = 'SELECT a.patent_id, SUBSTR(a.filing_date, 1, 7), a.abstract ' \
        #       'FROM patents_data_v2 a ' \
        #       'INNER JOIN top75perc_patents b ' \
        #       'ON a.patent_id = b.patent_id'
        sql = 'SELECT patent_id, SUBSTR(filing_date, 1, 7), abstract ' \
              'FROM patents_data_v2'
        data = self.select_data(sql)
        result = []
        for item in data:  # 针对数据库中的每一行，分词，加入total_abstract
            abstract = item[2]
            abstract = re.sub(r"[{}]+".format(punc), " ", abstract)  # 去标点符号
            abstract = re.sub(r"\d+", " ", abstract)
            abstract_list = [word.lower() for word in abstract.split() if word.lower() in abstract.split()]  # 分词
            clean_abstract_list = [word for word in abstract_list if word not in self.stop_word]  # 去停用词
            # 计算频率
            freq = nltk.FreqDist(clean_abstract_list)
            for key, val in freq.items():
                result.append([item[0], item[1], key, val])
        result_df = pd.DataFrame(result, columns=['patent_id', 'month', 'key_word', 'num'])
        # 将数据添加到数据库中
        engine = create_engine('mysql+pymysql://root:123456@localhost:3306/thesis?charset=utf8')
        conn = engine.connect()
        result_df.to_sql('key_word_count_by_month_AllPatents', conn, index=False, if_exists='replace')
        conn.close()
        print('存储成功！')

    # 5.2.词数量变化图
    def paint_word_count(self, low, high, file_name):
        sql1 = 'SELECT month, COUNT(DISTINCT patent_id) ' \
               'FROM key_word_count_by_month ' \
               'WHERE key_word = "%s" AND month > "2014-01" ' \
               'GROUP BY month ' \
               'ORDER BY month'  # 给定关键词，获得其各月份覆盖的专利数量
        sql2 = 'SELECT month, COUNT(DISTINCT patent_id) ' \
               'FROM key_word_count_by_month ' \
               'GROUP BY month ' \
               'ORDER BY month'   # 获得各月份的专利总数量
        sql3 = 'SELECT key_word ' \
               'FROM key_word_count_top100 ' \
               'ORDER BY total_count DESC'
        data2 = self.select_data(sql2)
        data3 = self.select_data(sql3)

        # 1.先绘制月份的总专利数量
        x_list = []
        total_count_list = []
        for item in data2:
            if item[0] >= '2014-01':
                x_list.append(item[0])
                total_count_list.append(item[1])
        x_num_list = [i for i in range(len(x_list))]
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.bar(x_num_list, total_count_list, label='各月份专利数量', color='gainsboro')

        # 2.再绘制各关键词各月份的专利申请数量
        key_word_list = [item[0] for item in data3]
        for word in key_word_list[low:high]:
            patent_count_by_word_month = self.select_data(sql1 % word)
            y_list = []
            i = 0
            j = 0
            while j < len(x_list):
                if i < len(patent_count_by_word_month) and patent_count_by_word_month[i][0] == x_list[j]:
                    y_list.append(patent_count_by_word_month[i][1])
                    i += 1
                    j += 1
                else:
                    y_list.append(0)
                    j += 1
            plt.plot(x_num_list, y_list, label=word)
        plt.legend()
        plt.xticks(x_num_list, x_list, rotation=90)
        plt.xlabel('月份')
        plt.ylabel('数量')
        plt.tight_layout()  # 调整整体空白
        plt.savefig('/Part1/' + file_name + '.jpg', dpi=300)
        plt.show()

    # 5.3.近3年关键词的高质量专利占比
    def paint_high_quality_proportion(self, low, high, file_name):
        sql1 = 'SELECT a.key_word, a.proportion ' \
               'FROM high_quality_proportion a ' \
               'INNER JOIN key_word_count_top100 b ' \
               'ON a.key_word = b.key_word ' \
               'WHERE a.year = "%s" ' \
               'ORDER BY b.total_count DESC'
        sql2 = 'SELECT key_word ' \
               'FROM key_word_count_top100 ' \
               'ORDER BY total_count DESC'

        data2 = self.select_data(sql2)
        year_list = ['2016', '2017', '2018']
        key_word_list = [item[0] for item in data2]
        bar_with = 0.25

        x1 = [i for i in range(len(key_word_list[low:high]))]
        x2 = [i+bar_with for i in x1]
        x3 = [i+bar_with for i in x2]
        year1_data = [item[1] for item in self.select_data(sql1 % '2016')]
        year2_data = [item[1] for item in self.select_data(sql1 % '2017')]
        year3_data = [item[1] for item in self.select_data(sql1 % '2018')]
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.bar(x1, year1_data[low:high], width=bar_with, label='2016年')
        plt.bar(x2, year2_data[low:high], width=bar_with, label='2017年')
        plt.bar(x3, year3_data[low:high], width=bar_with, label='2018年')
        plt.axhline(y=0.75, ls='dotted', color='k')
        plt.xticks(x2, key_word_list[low:high], rotation=75)
        plt.legend()
        plt.ylabel('占比')
        plt.tight_layout()  # 调整整体空白
        plt.savefig(file_name + '.jpg', dpi=300, top=0.2)
        plt.show()


if __name__ == '__main__':
    A = Part1()
    # A.TOPSIS(A.data, 'top75perc_patents')
    # sql1 = 'SELECT current_assignee, count(patent_id) \
    #       'FROM patents_data_v2'
    #       'GROUP BY current_assignee'
    # sql2 = 'SELECT a.current_assignee, count(a.patent_id) ' \
    #      'FROM patents_data_v2 a ' \
    #      'INNER JOIN top75perc_patents b ' \
    #      'ON a.patent_id = b.patent_id ' \
    #      'GROUP BY a.current_assignee'
    # file_name1 = 'word_count.jpg'
    # file_name2 = 'word_count_high_quality.jpg'
    # A.paint_word_cloud(sql, file_name)

    # sql3 = 'SELECT filing_date FROM patents_data_v2 WHERE filing_date > "2014-01-01"'
    # sql4 = 'SELECT priority_date FROM patents_data_v2'
    # sql5 = 'SELECT publication_date FROM patents_data_v2'
    # sql6 = 'SELECT a.filing_date FROM patents_data_v2 a INNER JOIN top75perc_patents b ON a.patent_id = b.patent_id WHERE a.filing_date > "2014-01-01"'
    # sql7 = 'SELECT a.priority_date FROM patents_data_v2 a INNER JOIN top75perc_patents b ON a.patent_id = b.patent_id'
    # sql8 = 'SELECT a.publication_date FROM patents_data_v2 a INNER JOIN top75perc_patents b ON a.patent_id = b.patent_id'
    # title_name1 = '各月份专利数量（全量）_依据申请日期'
    # title_name2 = '各年份专利数量（全量）_依据优先权日期'
    # title_name3 = '各年份专利数量（全量）_依据公开日期'
    # title_name4 = '各月份专利数量（高质量）_依据申请日期'
    # title_name5 = '各年份专利数量（高质量）_依据优先权日期'
    # title_name6 = '各年份专利数量（高质量）_依据公开日期'
    # A.paint_patent_count(sql3, title_name1)
    # A.paint_patent_count(sql4, title_name2)
    # A.paint_patent_count(sql5, title_name3)
    # A.paint_patent_count(sql6, title_name4)
    # A.paint_patent_count(sql7, title_name5)
    # A.paint_patent_count(sql8, title_name6)

    # A.get_key_word()
    # A.get_CPC()
    # A.paint_CPC_count()
    # A.paint_CPC_count_by_year()
    # A.paint_word_count(0, 10, 'TOP1-10')
    # A.paint_word_count(10, 20, 'TOP11-20')
    # A.paint_word_count(20, 30, 'TOP21-30')
    # A.paint_word_count(30, 34, 'TOP30-34')
    # A.paint_high_quality_proportion(0, 10, 'TOP1-10_in3year')
    # A.paint_high_quality_proportion(10, 20, 'TOP11-20_in3year')
    # A.paint_high_quality_proportion(20, 30, 'TOP21-30_in3year')
    # A.paint_high_quality_proportion(30, 34, 'TOP31-34_in3year')


