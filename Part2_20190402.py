import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sqlalchemy import create_engine
from string import punctuation as punc
from sklearn.feature_extraction.text import TfidfVectorizer
from Thesis.Part1_20190321 import Part1


class Part2:
    def __init__(self):
        sql = 'SELECT a.patent_id, SUBSTR(a.filing_date, 1, 7), a.abstract ' \
               'FROM patents_data_v2 a ' \
               'INNER JOIN top75perc_patents b ' \
               'ON a.patent_id = b.patent_id'
        self.data = Part1().select_data(sql)
        self.stop_word = Part1().stop_word

    # 1.研究相关关键词的专利数量趋势走向
    # 各个专利中TF-IDF值较高的三个词作为关键词
    def get_TOP3_key_word(self):
        data = self.data
        result = []
        for item in data:  # 针对数据库中的每一行，分词，然后再组合成一句（sklearn需要）加入result中
            abstract = item[2]
            abstract = re.sub(r"[{}]+".format(punc), " ", abstract)  # 去标点符号
            abstract = re.sub("\d+", " ", abstract)
            abstract_list = [word.lower() for word in abstract.split() if word.lower() in abstract.split()]  # 分词
            clean_abstract_list = [word for word in abstract_list if word not in self.stop_word]  # 去停用词
            result.append(' '.join(clean_abstract_list))
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(result)
        word = vectorizer.get_feature_names()
        tf_idf = X.toarray()
        tf_idf_matrix = pd.DataFrame(tf_idf, columns=word)  # 形成dataframe，行为文档序号，列为词

        TOP3_word_result = []
        for i in tf_idf_matrix.index:
            article = tf_idf_matrix.loc[i]
            TOP3_word = [data[i][0], data[i][1]] + list(article.sort_values(ascending=False)[:3].index)  # 添加专利号、日期
            TOP3_word_result.append(TOP3_word)
        TOP3_word_df = pd.DataFrame(TOP3_word_result, columns=['patent_id', 'month', 'word1', 'word2', 'word3'])
        TOP3_word_df.to_csv('Part2/TOP3_word.csv', index=False)

    # 准备关键词共现分析的数据
    def get_co_occurrence_data_key_word(self, theshold=5):
        TOP3_word_data = pd.read_csv('Part2/TOP3_word.csv')
        del TOP3_word_data['patent_id']
        del TOP3_word_data['month']

        key_words = dict()
        co_occurrence = dict()
        for i in TOP3_word_data.index:
            line = list(TOP3_word_data.loc[i].values)
            for word in line:
                # 添加节点数据
                if key_words.get(word) is None:
                    key_words[word] = 1
                    co_occurrence[word] = dict()
                else:
                    key_words[word] = key_words[word] + 1
                # 添加边数据
                other_word = [item for item in line if item != word]
                for other in other_word:
                    # 如果两个节点已经存在，则忽略对应节点的边建立
                    if co_occurrence.get(other) is not None and co_occurrence[other].get(word) is not None:
                        pass
                    elif co_occurrence[word].get(other) is None:
                        co_occurrence[word][other] = 1
                    else:
                        co_occurrence[word][other] = co_occurrence[word][other] + 1

        # 输出节点、边结果，边设置阈值为5，节点依据边产生
        edge = []
        node = dict()
        for word in co_occurrence.keys():
            for key in co_occurrence[word].keys():
                # 添加超过5的边数据
                if co_occurrence[word][key] > theshold:
                    edge.append([word, key, co_occurrence[word][key]])
                    # 添加节点数据
                    if word not in node.keys():
                        node[word] = key_words[word]
                    if key not in node.keys():
                        node[key] = key_words[key]
        # 输出节点结果
        node = pd.DataFrame(node, index=['Weight']).T
        node['Id'] = node.index
        node['Label'] = node.index
        node = node[['Id', 'Label', 'Weight']]
        node.to_csv('Part2/node.csv', index=False)
        # 输出边结果
        edge = pd.DataFrame(edge, columns=['Source', 'Target', 'Weight'])
        edge.to_csv('Part2/edge.csv', index=False)

    # 2.研究CPC小类的专利数量趋势走向
    # 获取专利-月份-CPC小类列表
    def get_CPC_detail(self):
        sql = 'SELECT a.patent_id, SUBSTR(a.filing_date, 1, 7), a.CPC ' \
               'FROM patents_data_v2 a ' \
               'INNER JOIN top75perc_patents b ' \
               'ON a.patent_id = b.patent_id'
        data = Part1().select_data(sql)
        result = []
        for item in data:  # 针对数据库中的每一行，分词，加入total_abstract
            CPC = item[2]
            CPC_list = CPC.split(',')
            for CPC in CPC_list:
                result.append([item[0], item[1], CPC])
        result_df = pd.DataFrame(result, columns=['patent_id', 'month', 'CPC_detail'])
        # 将数据添加到数据库中
        engine = create_engine('mysql+pymysql://root:123456@localhost:3306/thesis?charset=utf8')
        conn = engine.connect()
        result_df.to_sql('CPC_code_count_by_month_detail', conn, index=False, if_exists='replace')
        conn.close()
        print('存储成功！')

    # 准备CPC共现分析的数据
    def get_co_occurrence_data_CPC(self, threshold=10):
        sql = 'SELECT CPC ' \
              'FROM patents_data_v2 a ' \
              'INNER JOIN top75perc_patents b ' \
              'ON a.patent_id = b.patent_id ' \
              'WHERE CPC <> ""'
        data = Part1().select_data(sql)
        cpc_dict = dict()
        cpc_co_occurrence = dict()
        for item in data:
            line = item[0].split(',')
            for cpc in line:
                # 添加节点数据
                if cpc_dict.get(cpc) is None:
                    cpc_dict[cpc] = 1
                    cpc_co_occurrence[cpc] = dict()
                else:
                    cpc_dict[cpc] = cpc_dict[cpc] + 1
                # 添加边数据
                other_cpc = [item for item in line if item != cpc]
                for other in other_cpc:
                    # 如果两个节点已经存在，则忽略对应节点的边建立
                    if cpc_co_occurrence.get(other) is not None and cpc_co_occurrence[other].get(cpc) is not None:
                        pass
                    elif cpc_co_occurrence[cpc].get(other) is None:
                        cpc_co_occurrence[cpc][other] = 1
                    else:
                        cpc_co_occurrence[cpc][other] = cpc_co_occurrence[cpc][other] + 1

        # 输出节点、边结果，边设置阈值为5，节点依据边产生
        edge = []
        node = dict()
        for cpc in cpc_co_occurrence.keys():
            for other in cpc_co_occurrence[cpc].keys():
                # 添加超过5的边数据
                if cpc_co_occurrence[cpc][other] > threshold:
                    edge.append([cpc, other, cpc_co_occurrence[cpc][other]])
                    # 添加节点数据
                    if cpc not in node.keys():
                        node[cpc] = cpc_dict[cpc]
                    if other not in node.keys():
                        node[other] = cpc_dict[other]
        # 输出节点结果
        node = pd.DataFrame(node, index=['Weight']).T
        node['Id'] = node.index
        node['Label'] = node.index
        node = node[['Id', 'Label', 'Weight']]
        node.to_csv('Part2/cpc_node.csv', index=False)
        # 输出边结果
        edge = pd.DataFrame(edge, columns=['Source', 'Target', 'Weight'])
        edge.to_csv('Part2/cpc_edge.csv', index=False)

    # 准备用于时间序列预测的热度数据
    def get_keyword_time_sequence(self):
        sql1 = 'SELECT month, SUM(kw_ratio) ' \
              'FROM(' \
              'SELECT a.patent_id, a.month, ' \
              'SUM(CASE c.community WHEN %s THEN a.num ELSE 0 END)/SUM(a.num) AS kw_ratio ' \
              'FROM key_word_count_by_month a ' \
              'INNER JOIN top75perc_patents b ' \
              'ON a.patent_id = b.patent_id ' \
              'INNER JOIN key_word_community c ' \
              'ON a.key_word = c.key_word ' \
              'GROUP BY a.patent_id, a.month ' \
              ') patent_idx ' \
              'GROUP BY month ' \
              'HAVING month BETWEEN "2016-01" AND "2018-06"'
        sql2 = 'SELECT COUNT(key_word) ' \
               'FROM key_word_community ' \
               'WHERE community = %s'
        community_sequence = dict()
        for i in range(1, 12):
            ci_data = Part1().select_data(sql1 % i)
            ci_keyword_count = Part1().select_data(sql2 % i)[0][0]
            month_sequence = dict()
            for item in ci_data:
                month_sequence[item[0]] = round(float(item[1])/ci_keyword_count, 6)
            community_sequence['社区'+str(i)] = month_sequence
        community_sequence_df = pd.DataFrame(community_sequence)
        community_sequence_df.to_csv('Part2/community_sequence.csv')

    # ARIMA模型的准备工作
    def ARIMA_pre(self):
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # 绘制自相关图和偏相关图
        from statsmodels.tsa.stattools import adfuller  # 单位根检验
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 定义使其正常显示中文字体黑体
        plt.rcParams['axes.unicode_minus'] = False

        data = pd.read_csv('Part2/community_sequence.csv', encoding='gbk', index_col=0)
        print(data.head(5))

        # 1.平稳性检验及差分阶数探索
        # 原始数据的自相关图
        adf_test_result = dict()
        for i in range(1, 12):
            community = '社区' + str(i)
            ci_data = data[community]
            plot_acf(ci_data)
            plt.title(community + '的自相关图')
            # plt.savefig('Part2/Autocorrelation/原始/自相关图' + community + '的自相关图')

            adf_result = adfuller(ci_data)
            critical_values = adf_result[4]
            adf_test_result[community] = [adf_result[i] for i in range(0, 4)] + [value for value in critical_values.values()] + [adf_result[5]]
        adf_test_result_df = pd.DataFrame(adf_test_result, index=['adf', 'p_value', 'usedlag', 'nobs', '1%', '5%', '10%', 'icbest']).T
        # adf_test_result_df.to_csv('Part2/Autocorrelation/原始/adf_test_result.csv')

        # 一阶差分
        D_adf_test_result = dict()
        for i in range(1, 12):
            community = '社区' + str(i)
            D_ci_data = data[community].diff().dropna()
            pd.DataFrame(D_ci_data).plot()
            # plt.savefig('Part2/Autocorrelation/一阶差分/' + community + '的一阶差分时序图')

            plot_acf(D_ci_data)
            plt.title(community + '的自相关图')
            # plt.savefig('Part2/Autocorrelation/一阶差分/自相关图/' + community + '的一阶差分自相关图')

            plot_pacf(D_ci_data)
            plt.title(community + '的偏相关图')
            # plt.savefig('Part2/Autocorrelation/一阶差分/偏相关图/' + community + '的一阶差分偏相关图')

            D_adf_result = adfuller(D_ci_data)
            print(community + '的检验结果为：', D_adf_result)
            D_critical_values = D_adf_result[4]
            D_adf_test_result[community] = [D_adf_result[i] for i in range(0, 4)] + [value for value in D_critical_values.values()] + [D_adf_result[5]]
        D_adf_test_result_df = pd.DataFrame(D_adf_test_result, index=['adf', 'p_value', 'usedlag', 'nobs', '1%', '5%', '10%', 'icbest']).T
        # D_adf_test_result_df.to_csv('Part2/Autocorrelation/一阶差分/D_adf_test_result.csv')

        # 二阶差分
        D2_adf_test_result = dict()
        for i in range(1, 12):
            community = '社区' + str(i)
            D2_ci_data = data[community].diff(periods=2).dropna()
            pd.DataFrame(D2_ci_data).plot()
            # plt.savefig('Part2/Autocorrelation/二阶差分/' + community + '的二阶差分时序图')

            plot_acf(D2_ci_data)
            plt.title(community + '的自相关图')
            # plt.savefig('Part2/Autocorrelation/二阶差分/自相关图/' + community + '的二阶差分自相关图')

            plot_pacf(D2_ci_data)
            plt.title(community + '的偏相关图')
            # plt.savefig('Part2/Autocorrelation/二阶差分/偏相关图/' + community + '的二阶差分偏相关图')

            D2_adf_result = adfuller(D2_ci_data)
            print(community + '的检验结果为：', D2_adf_result)
            D2_critical_values = D2_adf_result[4]
            D2_adf_test_result[community] = [D2_adf_result[i] for i in range(0, 4)] + [value for value in D2_critical_values.values()] + [D2_adf_result[5]]
        D2_adf_test_result_df = pd.DataFrame(D2_adf_test_result, index=['adf', 'p_value', 'usedlag', 'nobs', '1%', '5%', '10%', 'icbest']).T
        # D2_adf_test_result_df.to_csv('Part2/Autocorrelation/二阶差分/D2_adf_test_result.csv')

        # 2.白噪声检验
        from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
        acorrljungbox_test_result = dict()
        for i in range(1, 12):
            community = '社区' + str(i)
            if i == 5:
                Dx_ci_data = data[community].diff(periods=2).dropna()
            else:
                Dx_ci_data = data[community].diff().dropna()
            result = acorr_ljungbox(Dx_ci_data, lags=2)
            acorrljungbox_test_result[community] = [result[0][0], result[1][0], result[0][1], result[1][1]]
        acorrljungbox_test_result_df = pd.DataFrame(acorrljungbox_test_result, index=['statistics_lags1', 'p_value_lags1', 'statistics_lags2', 'p_value_lags2']).T
        acorrljungbox_test_result_df.to_csv('Part2/acorrljunbox_test.csv')  # 滞后2期，p值均小于0.05，说明序列差分序列不是白噪声

    # ARIMA模型参数确定及评估预测
    def ARIMA(self):
        from statsmodels.tsa.arima_model import ARIMA
        data = pd.read_csv('Part2/community_sequence.csv', encoding='gbk', index_col=0)
        length = len(data)

        ARIMA_model_parameter = dict()
        for i in range(1, 12):
            community = '社区' + str(i)
            # 1.划分训练集、测试集，最后的三个月为测试集
            ci_data = data[community]
            ci_train_data = ci_data[:length-3]
            ci_test_data = ci_data[length-3:]
            # 社区5二阶差分，其它社区一阶差分
            if i == 5:
                d = 2
            else:
                d = 1
            # 2.依据BIC选择最优的AR阶数p和MR阶数q
            bic_matrix = []
            for p in range(5):
                q_bic_matrix = []
                for q in range(5):
                    try:
                        q_bic_matrix.append(ARIMA(ci_train_data, (p, d, q)).fit().bic)
                    except:
                        q_bic_matrix.append(None)
                bic_matrix.append(q_bic_matrix)
            bic_matrix = pd.DataFrame(bic_matrix)
            p, q = bic_matrix.stack().idxmin()
            community_result = {'p': p, 'd': d, 'q': q}
            ARIMA_model_parameter[community] = community_result
            print('BIC最小的p的值：%s，q的值：%s' % (p, q))
            model = ARIMA(ci_train_data, (p, d, q)).fit()
            # 3.静态预测，17年1月-18年3月部分是用样本值进行预测, 18年4月-18年6月部分是用前一期拟合值进行预测，属于外推
            sample_pre_sequence = model.predict('2017-01', '2018-06', typ='levels', dynamic=False)
            sample_pre_sequence.index = [time.strftime('%Y-%m') for time in sample_pre_sequence.index]
            model = ARIMA(ci_data, (p, d, q)).fit()
            # 4.未来预测，运用所有训练数据对7月-9月进行预测
            future_pre_sequence = model.forecast(3)[0]
            future_pre_sequence_df = pd.DataFrame(future_pre_sequence, index=['2018-07', '2018-08', '2018-09'])
            # 5.将静态预测和未来预测拼接在一起
            sample_pre_sequence = pd.concat([sample_pre_sequence, future_pre_sequence_df], axis=0)
            ci_data = pd.DataFrame(ci_data)
            ci_data.columns = [community + '的原始数据']
            sample_pre_sequence = pd.DataFrame(sample_pre_sequence)
            sample_pre_sequence.columns = [community + '的预测数据']
            ci_real_pre_data = pd.merge(ci_data, sample_pre_sequence, left_index=True, right_index=True, how='outer')
            if i == 1:
                ARIMA_real_pre_result_df = ci_real_pre_data
            else:
                ARIMA_real_pre_result_df = pd.merge(ARIMA_real_pre_result_df, ci_real_pre_data
                                                    , left_index=True, right_index=True)
        ARIMA_model_parameter_df = pd.DataFrame(ARIMA_model_parameter)
        ARIMA_model_parameter_df.to_csv('Part2/ARIMA_model_parameter.csv')
        ARIMA_real_pre_result_df.to_csv('Part2/ARIMA_real_pre_result.csv')

    # 获得各社区各月份的专利质量指标，因为专利质量指标无法应用在时间序列预测上，因此此部分数据暂时无用
    def get_pantent_quality_time_sequence(self):
        sql1 = 'SELECT month, community, SUM(kw_ratio), AVG(kw_ratio*family_num), AVG(kw_ratio*patent_citations) ' \
            ', AVG(kw_ratio*non_patent_citations), AVG(kw_ratio*claim_num), AVG(kw_ratio*description_num) ' \
            ', AVG(kw_ratio*CPC_num), AVG(kw_ratio*cited_by_idx), AVG(kw_ratio*(365/date_age)) AS novelty ' \
            'FROM ' \
            '( ' \
            'SELECT a.patent_id, a.month, %s AS community ' \
            ', SUM(CASE c.community WHEN %s THEN a.num ELSE 0 END)/SUM(a.num) AS kw_ratio ' \
            'FROM key_word_count_by_month a ' \
            'INNER JOIN top75perc_patents b ' \
            'ON a.patent_id = b.patent_id ' \
            'INNER JOIN key_word_community c ' \
            'ON a.key_word = c.key_word ' \
            'GROUP BY a.patent_id, a.month ' \
            ') patent_ratio ' \
            'INNER JOIN  ' \
            '(SELECT patent_id, family_num, patent_citations, non_patent_citations, claim_num' \
            ', description_num, CPC_num , cited_by*365/DATEDIFF("2019-03-08",filing_date) AS cited_by_idx ' \
            ', DATEDIFF("2019-03-08",filing_date) AS date_age ' \
            'FROM patents_data_v2 ' \
            ') as patent_quality ' \
            'ON patent_ratio.patent_id = patent_quality.patent_id ' \
            'GROUP BY patent_ratio.month ' \
            'HAVING month BETWEEN "2015-10" AND "2018-06"'
        sql2 = 'SELECT COUNT(key_word) ' \
               'FROM key_word_community ' \
               'WHERE community = %s'
        cols = ['month', 'community', 'kw_ratio', 'family_num', 'patent_num', 'non_patent_num', 'claim_num', 'description_num',
               'CPC_num', 'cited_by', 'novelty']
        comunity_quality_sequence_data = []
        for i in range(1, 12):
            pqi_data = Part1().select_data(sql1 % (i, i))
            ci_keyword_count = Part1().select_data(sql2 % i)[0][0]
            for item in pqi_data:
                comunity_quality_sequence_data.append([item[0], item[1]] + [round(float(item[i])/ci_keyword_count, 6) for i in range(2, 11)])
        ci_quality_sequence_df = pd.DataFrame(comunity_quality_sequence_data, columns=cols)
        ci_quality_sequence_df.to_csv('Part2/community_quality_sequence.csv', index=False)

    # 准备BP神经网络的数据，还是以社区为单位
    # 横向为1个社区类别，3个时间序列相关：M-3，M-2，M-1
    # 纵向为时间
    def BP_neural_network_pre(self, pre_month_num):
        from dateutil.relativedelta import relativedelta
        data = pd.read_csv('Part2/community_sequence_Vtotal.csv', encoding='gbk')
        data.columns = ['month'] + list(data.columns)[1:]
        data['month'] = data.loc[:, 'month'].apply(lambda x: datetime.strptime(x, '%y-%b').strftime('%Y-%m'))
        time = datetime.strptime('2016-01', '%Y-%m') + relativedelta(months=-pre_month_num)
        data = data[data['month'] >= time.strftime('%Y-%m')]
        data = data.reset_index(drop=True)
        resort_data = []
        for i in range(1, 12):
            ci_data = data[['month', str(i)]]
            for j in range(pre_month_num, len(ci_data)):
                resort_data.append([ci_data['month'].iloc[j]] + [i] + [ci_data[str(i)].iloc[j-x] for x in range(pre_month_num, -1, -1)])
        cols = ['month', 'community'] + ['M_pre' + str(i) for i in range(pre_month_num, 0, -1)] + ['kw_ratio']
        resort_data_df = pd.DataFrame(resort_data, columns=cols)
        resort_data_df.to_csv('Part2/BP_neural_network_data.csv', index=False)

    # 构建BP神经网络
    def BP_neural_network(self, pre_month_num):
        import numpy as np
        from sklearn import preprocessing
        from sklearn.model_selection import GridSearchCV
        from sklearn.neural_network import MLPRegressor
        data = pd.read_csv('Part2/BP_neural_network_data.csv')
        # 1.划分训练集、测试集，最后三个月为测试集
        x, y = data.loc[:, data.columns[:-1]], data.loc[:, ['month', 'kw_ratio']]
        train_x, train_y = x[x['month'] <= '2018-03'][x.columns[1:]], y[y['month'] <= '2018-03'][y.columns[1]]
        # 2.归一化
        x_min_max_scaler = preprocessing.MinMaxScaler()
        train_x = x_min_max_scaler.fit_transform(train_x)
        y_min_max_scaler = preprocessing.StandardScaler()
        train_y = y_min_max_scaler.fit_transform(train_y.values.reshape(-1, 1))
        # 3.交叉验证，确定隐层神经元数量
        parameters = {'hidden_layer_sizes': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        grid_search = GridSearchCV(MLPRegressor(activation='logistic', solver='sgd', alpha=1e-5, random_state=123
                                                , max_iter=200, learning_rate_init=0.05), parameters
                                                , cv=10, scoring='neg_mean_squared_error')
        grid_search.fit(train_x, train_y.ravel())
        # grid_search.fit(train_x, train_y)
        # 保存参数
        param = grid_search.best_params_
        param['best_score'] = 0 - grid_search.best_score_  # 因为原计算结果是均方误差取负，因此要求其负数作为均方误差值
        param_df = pd.DataFrame(param, index=[0])
        param_df.to_csv('Part2/BP_model_parameter.csv', index=False)
        # 4.静态预测，17年1月-18年3月部分是用样本值进行预测, 18年4月-18年6月部分是用前一期预测值进行预测，属于外推
        for i in range(1, 12):
            # a.先预测17年1月到18年3月部分
            in_sample_input = x[(x['month'] >= '2017-01') & (x['month'] <= '2018-03') & (x['community'] == i)][x.columns[1:]]
            in_sample_input = x_min_max_scaler.transform(in_sample_input)
            in_sample_pre = grid_search.predict(in_sample_input)
            in_sample_pre = [item[0] for item in y_min_max_scaler.inverse_transform(in_sample_pre.reshape(-1, 1))]
            # in_sample_pre = in_sample_pre.tolist()
            # b.预测18年4月-18年6月部分
            ci_mix_data = data[(data['community'] == i) & (data['month'] <= '2018-03')]
            ci_mix_data = ci_mix_data.reset_index(drop=True)
            month = ['2018-04', '2018-05', '2018-06']
            # 最后三个月的信息，动态地根据每个要预测月份的前n个月进行预测
            for j in range(3):
                input = [i] + [ci_mix_data.loc[row]['kw_ratio'] for row in range(len(ci_mix_data)-pre_month_num, len(ci_mix_data))]
                # 注意测试数据也要进行和训练数据相同的最大最小归一化之后才可能用于预测
                min_max_input = x_min_max_scaler.transform([input])
                pre = grid_search.predict(min_max_input)
                pre = [item[0] for item in y_min_max_scaler.inverse_transform(pre.reshape(-1, 1))]
                # pre = pre.tolist()
                ci_mix_data.loc[len(ci_mix_data)] = [month[j]] + input + pre
            ci_pre_test_data = pd.DataFrame(in_sample_pre + list(ci_mix_data[ci_mix_data['month'] >= '2018-04']['kw_ratio'])
                                             , index=list(x[(x['month'] >= '2017-01') & (x['community'] == i)]['month'])
                                             , columns=['社区' + str(i) + '的预测数据'])
            if i == 1:
                pre_test_data = ci_pre_test_data
            else:
                pre_test_data = pd.merge(pre_test_data, ci_pre_test_data, left_index=True, right_index=True)
        # pre_test_data.to_csv('Part2/BP_real_pre_result_201701_201806.csv')
        # 5.未来预测，运用所有训练数据对7月-9月进行预测
        # 运用所有数据
        train_X, train_Y = x[x.columns[1:]], y[y.columns[1]]
        # 归一化
        X_min_max_scaler = preprocessing.MinMaxScaler()
        train_X = X_min_max_scaler.fit_transform(train_X)
        Y_min_max_scaler = preprocessing.StandardScaler()
        train_Y = Y_min_max_scaler.fit_transform(train_Y.values.reshape(-1, 1))
        # 用最优参数构建模型，用所有数据训练
        mlpr = MLPRegressor(hidden_layer_sizes=param['hidden_layer_sizes'], activation='logistic', solver='sgd'
                            , alpha=1e-5, random_state=123, max_iter=200, learning_rate_init=0.05)
        mlpr.fit(train_X, train_Y.ravel())
        # mlpr.fit(train_X, train_Y)
        # 准备18年7月-18年9月的数据
        for i in range(1, 12):
            ci_mix_data = data[data['community'] == i]
            ci_mix_data = ci_mix_data.reset_index(drop=True)
            month = ['2018-07', '2018-08', '2018-09']
            # 最后三个月的信息，动态地根据每个要预测月份的前n个月进行预测
            for j in range(3):
                input = [i] + [ci_mix_data.loc[row]['kw_ratio'] for row in range(len(ci_mix_data)-pre_month_num, len(ci_mix_data))]
                # 注意测试数据也要进行和训练数据相同的最大最小归一化之后才可能用于预测
                min_max_input = X_min_max_scaler.transform([input])
                pre = mlpr.predict(min_max_input)
                pre = [item[0] for item in Y_min_max_scaler.inverse_transform(pre.reshape(-1, 1))]
                # pre = pre.tolist()
                ci_mix_data.loc[len(ci_mix_data)] = [month[j]] + input + pre
            ci_pre_future_data = pd.DataFrame(list(ci_mix_data[ci_mix_data['month'] >= '2018-07']['kw_ratio'])
                                              , index=month
                                              , columns=['社区' + str(i) + '的预测数据'])
            if i == 1:
                pre_future_data = ci_pre_future_data
            else:
                pre_future_data = pd.merge(pre_future_data, ci_pre_future_data, left_index=True, right_index=True)
        # pre_future_data.to_csv('Part2/BP_real_pre_result_201807_201809.csv')
        pre_data = pd.concat([pre_test_data, pre_future_data], axis=0)
        pre_data.to_csv('Part2/BP_real_pre_result.csv')

    # 拼接ARIMA和BP的数据，也作为组合模型的准备数据
    def composite_model_pre(self):
        ARIMA_pre_data = pd.read_csv('Part2/ARIMA_real_pre_result.csv', encoding='gbk', index_col=0)
        BP_pre_data = pd.read_csv('Part2/BP_real_pre_result.csv', encoding='gbk', index_col=0)
        for i in range(1, 12):
            community = '社区' + str(i)
            ci_region_ARIMA_pre_data = ARIMA_pre_data.loc[:, [community + '的原始数据', community + '的预测数据']]
            ci_BP_pre_data = BP_pre_data.loc[:, [community + '的预测数据']]
            ci_result_data = pd.merge(ci_region_ARIMA_pre_data, ci_BP_pre_data
                                      , left_index=True, right_index=True, how='outer')
            ci_result_data.columns = [community + '的原始数据', community + '的ARIMA预测数据', community + '的BP神经网络预测数据']
            if i == 1:
                result_data = ci_result_data
            else:
                result_data = pd.merge(result_data, ci_result_data, left_index=True, right_index=True, how='outer')
        result_data.to_csv('Part2/region_ARIMA_BP_pre_result.csv')

    # 组合模型计算
    def compositing_model(self):
        data = pd.read_csv('Part2/region_ARIMA_BP_pre_result.csv', encoding='gbk')
        data.columns = ['month'] + list(data.columns[1:])
        data.index = data['month']
        pre_data = data[(data['month'] >= '2017-01') & (data['month'] <= '2018-03')]
        del pre_data['month']
        result_weight = {}
        for i in range(1, 12):
            community = '社区' + str(i)
            # 计算各模型权重
            ci_region = pre_data[community + '的原始数据']
            ci_ARIMA = pre_data[community + '的ARIMA预测数据']
            ci_BP = pre_data[community + '的BP神经网络预测数据']
            E_11 = sum((ci_region - ci_ARIMA)**2)
            E_22 = sum((ci_region - ci_BP)**2)
            E_12_21 = sum((ci_region - ci_ARIMA)*(ci_region - ci_BP))
            w_1 = (E_22 - E_12_21)/(E_11 + E_22 - 2*E_12_21)
            w_2 = (E_11 - E_12_21)/(E_11 + E_22 - 2*E_12_21)
            if (w_1 < 0) and (w_2 > 0):
                w_1, w_2 = 0, 1
            elif (w_1 > 0) and (w_2 < 0):
                w_1, w_2 = 1, 0
            else:
                pass
            result_weight[community] = {'w1': w_1, 'w2': w_2}
            # 计算组合结果
            ci_ARIMA = data[community + '的ARIMA预测数据']
            ci_BP = data[community + '的BP神经网络预测数据']
            ci_compositing_result = w_1*ci_ARIMA + w_2*ci_BP
            ci_compositing_result = pd.DataFrame(ci_compositing_result, columns=[community + '的组合预测数据'])
            ci_region_ARIMA_BP_compositing_result = pd.merge(data[[community + '的原始数据', community + '的ARIMA预测数据', community + '的BP神经网络预测数据']],
                                                             ci_compositing_result, right_index=True, left_index=True, how='outer')
            if i == 1:
                region_ARIMA_BP_compositing_result = ci_region_ARIMA_BP_compositing_result
            else:
                region_ARIMA_BP_compositing_result = pd.merge(region_ARIMA_BP_compositing_result, ci_region_ARIMA_BP_compositing_result,
                                                              right_index=True, left_index=True, how='outer')
        result_weight_df = pd.DataFrame(result_weight)
        result_weight_df.to_csv('Part2/model_weight.csv')
        region_ARIMA_BP_compositing_result.to_csv('Part2/region_ARIMA_BP_compositing_pre_result.csv')

    # 计算两模型及其组合模型的结果进行对比
    def compare_ARIMA_BP_compositing(self):
        from sklearn.metrics import mean_squared_error
        data = pd.read_csv('Part2/region_ARIMA_BP_compositing_pre_result.csv', encoding='gbk')
        data.columns = ['month'] + list(data.columns[1:])
        data = data[(data['month'] >= '2018-03') & (data['month'] <= '2018-06')]
        data.index = data['month']
        del data['month']
        access_result = dict()
        for i in range(1, 12):
            community = '社区' + str(i)
            ci_ARIMA_MSE = mean_squared_error(data[community + '的原始数据'], data[community + '的ARIMA预测数据'])
            ci_BP_MSE = mean_squared_error(data[community + '的原始数据'], data[community + '的BP神经网络预测数据'])
            ci_compositing_MSE = mean_squared_error(data[community + '的原始数据'], data[community + '的组合预测数据'])
            access_result[community] = {'ARIMA': ci_ARIMA_MSE, 'BP': ci_BP_MSE, '组合预测': ci_compositing_MSE}
        access_result_df = pd.DataFrame(access_result)
        access_result_df.to_csv('Part2/access_result.csv')

    # 画图
    def paint_pre_result(self):
        data = pd.read_csv('Part2/region_ARIMA_BP_compositing_pre_result.csv', encoding='gbk')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        for i in range(1, 12):
            fig = plt.figure(i)
            community = '社区' + str(i)
            # 绘制【训练集】预测数据
            ci_region = data[data['month'] <= '2018-03'][community + '的原始数据']
            ci_ARIMA = data[data['month'] <= '2018-03'][community + '的ARIMA预测数据']
            ci_BP = data[data['month'] <= '2018-03'][community + '的BP神经网络预测数据']
            ci_compositing = data[data['month'] <= '2018-03'][community + '的组合预测数据']
            ci_region.plot(color='grey', label='观察值')
            ci_compositing.plot(color='red', label='组合模型-预测值')
            ci_ARIMA.plot(color='blue', linestyle='--', linewidth=1,  label='ARIMA模型-预测值')
            ci_BP.plot(color='green', linestyle='--', linewidth=1,  label='BP神经网络-预测值')
            # 绘制【测试集】预测数据
            ci_region = data[(data['month'] >= '2018-04') & (data['month'] <= '2018-06')][community + '的原始数据']
            ci_ARIMA = data[(data['month'] >= '2018-04') & (data['month'] <= '2018-06')][community + '的ARIMA预测数据']
            ci_BP = data[(data['month'] >= '2018-04') & (data['month'] <= '2018-06')][community + '的BP神经网络预测数据']
            ci_compositing = data[(data['month'] >= '2018-04') & (data['month'] <= '2018-06')][community + '的组合预测数据']
            ci_region.plot(color='grey', label="")
            ci_compositing.plot(color='red', label="")
            ci_ARIMA.plot(color='blue', linestyle='--', linewidth=1, label="")
            ci_BP.plot(color='green', linestyle='--', linewidth=1, label="")
            # 绘制【外推】预测数据
            # 绘制测试集预测数据
            ci_ARIMA = data[(data['month'] >= '2018-07') & (data['month'] <= '2018-09')][community + '的ARIMA预测数据']
            ci_BP = data[(data['month'] >= '2018-07') & (data['month'] <= '2018-09')][community + '的BP神经网络预测数据']
            ci_compositing = data[(data['month'] >= '2018-07') & (data['month'] <= '2018-09')][community + '的组合预测数据']
            ci_region.plot(color='grey', label="")
            ci_compositing.plot(color='red', label="")
            ci_ARIMA.plot(color='blue', linestyle='--', linewidth=1, label="")
            ci_BP.plot(color='green', linestyle='--', linewidth=1, label="")
            # 垂直线设置
            plt.axvline(len(data)-6, color='black', linewidth=1)
            plt.axvline(len(data)-3, color='black', linewidth=1)
            # 区域说明
            plt.figtext(0.3, 0.5, r'$①$', fontdict={'size': '20', 'color': 'black'})
            plt.figtext(0.845, 0.5, r'$②$', fontdict={'size': '20', 'color': 'black'})
            plt.figtext(0.91, 0.5, r'$③$', fontdict={'size': '20', 'color': 'black'})
            # 图表设置
            plt.xticks(range(len(data)), list(data['month']), rotation=75)
            plt.xlabel('月份')
            plt.ylabel('热度指数')
            plt.legend()
            plt.title(community + '的热度图')
            fig.tight_layout()
            plt.savefig('Part2/Community_pre_picture/' + community + '的热度预测图', dpi=300)
        plt.show()

    # 绘制总体图
    def region_pre_result(self):
        data = pd.read_csv('Part2/region_ARIMA_BP_compositing_pre_result.csv', encoding='gbk')
        for i in range(1, 12):
            community = '社区' + str(i)
            ci_region = data[data['month'] <= '2018-06'][community + '的原始数据']
            ci_pre = data[(data['month'] >= '2018-07') & (data['month'] <= '2018-09')][community + '的组合预测数据']
            ci_result = pd.DataFrame(pd.concat([ci_region, ci_pre], axis=0), columns=[community])
            if i == 1:
                result_df = ci_result
            else:
                result_df = pd.merge(result_df, ci_result, left_index=True, right_index=True)
        result_df.index = data['month'].tolist()
        result_df.to_csv('Part2/region_compositing_pre_result.csv')

if __name__ == '__main__':
    B = Part2()
    # B.ARIMA()
    B.BP_neural_network_pre(6)
    B.BP_neural_network(6)
    B.composite_model_pre()
    B.compositing_model()
    B.compare_ARIMA_BP_compositing()
    B.region_pre_result()
