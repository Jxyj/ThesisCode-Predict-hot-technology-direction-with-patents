from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import selenium
import re, random
import time, datetime
import pymysql
import os, logging


class MainScrapy(object):
    def __init__(self):
        # 一些参数的设置
        self.url = 'https://patents.google.com/'
        self.search_word = '(5G network)(NR)(new radio)(-compounds)(-tire)(-copolymer) before:filing:%s after:filing:%s'
        self.date_seg = 120

    # 获取所有专利
    def scrapy(self, logger):
        try:
            date_seg = self.date_seg
            table_name = 'patents_data'
            Is_exits_table = self.table_exists(table_name)
            if Is_exits_table == 0:
                self.createsql(table_name)  # 创建数据表
                patents_count = 0
                after_time = datetime.datetime(2000, 1, 1)
            else:
                patents_count, after_time = self.get_db_last_patent(table_name)
                if patents_count == 0:
                    after_time = datetime.datetime(2000, 1, 1)
                else:
                    after_time = datetime.datetime.strptime(after_time, '%Y-%m-%d')
            before_time = after_time + datetime.timedelta(days=date_seg)
            while after_time < datetime.datetime.now():
                str_after_time = datetime.datetime.strftime(after_time, '%Y%m%d')
                str_before_time = datetime.datetime.strftime(before_time, '%Y%m%d')
                search_word = self.search_word % (str_before_time, str_after_time)  # 将时间区间放入进行限制
                driver = self.search_patent(search_word)  # 输入关键词进行搜索
                self.change_sort_by_oldest(driver)   # 改变时间限制的对象为国际申请时间
                count = self.get_result_count(driver)  # 搜索获得的专利数量
                time.sleep(random.randint(3,5))
                if count == 0:   # 如果专利数量为0，区间向后推移
                    message = '%s至%s期间无专利信息' % (str_after_time, str_before_time)
                    logger.info(message)
                    date_seg = self.date_seg
                    after_time = before_time
                elif count > 300:   # 如果专利数量大于300，缩小区间范围
                    date_seg = int(date_seg/2)
                else:   # 专利数量在0-300之间可以进行获取
                    WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.TAG_NAME, 'search-result-item')))
                    tag = driver.find_element_by_tag_name('search-result-item')
                    tag.find_element_by_tag_name('a').click()  # 获取首个专利的链接，并点击
                    current, total = [int(i) for i in re.findall('\d+', driver.find_element_by_id('current').text)]  # 获取当前专利的序号和专利总数量
                    while current <= total:
                        result_data = self.get_result_data(driver)
                        self.insertsql(table_name, patents_count, result_data)
                        patents_count += 1
                        message = '【第%s个专利】%s至%s期间共%s个专利，No:%s，专利号:%s，名称：%s' % (patents_count, str_after_time, str_before_time, total, current, result_data[1], result_data[2])
                        logger.info(message)
                        time.sleep(patents_count % 3 + current % 4)
                        if current != total:
                            driver = self.turn_page(driver, current)
                        current += 1
                    date_seg = self.date_seg
                    after_time = before_time
                before_time = after_time + datetime.timedelta(days=date_seg)
                driver.quit()
            print('抓取完成，共抓取%s条记录' % patents_count)
        except Exception as e:
            print(e)
            logger.debug(e)
            driver.quit()
            time.sleep(60)
            self.scrapy(logger)

    # 获取某一时间段内的专利信息
    def scrapy_special_time(self, logger, start_time, end_time):
        datetype_start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d')
        datetype_end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d')
        try:
            date_seg = 4
            table_name = 'patents_data'
            patents_count, after_time = self.get_db_max_id_patent(table_name)
            after_time = datetype_start_time
            #if after_time < start_time:
            #    after_time = datetype_start_time
            #else:
            #    after_time = datetime.datetime.strptime(after_time, '%Y-%m-%d')
            before_time = after_time + datetime.timedelta(days=date_seg)
            while after_time < datetype_end_time:
                str_after_time = datetime.datetime.strftime(after_time, '%Y-%m-%d')
                str_before_time = datetime.datetime.strftime(before_time, '%Y-%m-%d')
                search_word = self.search_word % (str_before_time, str_after_time)  # 将时间区间放入进行限制
                driver = self.search_patent(search_word)  # 输入关键词进行搜索
                self.change_sort_by_oldest(driver)   # 改变时间限制的对象为国际申请时间
                count = self.get_result_count(driver)  # 搜索获得的专利数量
                time.sleep(random.randint(3,5))
                if count == 0:   # 如果专利数量为0，区间向后推移
                    message = '%s至%s期间无专利信息' % (str_after_time, str_before_time)
                    logger.info(message)
                    date_seg = 4
                    after_time = before_time
                elif count > 300:   # 如果专利数量大于300，缩小区间范围
                    date_seg = int(date_seg/2)
                else:   # 专利数量在0-300之间可以进行获取
                    WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.TAG_NAME, 'search-result-item')))
                    tag = driver.find_element_by_tag_name('search-result-item')
                    tag.find_element_by_tag_name('a').click()  # 获取首个专利的链接，并点击
                    current, total = [int(i) for i in re.findall('\d+', driver.find_element_by_id('current').text)]  # 获取当前专利的序号和专利总数量
                    while current <= total:
                        result_data = self.get_result_data(driver)
                        self.insertsql(table_name, patents_count, result_data)
                        patents_count += 1
                        message = '【第%s个专利】%s至%s期间共%s个专利，No:%s，专利号:%s，名称：%s' % (patents_count, str_after_time, str_before_time, total, current, result_data[1], result_data[2])
                        logger.info(message)
                        time.sleep(patents_count % 3 + current % 4)
                        if current != total:
                            driver = self.turn_page(driver, current)
                        current += 1
                    date_seg = 4
                    after_time = before_time
                before_time = after_time + datetime.timedelta(days=date_seg)
                driver.quit()
        except Exception as e:
            print(e)
            logger.debug(e)
            driver.quit()
            time.sleep(60)
            start_time = self.get_db_max_id_patent(table_name)[1]
            print(start_time)
            self.scrapy_special_time(logger, start_time, end_time)
        print('抓取完成，共抓取%s条记录' % patents_count)

    # 设置日志
    def logger_setter(self):
        local_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
        # 1.创建一个logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关
        # 2.创建一个日志handler、一个控制台handler，用于写入日志文件、输出到控制台
        log_path = os.path.dirname(os.getcwd()) + '/' + local_time + '.log'
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.INFO)  # 输出到file的log等级的开关
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # 3.定义日志handler、控制台handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # 4.将日志handler添加到logger里面
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    # 搜索操作
    def search_patent(self, search_word):
        driver = webdriver.Chrome()
        driver.maximize_window()
        driver.get(self.url)
        inputElement = driver.find_element_by_name('q')
        inputElement.send_keys(search_word)
        driver.find_element_by_id('searchButton').click()
        return driver

    # 获得搜索结果的数量
    def get_result_count(self, driver):
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.TAG_NAME, 'search-result-item')))
        item = driver.find_element_by_id('count')
        count = re.search('about\s+(.+)\s+results', item.get_attribute('textContent')).group(1)
        count = int(count.replace(",", ""))
        return count

    # 将搜索结果页设置为时间范围的限制目标为国际申请时间，按时间由旧到新排序，并且每页显示100个结果
    def change_sort_by_oldest(self, driver):
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.XPATH, '//span[contains(text(), "Priority")]')))
        sort_item = driver.find_element_by_xpath('//span[contains(text(), "Date")]')
        sort_item.click()
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.XPATH, '//div[contains(text(), "Filing")]')))
        sort_select = driver.find_element_by_xpath('//div[contains(text(), "Filing")]')
        time.sleep(2)
        sort_select.click()
        time.sleep(3)
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.XPATH, '//span[contains(text(), "Relevance")]')))
        sort_item = driver.find_element_by_xpath('//span[contains(text(), "Sort by")]')
        sort_item.click()
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.XPATH, '//div[contains(text(), "Oldest")]')))
        sort_select = driver.find_element_by_xpath('//div[contains(text(), "Oldest")]')
        time.sleep(2)
        sort_select.click()
        time.sleep(3)
        pages_item = driver.find_element_by_xpath('//span[contains(text(), "Results / page")]')
        pages_item.click()
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.XPATH, '//div[contains(text(), "100")]')))
        pages_select = driver.find_element_by_xpath('//div[contains(text(), "100")]')
        time.sleep(2)
        pages_select.click()
        time.sleep(3)

    # 翻页
    def turn_page(self, driver, current):
        if int(current) % 100 == 0:
            time.sleep(random.randint(50, 60))
        time.sleep(random.randint(2,4))
        WebDriverWait(driver, 120, 1).until(EC.element_to_be_clickable((By.ID, "nextResult")))
        next_result = driver.find_element_by_id('nextResult').find_element_by_tag_name('a')
        next_result.click()
        return driver

    # 获取页面信息
    def get_result_data(self, driver):
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[class = 'flex-2 style-scope patent-result']")))
        self.get_english_version(driver)  # 转换为英文版本

        # 1.获取大部分易获取信息
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "[class = 'flex-2 style-scope patent-result']")))
        title = driver.find_element_by_id('title').text  # 获得专利名称
        pubnum = driver.find_element_by_id('pubnum').text  # 获得专利号
        current_link = driver.current_url  # 获取当前页面的链接
        family = list(set([item.text for item in driver.find_elements_by_id('cc')]))  # 获取专利族列表
        family_list = ','.join(family)  # 获取专利族
        family_num = len(family)  # 获取专利族数量
        num = re.compile('\d+')
        patent_citations = self.get_element_num(driver, num, '//a[contains(text(), "Patent citations")]')  # 获取专利引用数量
        non_patent_citations = self.get_element_num(driver, num, '//a[contains(text(), "Non-patent citations")]') # 获取非专利引用数量
        cited_by = self.get_element_num(driver, num, '//a[contains(text(), "Cited by")]')  # 获取被引数量
        final_classification = [item.find_elements_by_css_selector("[class = 'style-scope classification-tree']")[-3] for item in driver.find_elements_by_tag_name('classification-tree')]  # 获取存有最终CPC号的节点
        final_classification = [item.get_attribute('textContent').strip().split('\n')[0] for item in final_classification]  # 获取CPC号列表
        final_classification_list = ','.join(final_classification)  # 获取CPC号
        final_classification_num = len(final_classification)  # 获取CPC号数量
        filing_time = driver.find_element_by_xpath("//div[@class = 'filed style-scope application-timeline']").text # 获取国际申请时间
        try:
            priority_time = driver.find_element_by_xpath("//div[@class = 'priority style-scope application-timeline']").text # 获取优先申请时间
        except:
            priority_time = filing_time  # 存在优先权如无记录的情况，此时以国际申请时间为准
        publication_time = driver.find_element_by_xpath("//div[@class = 'publication style-scope application-timeline']").text  # 获取专利公开时间

        # 2.获取发明人和授权人
        inventors_assignee = driver.find_elements_by_xpath('//dt[text()="Inventor"]/following-sibling::*')
        inv_ass_item_list = [item.text for item in inventors_assignee if item.text != '']  # 去除空格
        if 'Current Assignee' in inv_ass_item_list:   # 分解发明人和授权人节点
            split_index = inv_ass_item_list.index('Current Assignee')
            inventors = ','.join(inv_ass_item_list[:split_index])  # 获取发明人
            assignee = ','.join(inv_ass_item_list[split_index+1:])  # 获取授权人
        else:
            inventors = ','.join(inv_ass_item_list)
            assignee = ''
        if assignee == '':   # 补充授权人
            try:
                assignee = driver.find_element_by_xpath('//span[contains(text(), "Application filed by")]').text
                assignee = assignee.split('by')[1].strip()
            except:
                pass

        # 3.获取摘要内容
        try:
            abstract = driver.find_element_by_tag_name('abstract').text
        except:
            abstract = ''

        # 4.获取权利要求数量
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.ID, "claims")))
        claim_num = re.findall(num, driver.find_element_by_id('claims').find_element_by_tag_name('h3').text)
        if claim_num:
            claim_num = int(claim_num[0])
        else:
            try:
                claim_num = len(driver.find_element_by_id('claims').find_elements_by_tag_name('claim'))
                if claim_num == 1:
                    claims = driver.find_element_by_id('claims').find_elements_by_tag_name('div')
                    claims = [item.text for item in claims if re.match('^\d+\..*', item.text)]
                    claim_num = re.findall('\d+', claims[-1])[0]
            except:
                claim_num = 0

        # 5.获取描述范围数量
        WebDriverWait(driver, 60, 0.5).until(EC.presence_of_element_located((By.ID, "descriptionText")))
        descr_num1 = len(driver.find_element_by_id('description').find_elements_by_tag_name('p'))
        descr_num2 = len(driver.find_element_by_id('description').find_elements_by_tag_name('li'))
        descr_num = max(descr_num1, descr_num2)

        # 6.获取pdf链接
        try:
            pdf_link = driver.find_element_by_link_text('Download PDF').get_attribute('href')  # 获取pdf链接
        except:
            pdf_link = ''

        result_data_list = [pymysql.escape_string(pubnum), pymysql.escape_string(title), pymysql.escape_string(inventors), assignee,
                            priority_time, filing_time, publication_time, family_list, family_num, patent_citations,
                            non_patent_citations, cited_by, claim_num, descr_num, final_classification_num,
                            final_classification_list, current_link, pdf_link, pymysql.escape_string(abstract)]
        return result_data_list

    # 判断是否存在英文版本（说明当前版本不是英文版），如果有，即点击转换为英文版
    def get_english_version(self, driver):
        try:
            driver.find_element_by_link_text('English').click()
        except:
            pass

    # 依据pattern和xpath获得元素中存在的数字
    def get_element_num(self, driver, pattern, xpath):
        try:
            patent_citations = driver.find_element_by_xpath(xpath).text
            return re.findall(pattern, patent_citations)[0]
        except:
            return 0

    # 创建数据表
    def createsql(self, table_name):
        conn = pymysql.connect(host='localhost', user='root', passwd='123456', db='thesis', port=3306, charset='utf8')
        cur = conn.cursor()
        sql = "CREATE TABLE `" + table_name + "` \
            (`id` int(255) NOT NULL,\
            `patent_id` varchar(50) NOT NULL,\
            `name` longtext NOT NULL,\
            `inventor` longtext NOT NULL,\
            `current_assignee` varchar(255) NOT NULL,\
            `priority_date` varchar(255) NOT NULL,\
            `filing_date` varchar(255) NOT NULL,\
            `publication_date` varchar(255) NOT NULL,\
            `family` varchar(255) NOT NULL,\
            `family_num` int(8) NOT NULL DEFAULT '1',\
            `patent_citations` int(8) NOT NULL DEFAULT '0',\
            `non_patent_citations` int(8) NOT NULL DEFAULT '0',\
            `cited_by` int(8) NOT NULL DEFAULT '0',\
            `claim_num` int(8) NOT NULL,\
            `description_num` int(8) NOT NULL,\
            `CPC_num` int(8) NOT NULL,\
            `CPC` longtext NOT NULL,\
            `webpage_link` longtext NOT NULL,\
            `pdf_link` varchar(255) NOT NULL,\
            `abstract` longtext NOT NULL,\
            PRIMARY KEY (`patent_id`)\
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8;"
        cur.execute(sql)
        conn.commit()
        cur.close()
        conn.close()

    # 插入数据
    def insertsql(self, table_name, patents_count, result_data):
        result_data.insert(0, patents_count)
        result_data = tuple(result_data)
        conn = pymysql.connect(host='localhost', user='root', passwd='123456', db='thesis', port=3306, charset='utf8')
        cur = conn.cursor()
        sql = "REPLACE INTO " + table_name + "(id, patent_id, name, inventor, current_assignee, priority_date," \
              "filing_date, publication_date, family, family_num, patent_citations, non_patent_citations, " \
              "cited_by, claim_num, description_num, CPC_num, CPC, webpage_link, pdf_link, abstract) VALUES " \
              "('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s'," \
              " '%s', '%s', '%s', '%s', '%s')"
        cur.execute(sql % result_data)
        conn.commit()
        cur.close()
        conn.close()

    def table_exists(self, table_name):  # 这个函数用来判断表是否存在
        conn = pymysql.connect(host='localhost', user='root', passwd='123456', db='thesis', port=3306, charset='utf8')
        cur = conn.cursor()
        sql = "show tables;"
        cur.execute(sql)
        tables = [cur.fetchall()]
        table_list = re.findall('(\'.*?\')', str(tables))
        table_list = [re.sub("'", '', each) for each in table_list]
        if table_name in table_list:
            return 1
        return 0

    # 获取数据表中优先权最晚的时间，数据表已记录的专利数量
    def get_db_last_patent(self, table_name):
        conn = pymysql.connect(host='localhost', user='root', passwd='123456', db='thesis', port=3306, charset='utf8')
        cur = conn.cursor()
        sql = "SELECT COUNT(`patent_id`), MAX(`filing_date`) FROM `" + table_name + "`"
        cur.execute(sql)
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row

    # 获取数据表中ID最大的时间
    def get_db_max_id_patent(self, table_name):
        conn = pymysql.connect(host='localhost', user='root', passwd='123456', db='thesis', port=3306, charset='utf8')
        cur = conn.cursor()
        sql = "SELECT `id`, `filing_date` FROM `" + table_name + "` ORDER BY `id` DESC LIMIT 1"
        cur.execute(sql)
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row


if __name__ == '__main__':
    A = MainScrapy()
    # logger = A.logger_setter()  # 设置日志
    # A.scrapy(logger)
    # A.scrapy_special_time(logger, '2017-12-07', '2018-01-16')

