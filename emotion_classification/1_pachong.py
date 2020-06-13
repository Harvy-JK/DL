# 抓取网页内容所需要的包
import json
import requests  # Requests 是⽤Python语⾔编写，基于urllib，采⽤Apache2 Licensed开源协议的 HTTP 库

# Pytorch所需要的包
import torch
import torch.nn as nn
import torch.optim

# 自然语言处理（NLP）所需要的包
import re  # 正式表达式的包，用来对文本的过滤或者规则的匹配
import jieba  # 中文分词库
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# %matplotlib inline     # 这里jupyter魔法公式，pycharm不适用

# **********************************************************
# 先搜集数据，通过制作一个简单的网页爬虫程序即可
# 在指定的url处获得评论
def get_comments(url):
    comments = []  # 收集
    # 打开指定页面
    resp = requests.get(url)
    resp.encoding = 'gbk'  # 汉字字库之一

    # 如果200秒没有打开则失败
    if resp.status_code != 200:
        return []

    # 获得内容
    content = resp.text
    if content:
        # 获得（）括号中的内容
        ind = content.find('(')
        s1 = content[ind + 1:-2]
        try:
            # 尝试利用jason接口来读取内容，并做jason的解析
            js = json.loads(s1)
            # 提取出comments字段的内容
            comment_infos = js['comments']
        except:
            print('error')
            return ([])

        # 对每一条评论进行内容部分的抽取
        for comment_info in comment_infos:
            comment_content = comment_info['content']
            str1 = comment_content + '\n'
            comments.append(str1)
    return comments

good_comments = []

# 评论抓取的来源地址，其中参数包括：
# productId为商品的id，score为评分，page为对应的评论翻页的页码，pageSize为总页数
# 这里，我们设定score＝3表示好的评分。
# 任天堂Switch游戏机，sony降噪耳机豆，大疆Mavic Air2， go pro Hero8， insta 360， SONY Alpha 7R IV， RTX 2080，mac
good_comment_url_templates = [
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100010566524&score=0&sortType=5&page=0&pageSize=10&isShadowSku=100010343850&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100006585530&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100012791070&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100004918245&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100007877688&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100006852812&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100001121028&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100007136953&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1'
]

# 对上述网址进行循环，并模拟翻页100次
j = 0
for good_comment_url_template in good_comment_url_templates:
    for i in range(100):
        url = good_comment_url_template.format(i)
        good_comments += get_comments(url)
        print('第{}条纪录，总文本长度{}'.format(j, len(good_comments)))
        j += 1
# 将结果存储到good.txt文件中
fw = open('data/good.txt', 'w')
fw.writelines(good_comments)

# 负向评论如法炮制
bad_comments = []
bad_comment_url_templates = [
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100010566524&score=0&sortType=5&page=0&pageSize=10&isShadowSku=100010343850&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100006585530&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100012791070&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100004918245&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100007877688&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100006852812&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100001121028&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1',
    'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=100007136953&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1'
]

j = 0
for bad_comment_url_template in bad_comment_url_templates:
    for i in range(100):
        url = bad_comment_url_template.format(i)
        bad_comments += get_comments(url)
        print('第{}条纪录，总文本长度{}'.format(j, len(bad_comments)))
        j += 1

fw = open('data/bad.txt', 'w')
fw.writelines(bad_comments)

