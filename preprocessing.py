#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hw56@iu.edu

Created on Fri DEC 21 23:59:59 2019

"""
import re

def text_handler(file_name):
    lines = ''
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def interceptor(text):
    import re
    # import nltk
    # import jieba
    # word_tokens = nltk.regexp_tokenize(text, pattern=r".")
    # re.findall(pattern=r'【[\u4e00-\u9fa5]+[\w\W]{0,100}】', string=origin)
    comments = re.findall(r".*【(.*?)】.*", origin)


    gc_txt = gc_txt.replace(u'\u3000\u3000', '\n')


    return None


# 《续金瓶梅》处理开始
xjpm = '/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/XJPM.txt'
xjpm_txt = text_handler(xjpm)

# 再去掉全角空格
xjpm_txt = xjpm_txt.replace(u'\u3000\u3000', '\n')
xjpm_txt = xjpm_txt.replace(u'\u3000', '\n')

# remove chapter markers
ch_mark_set1 = re.findall(pattern=r"(第[一二三四五六七八九十]{1,3}回[ \n]{1,3}.{7,9}?[ \n]{1,3}.{7,9}?[ \n])",
                          string=xjpm_txt,
                          flags=re.S | re.M)

for mark in ch_mark_set1:
    xjpm_txt = xjpm_txt.replace(mark, "")

with open('/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/XJPM_1.txt', 'w', encoding='utf8') as f:
    f.write(xjpm_txt)
# 《续金瓶梅》处理结束
# 仍有"口口口口"未处理


# 《醒世姻缘传》处理开始
xsyyz = '/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/XSYYZ.txt'
xsyyz_txt = text_handler(xsyyz)

# 再去掉全角空格
xsyyz_txt = xsyyz_txt.replace(u'\u3000\u3000', '\n')
xsyyz_txt = xsyyz_txt.replace(u'\u3000', '\n')

# remove chapter markers
ch_mark_set1 = re.findall(pattern=r"(第[一二三四五六七八九十百]{1,3}回[ \n]{1,3}.{7,9}?[ \n]{1,3}.{7,9}?[ \n])",
                          string=xsyyz_txt,
                          flags=re.S | re.M)

for mark in ch_mark_set1:
    xsyyz_txt = xsyyz_txt.replace(mark, "")

with open('/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/XSYYZ_1.txt', 'w', encoding='utf8') as f:
    f.write(xsyyz_txt)
# 《醒世姻缘传》处理结束

