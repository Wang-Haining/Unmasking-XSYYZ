#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hw56@iu.edu

Created on Fri DEC 21 23:59:59 2019

"""
import re
from hanziconv import HanziConv

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


# 去除"XX品"
ch_mark_set2 = re.findall(pattern=r"\s(游戏品|广仁品|广慧品|正法品|妙悟品|戒导品|正法品|净行品|庄严品|证入品|证人品|解脱品)\s",
                          string=xjpm_txt,
                          flags=re.S | re.M)
for mark in ch_mark_set2:
    xjpm_txt = xjpm_txt.replace(mark, "")

xjpm_txt = xjpm_txt.replace(u'\n', '')
xjpm_txt = xjpm_txt.replace(u' ', '')

# remove chapter markers
# ch_mark_set1 = re.findall(pattern=r"(第[一二三四五六七八九十]{1,3}回[ \n]{1,3}.{7,9}?[ \n]{1,3}.{7,9}?[ \n])",
#                           string=xjpm_txt,
#                           flags=re.S | re.M)
#
# for mark in ch_mark_set1:
#     xjpm_txt = xjpm_txt.replace(mark, "")

with open('/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/XJPM_1.txt', 'w', encoding='utf8') as f:
    f.write(xjpm_txt)
# 《续金瓶梅》处理结束
# 仍有"口口口口"未处理


# 《醒世姻缘传》处理开始
xsyyz = '/Users/hwang/Desktop/repo/Unmasking-XSYYZ/unknown/XSYYZ.txt'
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

xsyyz_txt = xsyyz_txt.replace(u'\n', '')
xsyyz_txt = xsyyz_txt.replace(u' ', '')

with open('/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/XSYYZ_1.txt', 'w', encoding='utf8') as f:
    f.write(xsyyz_txt)
# 《醒世姻缘传》处理结束


# 《女聊斋》处理开始
nlzzy = '/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/NLZZY.txt'
nlzzy_txt = text_handler(nlzzy)
nlzzy_txt = nlzzy_txt.replace(u'\u3000\u3000', '\n')
nlzzy_txt = nlzzy_txt.replace(u'\u3000', '\n')
nlzzy_txt = nlzzy_txt.replace(u'\n', '')
nlzzy_txt = nlzzy_txt.replace(u' ', '')

with open('/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/NLZZY_1.txt', 'w', encoding='utf8') as f:
    f.write(nlzzy_txt)
# 《女聊斋》处理结束


# 《聊斋》处理开始
lzzy = '/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/LZZY.txt'
lzzy_txt = text_handler(lzzy)
lzzy_txt = lzzy_txt.replace(u'\u3000\u3000', '\n')
lzzy_txt = lzzy_txt.replace(u'\u3000', '\n')
lzzy_txt = lzzy_txt.replace(u'\n', '')
lzzy_txt = lzzy_txt.replace(u' ', '')
lzzy_txt = lzzy_txt.replace(u'「', '“')
lzzy_txt = lzzy_txt.replace(u'」', '”')
lzzy_txt = lzzy_txt.replace(u'『', '“')
lzzy_txt = lzzy_txt.replace(u'』', '”')


with open('/Users/hwang/Desktop/repo/Unmasking-XSYYZ/corpus/LZZY_1.txt', 'w', encoding='utf8') as f:
    f.write(lzzy_txt)
# 《聊斋》处理结束


# 《金瓶梅》处理开始
jpm = '/Users/hwang/Desktop/repo/Unmasking-XSYYZ/LanlingXiaoxiaosheng/JPM.txt'
jpm_txt = text_handler(jpm)
jpm_txt = jpm_txt.replace(u'\u3000\u3000', '\n')
jpm_txt = jpm_txt.replace(u'\u3000', '\n')
jpm_txt = jpm_txt.replace(u'\n', '')
jpm_txt = jpm_txt.replace(u' ', '')
jpm_txt = jpm_txt.replace(u'「', '“')
jpm_txt = jpm_txt.replace(u'」', '”')
jpm_txt = jpm_txt.replace(u'『', '“')
jpm_txt = jpm_txt.replace(u'』', '”')


with open('/Users/hwang/Desktop/repo/Unmasking-XSYYZ/LanlingXiaoxiaosheng/JPM_1.txt', 'w', encoding='utf8') as f:
    f.write(jpm_txt)
# 《金瓶梅》处理结束


# 《天史》处理开始
ts = '/Users/hwang/Desktop/repo/Unmasking-XSYYZ/reference/TS.txt'
ts_txt = text_handler(ts)

from hanziconv import HanziConv
ts_txt = HanziConv.toSimplified(ts_txt)

ts_txt = ts_txt.replace(u'\u3000\u3000', '\n')
ts_txt = ts_txt.replace(u'\u3000', '\n')
ts_txt = ts_txt.replace(u'\n', '')
ts_txt = ts_txt.replace(u' ', '')
ts_txt = ts_txt.replace(u'「', '“')
ts_txt = ts_txt.replace(u'」', '”')
ts_txt = ts_txt.replace(u'『', '“')
ts_txt = ts_txt.replace(u'』', '”')

with open('/Users/hwang/Desktop/repo/Unmasking-XSYYZ/reference/TS_1.txt', 'w', encoding='utf8') as f:
    f.write(ts_txt)
# 《天史》处理结束


# 《木皮散人鼓词》处理开始
mpsrgc = '/Users/hwang/Desktop/repo/Unmasking-XSYYZ/JiaYingchong/MPSRGC.txt'
mpsrgc_txt = text_handler(mpsrgc)

# mpsrgc_txt = HanziConv.toSimplified(mpsrgc_txt)

mpsrgc_txt = mpsrgc_txt.replace(u'\u3000\u3000', '\n')
mpsrgc_txt = mpsrgc_txt.replace(u'\u3000', '\n')
mpsrgc_txt = mpsrgc_txt.replace(u'\n', '')
mpsrgc_txt = mpsrgc_txt.replace(u' ', '')
mpsrgc_txt = mpsrgc_txt.replace(u'「', '“')
mpsrgc_txt = mpsrgc_txt.replace(u'」', '”')
mpsrgc_txt = mpsrgc_txt.replace(u'『', '“')
mpsrgc_txt = mpsrgc_txt.replace(u'』', '”')

with open('/Users/hwang/Desktop/repo/Unmasking-XSYYZ/JiaYingchong/MPSRGC_1.txt', 'w', encoding='utf8') as f:
    f.write(mpsrgc_txt)
# 《木皮散人鼓词》处理结束