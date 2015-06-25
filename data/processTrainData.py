#coding= utf-8
# 本脚本完成以下三个任务：
# *  原始数据简化为：<用户id，新闻id，点击时间>
# *  统计形成一个所有新闻的列表，包含：<新闻id，新闻标题，新闻内容，发表时间>
# *  用TF.IDF 算法给新闻列表数据提取关键词，新闻列表的格式变为：<新闻id，新闻标题（set[keywords:weight,]），新闻内容（set[keywords:weight]），发表时间>

import sys
import jieba
import jieba.analyse

#infile = "train_data.csv"
outfile_brief = "_brief.csv"
outfile_news = "_news.csv"
outfile_news_tdidf = "_tfidf.csv"

def process_brief(filename):
    infi = open(filename,'r')
    outfi = open(filename+outfile_brief,'w+')
    for txt in infi:
        txts = txt.split('\t')
        line=u""
        line += (unicode(txts[0],'utf-8')+'\t')
        line += (unicode(txts[1],'utf-8')+'\t')
        line += (unicode(txts[2],'utf-8')+'\r\n')
        print line
        outfi.write(line.encode('utf-8'))
        outfi.flush()
    infi.close()
    outfi.close()
        
def process_news(filename):
    infi = open(filename,'r')
    outfi = open(filename+outfile_news,'w+')
    news ={}
    for txt in infi:
        txts = txt.split('\t')
        newsid = long(txts[1].strip())
        news[newsid]= [txts[3],txts[4],txts[5]]
    for txt in news.keys():
        line=u""
        txts = news[txt]
        line += (unicode(str(txt),'utf-8')+'\t')
        line += (unicode(txts[0],'utf-8')+'\t')
        line += (unicode(txts[1],'utf-8')+'\t')
        line += (unicode(txts[2].strip(),'utf-8')+'\r\n')
        print line
        outfi.write(line.encode('utf-8'))
        outfi.flush()
    infi.close()
    outfi.close()

def process_tdidf(filename):
    infi = open(filename,'r')
    outfi = open(filename+outfile_news_tdidf,'w+')
    for txt in infi:
        txts = txt.split('\t')
        #print(txts[4])
        #print("use TF.IDF to extract key words: jieba.analyse.extract_tags():")
        #tfidf = jieba.analyse.extract_tags(txts[4], topK=20,  withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
        tfidf = jieba.analyse.extract_tags(txts[2], topK=20,  withWeight=True)
        keywords=u""
        for xx in tfidf:
            #print xx 
            keywords+=xx[0]
            keywords+=(':'+ str(xx[1])+'|')
        tfidf = jieba.analyse.extract_tags(txts[1], topK=5,  withWeight=True)
        titlewords =u""
        for xx in tfidf:
            titlewords += xx[0]
            titlewords += (':'+str(xx[1])+'|')
        line=u""
        line += (unicode(txts[0],'utf-8')+'\t')
        line += (titlewords+'\t')
        line += (keywords+'\t')
        line += (unicode(txts[3].strip(),'utf-8')+'\r\n')
        print line
        outfi.write(line.encode('utf-8'))
        outfi.flush()
    infi.close()
    outfi.close()

def main():
    if(len(sys.argv) <1):
        print '''useage :  python thisfile.py xxxx.txt'''
        exit()
    else:
        name = sys.argv[1]
        process_brief(name)
        process_news(name)
        process_tdidf(name+outfile_news)
        

if __name__ == '__main__':
    main()

