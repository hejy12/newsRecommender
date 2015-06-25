#coding= utf-8
import sys
import jieba
import jieba.analyse

infile = "train_data.csv"
outfile = "_tfidf.csv"

def process(filename):
    infi = open(filename,'r')
    outfi = open(filename+outfile,'w+')
    for txt in infi:
        txts = txt.split('\t')
        #print(txts[4])
        #print("use TF.IDF to extract key words: jieba.analyse.extract_tags():")
        #tfidf = jieba.analyse.extract_tags(txts[4], topK=20,  withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
        tfidf = jieba.analyse.extract_tags(txts[4], topK=20,  withWeight=False)
        keywords=u""
        for xx in tfidf:
            #print xx 
            keywords+=xx
            keywords+='|'
        tfidf = jieba.analyse.extract_tags(txts[3], topK=5,  withWeight=False)
        titlewords =u""
        for xx in tfidf:
            titlewords += xx
            titlewords += '|'
        line=u""
        for i in range(3):
            line += (unicode(txts[i],'utf-8'))
            line +=('\t')
        line += (titlewords+'\t')
        line += (keywords+'\t')
        line += (unicode(txts[5],'utf-8')+'\r\n')
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
        process(name)

if __name__ == '__main__':
    main()

