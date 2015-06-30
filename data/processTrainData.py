#coding= utf-8
# 本脚本完成以下三个任务：
# *  原始数据简化为：<用户id，新闻id，点击时间>
# *  统计形成一个所有新闻的列表，包含：<新闻id，新闻标题，新闻内容，发表时间>
# *  用TF.IDF 算法给新闻列表数据提取关键词，新闻列表的格式变为：<新闻id，新闻标题+内容的关键词（set[keywords:weight]），发表时间>
# *  统计所有新闻中的关键词，形成一个关键词的哈希表： <hash(关键词)，关键词>
# *  用哈希值替换新闻列表中的关键词，新闻列表的格式变为：<新闻id，新闻标题关键词+内容关键词列表（set[hash(keyword):weight]） ），发表时间> 
# *  对用户打标签，用户表简化为：<用户id，关键词表（hashset），浏览时间段（分布）> 。 其中的关键词带了权重，其权重值为用户浏览过的历史的关键词权重的和。
# *  由于用户的浏览新闻的行为和内容是时间相关的，因此简单的对用户的浏览关键词权重相加是不准确的，所以，我们在这里可以考虑引入对关键词的权重引入“半衰期”的概念，即用户浏览一些关键词每过去一段时间，该关键词的权重衰减一半。这样或许可以更准确的表达时间相关的问题。
#

import sys,os
import jieba
import jieba.analyse
import hashlib
import pickle

#infile = "train_data.csv"
outfile_brief = "_brief.csv"
outfile_news = "_news.csv"
outfile_news_tdidf = "_tfidf.csv"
outfile_keyword_hash = "_keywords.csv"
outfile_news_tdidf_hash = "_hashed.csv"
outfile_users ="_users.csv"

KEYWORDS = {}
USERS = {}
NEWS = {}
NEWSRAW = {}

def md5(value):
    return hashlib.md5(value).hexdigest()

#简单的保存变量

def saveObjs(obj,savename):#dump用户的联系人列表到本地
    output = open(savename, 'wb+')
    pickle.dump(obj,output)
    output.close()

def loadObjsIfExist(savename):#启动的时候载入用户联系人列表
    result =None
    if os.path.exists(savename):
        pkl_file = open(savename, 'rb')
        result = pickle.load(pkl_file)
        pkl_file.close()
    return result

def process_brief(filename):
    print 'processing biref...'
    outfname = filename+outfile_brief
    infi = open(filename,'r')
    outfi = open(outfname,'w+')
    for txt in infi:
        txts = txt.split('\t')
        line=u""
        line += (unicode(txts[0].strip(),'utf-8')+'\t')#uid
        line += (unicode(txts[1].strip(),'utf-8')+'\t')#newsid
        line += (unicode(txts[2].strip(),'utf-8')+'\r\n')#access time
        #print line
        outfi.write(line.encode('utf-8'))
        outfi.flush()
    infi.close()
    outfi.close()
    return outfname

def process_news(filename):
    print 'processing: extracting news...'
    outfname = filename+outfile_news
    infi = open(filename,'r')
    outfi = open(outfname,'w+')
    for txt in infi:
        txts = txt.split('\t')
        newsid = long(txts[1].strip()) #news id
        NEWSRAW[newsid]= [txts[3],txts[4],txts[5]]
    for txt in NEWSRAW.keys():
        line=u""
        txts = NEWSRAW[txt] #news id
        line += (unicode(str(txt),'utf-8')+'\t') #news id
        line += (unicode(txts[0],'utf-8')+'\t') # news title
        line += (unicode(txts[1],'utf-8')+'\t') # news content
        line += (unicode(txts[2].strip(),'utf-8')+'\r\n') # time published
        #print line
        outfi.write(line.encode('utf-8'))
        outfi.flush()
    saveObjs(NEWSRAW,'NEWSRAW.pkl')
    infi.close()
    outfi.close()
    return outfname

def process_tdidf(filename):
    print 'processing: extracting news tags with TD.IDF...'
    outfname = filename+outfile_news_tdidf
    infi = open(filename,'r')
    outfi = open(outfname,'w+')
    for txt in infi:
        txts = txt.split('\t')
        #print(txts[4])
        #print("use TF.IDF to extract key words: jieba.analyse.extract_tags():")
        #tfidf = jieba.analyse.extract_tags(txts[4], topK=20,  withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
        #提取关键词，并且合并 标题和 内容的关键词，若有相同关键词，则权重相加
        tfidf = jieba.analyse.extract_tags(txts[2], topK=20,  withWeight=True)
        mkeywords={}
        for xx in tfidf:
            mkeywords[xx[0]]= xx[1]
        tfidf = jieba.analyse.extract_tags(txts[1], topK=5,  withWeight=True)
        for xx in tfidf:
            if xx[0] in mkeywords.keys():
                mkeywords[xx[0]] += xx[1]
            else:
                mkeywords[xx[0]] = xx[1]
        line=u""
        titlewords =u""
        for xx in mkeywords.keys():
            titlewords += xx
            titlewords+=(':'+ str(mkeywords[xx])+'|')
        line += (unicode(txts[0],'utf-8')+'\t')
        line += (titlewords+'\t')
        line += (unicode(txts[3].strip(),'utf-8')+'\r\n')
        #print line
        outfi.write(line.encode('utf-8'))
        outfi.flush()
    infi.close()
    outfi.close()
    return outfname

def process_keywords(filename):
    print 'processing: news tags to hash...'
    outfname = filename+ outfile_news_tdidf_hash
    infi = open(filename,'r')
    outfi = open(filename+outfile_keyword_hash,'w+')
    outfi_h = open(outfname,'w+')
    for txt in infi:
        txts = txt.split('\t')
        #print txts[0]  # newsid
        mkeywords = {}
        #输出关键词到全局变量
        w_kw = txts[1].split('|') #news keywords
        for kw in w_kw:
            kv = kw.split(':')
            if(len(kv)==2):
                mkey = md5(kv[0].strip())
                KEYWORDS[mkey] = kv[0].strip()#保存成变量
                mkeywords[mkey] = float(kv[1].strip())
        #替换关键词为hash并且输出到文件
        line = u""
        titlewords =u""
        for xx in mkeywords.keys():
            titlewords += xx
            titlewords+=(':'+ str(mkeywords[xx])+'|')
        line += (txts[0]+'\t')
        line += (titlewords+'\t')
        line += (unicode(txts[2].strip(),'utf-8')+'\r\n') # news time
        #print line
        outfi_h.write(line.encode('utf-8'))
        outfi_h.flush()
        NEWS[txts[0].strip()]= [mkeywords,txts[2].strip()]#保存成变量
    for kw in KEYWORDS.keys():
        line =u""
        line += (kw+'\t')
        line += (unicode(KEYWORDS[kw],'utf-8')+'\r\n')
        outfi.write(line.encode('utf-8'))
    outfi.flush()
    infi.close()
    outfi.close()
    outfi_h.close()
    saveObjs(NEWS,'NEWS.pkl')
    saveObjs(KEYWORDS,'KEYWORDS.pkl')
    return outfname

def process_users(filename):
    print 'processing: extracting users tag...'
    outfname = filename+ outfile_users
    infi = open(filename,'r')
    outfi = open(outfname,'w+')
    for txt in infi:
        txts = txt.split('\t')
        #print txts[0],txts[1],txts[2].strip()
        muid = txts[0].strip()
        mnewsid = txts[1].strip()
        if(muid in USERS.keys()): #取出用户可能已经有的关键字
            usertags = USERS[muid]
        else: 
            usertags ={}
        if(mnewsid in NEWS.keys()): #取出一条新闻的关键字
            newstags = NEWS[mnewsid][0]
            for tag in newstags.keys():
                w = newstags[tag]
                if (tag in usertags.keys()): #更新关键字的权重
                    usertags[tag]+= float(w)
                else:
                    usertags[tag] = float(w)
            #print usertags
        USERS[muid]=usertags
    for muid in USERS.keys(): #输出到文本文件
        keywords=u''
        usrtags = USERS[muid]
        for tag in usrtags.keys():
            keywords += unicode(KEYWORDS[tag],'utf-8')
            keywords +=':'
            keywords += (str(usrtags[tag])+"|")
        line = u""
        line += (unicode(muid,'utf-8')+'\t')
        line += (keywords+'\r\n')
        outfi.write(line.encode('utf-8'))
    saveObjs(USERS,'USERS.pkl')
    outfi.flush()
    infi.close()
    outfi.close()
    return outfname

def recommend(userid, usertags, newslist):
    print "Recommending TOP10 news for user: ",userid
    ukeys = usertags.keys()
    rc_news =[]
    #print ukeys
    for nk in newslist.keys():
        Re =0.0
        S1=0.000001#相同标签的权重
        S2=sum(usertags.values())#不同标签的权重
        ntags = newslist[nk][0] #每一条新闻的标签
        for newstag in ntags.keys():
            #print newstag #对比标签
            if newstag in ukeys:
                S1+= ntags[newstag]
                S1+= usertags[newstag]
                S2-= usertags[newstag]
            else: S2+= ntags[newstag]
        Re = S1/(S1+S2)
        rc_news.append((Re,nk))
    rc_news.sort(reverse=True)
    for x in range(10):
        nk = rc_news[x][1]
        print rc_news[x], NEWSRAW[long(nk)][0]
    pass

def main():
    usage ='''useage :  python thisfile.py xxxx.txt
        1st: You should training the model: $python thisfile.py train <rawdata>
        2nd: To test the model, use: $python thisfile.py active <uid> <newsList>
            example:  python processTrainData.py active 2817069 data2.txt
        '''
    if(len(sys.argv) <1):
        print usage
        exit()
    else:
        if (sys.argv[1]=='train'):
            name = sys.argv[2]
            outbrief = process_brief(name)
            outnews = process_news(name)
            outtdidf = process_tdidf(outnews)
            outkeyword = process_keywords(outtdidf)
            process_users(outbrief)
        elif (sys.argv[1]=='active'):
            userid = sys.argv[2]
            newslist = sys.argv[3]
            # load users
            print 'loading user info...'
            USERS= loadObjsIfExist('USERS.pkl')
            print 'finding user tag...' 
            utag = USERS[userid]
            # process test news
            outbrief = process_brief(newslist)
            outnews = process_news(newslist)
            outtdidf = process_tdidf(outnews)
            outkeyword = process_keywords(outtdidf)
            
            recommend(userid, utag, NEWS)
            # compare
        else:
            print usage
            exit()

if __name__ == '__main__':
    main()
