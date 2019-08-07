import pickle
import operator
import MySQLdb
from dateutil.parser import parse
import re

db = MySQLdb.connect("localhost","root","1234","flaskapp")

# prepare a cursor object using cursor() method
cursor = db.cursor()

# db.set_character_set('utf8')
# cursor.execute('SET NAMES utf8;')
# cursor.execute('SET CHARACTER SET utf8;')
# cursor.execute('SET character_set_connection=utf8;')

# sql = "CREATE TABLE threads1(id INT(11) AUTO_INCREMENT PRIMARY KEY, author VARCHAR(100), subject VARCHAR(350), date TIMESTAMP);"
# cursor.execute(sql)
# sql = "CREATE TABLE mails1(id INT(11) AUTO_INCREMENT PRIMARY KEY, thread_no INT(11),author VARCHAR(100), subject VARCHAR(350), content LONGTEXT, date TIMESTAMP);"
# cursor.execute(sql)
# sql = "SET NAMES utf8mb4;"
# cursor.execute(sql)
# sql = "ALTER DATABASE flaskapp CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci;"
# cursor.execute(sql)

def deb_toppostremoval(temp):
    strings = temp.splitlines()
    temp = ''
    for st in strings:
        st = st.strip()
        if len(st) > 0:
            if st[0] == '>':
                continue
            else:
                temp += '\n' + st
    return temp

def clean_debian(temp):
    temp = re.sub('\n+', '\n', temp)
    temp = re.sub('\n', ' ', temp)
    temp = re.sub('\t', ' ', temp)
    temp = re.sub(' +', ' ', temp)
    return temp

def short(l, val) :
    new_l = []
    for item in l :
        if item > val :
            new_l.append(item)
    return new_l

class Recom :
    def __init__(self) :
        self.threadDictPath = "Data/thread_dict.pkl"
        self.userDictPath = "Data/userdict.pkl"
        self.testingDataPath = "Data/test2.pkl"
        self.trainingDataPath = "Data/train2.pkl"

        self.threadDict = pickle.load(open(self.threadDictPath, "rb"))
        self.userDict = pickle.load(open(self.userDictPath, "rb"))
        self.testingData = pickle.load(open(self.testingDataPath, "rb"))
        self.trainingData = pickle.load(open(self.trainingDataPath, "rb"))
        self.allData = self.trainingData.append(self.testingData, ignore_index=True)


    def getThreadIds(self) :
        threadList = []
        for key, value in self.threadDict.items() :
            threadList.append(key) 
        print(threadList[0])
        print(len(threadList))
        return threadList

    def putintodb(self) :
        print(type(self.allData))
        self.allData = self.allData.sort_values(by='thread_no')
        print(list(self.allData['thread_no'].iloc[:50]))
        flag = True       
        previd = 0
        prevauthor = ''
        prevcontent = ''
        prevdate = '' 
        threadlist = self.getThreadIds()
        # print(len(threadlist))
        # threadlist = short(threadlist, 1665)
        # print(len(threadlist))
        self.allData = self.allData[self.allData['thread_no'].isin([10427])]
        for index, row in self.allData.iterrows() :
            
            previd = int(row['thread_no'])
            prevauthor = row['replier']
            prevcontent = clean_debian(deb_toppostremoval(row['mail']))
            # prevcontent = prevcontent[:min(65350, len(prevcontent))]
            prevdate = str(parse(row['cur_date'])).split('+')[0]
            prevdate = prevdate[:19]
            break

        # print(type(row['thread_no']))
        print(previd)

        for index, row in self.allData.iterrows() :
            curid = int(row['thread_no'])
            curauthor = row['replier']
            curcontent = clean_debian(deb_toppostremoval(row['mail']))
            # curcontent = curcontent[:min(60000, len(curcontent))]
            curdate = str(parse(row['cur_date'])).split('+')[0]
            curdate = curdate[:19]  
            print(curid)
            if previd != curid :
                # sql = """INSERT INTO threads1 (thread_no, author, subject, date) VALUES(%d,%s,%s,%s,%s)""",(previd, prevauthor, prevcontent[:300], str(prevdate))
                # print("gg : " + str(previd) + " , " + prevcontent[:100])
                try :
                    cursor.execute("""INSERT INTO threads1 (id, author, subject, date) VALUES(%s,%s,%s,%s)""",(str(previd+1), prevauthor, prevcontent[:300], str(prevdate)))
                    db.commit()
                except :
                    print("not putting : " + str(previd))

            if curid != previd :
                flag = True

            if flag :
                previd = curid
                prevauthor = curauthor

            prevcontent = curcontent
            prevdate = curdate

            flag = False
            # print("ll : " + str(curid) + " , " + curcontent[:100])
            # sql = """INSERT INTO mails1 (thread_no, author, subject, content, date) VALUES(%s,%s,%s,%s)""",(str(curid), curauthor, curcontent[:300], curcontent, str(curdate))
            # try :
            #     cursor.execute("""INSERT INTO mails1 (thread_no, author, subject, content, date) VALUES(%s,%s,%s,%s,%s)""",(str(curid+1), curauthor, curcontent[:300], curcontent, str(curdate)))
            #     db.commit()
            # except :
            #     print("not putting : " + str(curid))

        # sql = """INSERT INTO threads1 (id, author, subject, date) VALUES(%d,%s,%s,%s,%s)""",(previd, prevauthor, prevcontent[:300], str(prevdate))
        cursor.execute("""INSERT INTO threads1 (id, author, subject, date) VALUES(%s,%s,%s,%s)""",(str(previd+1), prevauthor, prevcontent[:300], str(prevdate)))
        db.commit()



            


        
   




ob = Recom()
ob.getThreadIds()
ob.putintodb()