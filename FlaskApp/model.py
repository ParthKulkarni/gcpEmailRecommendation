import pickle
import operator

class Recom :
    def __init__(self) :
        #self.threadDictPath = "Data/thread_dict.pkl"
        self.userDictPath = "Data/userdict.pkl"
        #self.testingDataPath = "Data/test2.pkl"
        #self.trainingDataPath = "Data/train2.pkl"

        # self.threadDict = pickle.load(open(self.threadDictPath, "rb"))
        self.userDict = pickle.load(open(self.userDictPath, "rb"))
        # self.testingData = pickle.load(open(self.testingDataPath, "rb"))
        # self.trainingData = pickle.load(open(self.trainingDataPath, "rb"))
        # print(self.threadDict)

    def getThreads(self, userName) :
        userId = self.userDict.get(userName, 10000)
        threadList = []
        if userId != 10000 :
            print(userId)
            for key, value in self.threadDict.items() :
                if userId in value :
                    threadList.append(key+1)
        return threadList


    # def getThreadIds(self) :
    #     threadList = []
    #     for key, value in self.threadDict.items() :
    #         threadList.append(key)
    #     return threadList

        
    # def getPopularThreads(self) :
    #     threadList = []
    #     tempDict = {}
    #     for key, values in self.threadDict.items() :
    #         tempDict[key] = len(values)
        
    #     sorted_thread = sorted(tempDict.items(), key = lambda kv: kv[1])
    #     print(len(sorted_thread))




# ob = Recom()
# ob.getThreads('Paul Wise')
# ob.getPopularThreads()
