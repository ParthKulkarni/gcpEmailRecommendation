import random
from ast import literal_eval

class model :
    def __init__(self) :
        self.li_name = []
        self.user_prediction = {}
        self.f = open("debug.txt", "r")  
        self.user = literal_eval(self.f.read())


    def generate_random_list(self) :
        li = []
        for x in range (10) :
            li.append((random.randint(0, 100)) % 2)

        for x in range (10) :
            if li[x] == 1 :
                self.li_name.append(x)
        #return self.li_name

	
    def map_user_to_thread(self) :
        temp_thread_id = 50
        print(self.li_name)
        for name_id in self.li_name :
            list = self.user_prediction.get(name_id, [])
            list.append(temp_thread_id)
            self.user_prediction[name_id] = list
        print(self.user_prediction)         
		
ob = model()
ob.generate_random_list()
ob.map_user_to_thread()
ob.generate_random_list()
ob.map_user_to_thread()