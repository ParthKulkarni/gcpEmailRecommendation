from bs4 import BeautifulSoup
import requests
from pprint import pprint
import re
import os
import requests
from bs4 import BeautifulSoup
import urllib
import threading
 
url = 'https://lists.debian.org/debian-boot/'
year = 2018
month = 1
url1 = 'threads.html'

def getLinks(url, url1):
    links = []
    try :
        html_page = urllib.request.urlopen(url + url1)
        soup = BeautifulSoup(html_page, "lxml")
        links1 = [a.get('href') for a in soup.find_all('a', href=True)]
		# links = []
        for link in links1 :
            if link.startswith('msg') :
                links.append(url + link)
        print(len(links))
        return links
    except urllib.error.HTTPError as err:
        if err.code == 404 :
            None
        else :
            raise

os.mkdir(str(year))

def extract_month(month):

#for x in range(12) :
    print("Extracting month "+str(month))
    links = []
#    month += 1
    base_url = ''
    if month < 10 :
        base_url = url + str(year) + '/0' + str(month) + '/'
    else :
        base_url = url + str(year) + '/' + str(month) + '/'
	# print(base_url + url1)
    links = getLinks(base_url, url1)
    for y in range (2, 5) :
        extra_url = 'thrd' + str(y) + '.html'
		# print(base_url + extra_url)
        temp_link = getLinks(base_url, extra_url)
        if temp_link is not None:
            links = links + temp_link
            print("Extracting from "+ str(y) + " " + base_url + extra_url)
	
    

    # # Visit each link in the list and scrape data. Data written to file
    x = 1
    # file = open('debian.txt','w')

    
    os.mkdir(str(year)+"/"+str(month))

    for l in links:
        page1 = requests.get(l)
        soup1 = BeautifulSoup(page1.text, 'html.parser')
        fname = str(x) + '.txt'
        
        file = open(str(year)+"/"+str(month)+"/"+fname,'w')
        x += 1

        li = soup1.find_all('li')
        dict = {}
        for i in range(len(li)):
            st = li[i].text
            s  = str(st).split(': ', 1)
            if s[0] == 'Cc':
                continue
            if len(s) < 2:
                break
            dict[s[0]] = s[1]
            #if s[0] == 'References':
            #    break
        dict['Message-id']=dict['Message-id'][5:-1]
        for n,m in dict.items():
            #print(n + ' : '+m+'\n')
            file.write(n + ' : '+m+'\n')
        file.write("\n\n")
        body = ''
        pre = soup1.find_all('pre')
        tt  = soup1.find_all('tt')
        if len(pre)>0:
            #print(len(pre))
            body = pre[0].text
        for t in tt:
            body += t.text
        if len(pre) > 1:
            body += pre[-1].text
        body = re.sub('\n+', '\n',body)
        body = body.lstrip()
        body = body.rstrip()
        #print(body)
        file.write(body)
        file.close()
    
    
if __name__ == "__main__": 
    # creating thread 
    t1 = threading.Thread(target=extract_month, args=(1,)) 
    t2 = threading.Thread(target=extract_month, args=(2,))
    t3 = threading.Thread(target=extract_month, args=(3,))
    t4 = threading.Thread(target=extract_month, args=(4,))
    t5 = threading.Thread(target=extract_month, args=(5,))
    t6 = threading.Thread(target=extract_month, args=(6,))
    t7 = threading.Thread(target=extract_month, args=(7,))
    t8 = threading.Thread(target=extract_month, args=(8,))
    t9 = threading.Thread(target=extract_month, args=(9,))
    t10 = threading.Thread(target=extract_month, args=(10,))
    t11 = threading.Thread(target=extract_month, args=(11,))
    t12= threading.Thread(target=extract_month, args=(12,))

  
    
    t1.start() 
    t2.start() 
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()
    t10.start()
    t11.start()
    t12.start()

  
    t1.join() 
    t2.join() 
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()
    t8.join()
    t9.join()
    t10.join()
    t11.join()
    t12.join()

  
    print("Done!") 
