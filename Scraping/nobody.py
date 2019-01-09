from bs4 import BeautifulSoup
import requests
from pprint import pprint
import re
import os
import requests
from bs4 import BeautifulSoup
import urllib
import threading

page1 = requests.get("https://lists.debian.org/debian-devel/2017/01/msg00401.html")
soup1 = BeautifulSoup(page1.text, 'html.parser')
fname = str(1) + '.txt'

file = open(fname,'w')
#x += 1



div = soup1.find('div')

if div is not None:
    print(div.text)



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
