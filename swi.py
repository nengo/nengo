"""
Simple Web Interface
Version: 2005-04-04
Author: Terry Stewart terry.stewart@gmail.com http://terrystewart.ca

This software is released under the GNU General Public License.
See http://www.gnu.org/copyleft/gpl.html for more details.


Simple Web Interface is a fast way to create Python programs that serve
  dynamic web pages, without all that tedious mucking about with CGI scripts.
  You simply define a class, and the methods in that class will be called to
  create the various web pages.

An HTTP request for http://server/page?a=val1&b=val2 will result in a call to
    def swi_page(self,a,b)
  and 'a' will get the value 'val1' and 'b' will be 'val2'.  You can also
  define defaults for these arguments, and it will use those if they
  are unspecified.  This method will also be called for a request like this:
    http://server/page/val1/val2
  There is no limitation on the number of arguments.  The system supports
  both GET and POST (and multipart POST, so file uploading will work).

The system also supports a cookie-based login system.  You can define users
  with swi.addUser(name,password).  You can then check for logins by adding
  this to the beginning of your methods:
    if self.user==None:
      return self.createLoginForm()
  You can log out by causing a call to self.logOut().

For further documentation, see http://terrystewart.ca/swi.html


**** History
- 2005-08-06: Added support for <select multiple>
- 2005-07-31: Added serveFiles and serverDirs to act as webserver
- 2005-05-28: Added asynch option to start() to allow multiple requests
- 2005-04-04: Added default favicon.ico icon
- 2005-04-03: Initial Release
"""

import BaseHTTPServer
import traceback
import random,string,os,StringIO,mimetools,multifile
import re
import webbrowser,thread
import mimetypes

import SocketServer
class HTTPServer(SocketServer.ThreadingMixIn,BaseHTTPServer.HTTPServer):
  pass


testingUser=None

def makeDBFromMultipart(data):
  db={}
  boundary=data[:data.find('\n')+1].strip()
  n=len(boundary)

  i=0
  while 1:
    j=data.find(boundary,i+n)
    if j<0:
      break

    content=data[i+n+2:j-2]
    m=re.search(r'name="([^"]+)"',content)
    name=m.group(1)
    k=content.find('\r\n\r\n')+4
    val=content[k:]

    m=re.search(r'filename="([^"]+)"',content[:k])
    if m!=None:
      filename=m.group(1)
      val=(filename,val)

    db[name]=val

    i=j
  print db
  return db


def makeDBFromLine(data):
  if '\n' in data:
    data,x=data.split('\n',1)
    data=data.strip()
  db={}
  for line in data.split('&'):
    if '=' in line:
      key,val=line.split('=',1)
      val=fixText(val)
      key=fixText(key)
      if key in db:
        v=db[key]
        if type(v)!=type([]): v=[v]
        v.append(val)
        val=v
      db[key]=val
  return db

def fixText(val):
  val=val.replace('+',' ')
  i=0
  while i<len(val)-2:
    if val[i]=='%' and val[i+1] in '1234567890abcdefABCDEF' and val[i+2] in '1234567890abcdefABCDEF':
      c=chr(int(val[i+1:i+3],16))
      val=val[:i]+c+val[i+3:]
    i+=1
  return val

favicon='\x00\x00\x01\x00\x01\x00\x10\x10\x00\x00\x01\x00\x18\x00h\x03\x00\x00\x16\x00\x00\x00(\x00\x00\x00\x10\x00\x00\x00 \x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x009\x13\x115!\x0f/Z\x0f$\x81\x0e(\x9e\x10"\xbb &\xd7\x1c,\xe1\x1a9\xe2\x17a\xe3\x1au\xe4\x18\x93\xec!\xb5\xe6\x18\xcf\xe6\x18\xe7\xe8\'\xed\xdd62\x1c\x17\x185\x1e\x15W&\x18x+\x1f\x9e;\x1b\xbd@\x17\xd9F(\xe4?7\xe59X\xe99u\xe61\x8f\xe15\xaf\xd8)\xc6\xcf\x1a\xe1\xcd(\xe4\xc1;\'\x1f=\x173:\x16LG\x18oK"\x90U\x1c\xb6e\x1c\xdfy0\xe9kH\xeac`\xe0]v\xdeQ\x8a\xc89\x9b\xb3%\xb3\xb5\x1b\xca\xa8\x1f\xf5\x9c)0\x1eS#+P\x16Ed\x14br\x19\x90{\x1b\xbd\x93#\xe2\x8d7\xe3\x83K\xe3\x84e\xdbto\xc5Uq\xa7,\x8d\xa4 \xa3\x9b\x18\xc4\x92\x1a\xe1\x8c\x1e)\x13k\'$s%G\x83\x16Z\x8f(u\x9c \x9a\xa88\xca\xaaJ\xcf\x96`\xce\x8ci\xbcxm\xa7Nd\x8b)}\x81!\x8au\x19\xbbi"\xdei\x18\'\x14\x8b.\x1e\x8aA,\x89BM\x8bDk\x98E~\x9eL\xa1\xab[\xaa\xa1^\xaf\x94`\xa6}[\x84J_j,f[\x1d\x83R\x1a\xa3L\x13\xcdJ\x18\x1d\x17\xa4? \xa3X!\x8e]/\x92XW\x97Sv\xa8^\x94\xabb\x99\xa0g\x9f\x94`\x89tSgJTQ2g@$\x7f8\x13\x9f8\x0b\xc7:\t\x13\x12\xb62\x1c\xb1d\x1a\x9a}$\x9cvK\xa2qm\xb5s\x8e\xb0m\x96\x9fw\x99\x98n}yf]YZFEr1:\x8a,\'\xaa.\x16\xde.\x10\x17\x1d\xce1\x1f\xd0^\x1c\xc1\x8b"\xae\x96@\xae\x8c[\xaf\x8d\x7f\xa8\x81\x8a\x94\x7f\x8d\x89}tqyZ]x7Qv\x1fK\x8e\x12>\xba\x1a&\xdd#\x1b\x15/\xda0-\xe0W#\xe2\x85,\xc9\x9fG\xbd\xa9g\xb9\xa4\x81\xa3\x93\x8e\x8b\x94\x93~\x8f\x80m\x95\\[\x94<T\x92%V\x9c\x0eI\xc0\x0bH\xdf\x1b9\x17D\xe1&E\xe8_>\xed\x9dN\xeb\xafb\xd5\xbc\x86\xcb\xaf\x94\xae\x99\xa4\x9a\x9d\xad\x8b\xad\x9ao\xc1{^\xbeW\\\xbeEZ\xb5\x15O\xdb\x14Q\xdd\x14e\x1a^\xdf0i\xddal\xe5\x97o\xf2\xb3\x85\xe8\xb6\x9e\xd4\xa8\xb4\xae\xa3\xbe\x96\xac\xc5\x89\xbd\xb9h\xd9\x90\\\xdfs[\xdae\\\xe0Ju\xd2 \x7f\xd4\x1b{!v\xde3}\xe7^\x8e\xdc\x80\x8c\xde\xa5\xaa\xe9\xad\xb5\xe4\xaa\xc9\xaa\xb2\xd6\x94\xbd\xdf\x81\xd7\xd1p\xdf\xabo\xdb\x8dp\xe8m{\xdeJ\x8a\xd61\x92\xce\x19\x9a\x1b\x95\xe9=\xa6\xe5W\xaa\xden\xb1\xde\x84\xc7\xe6\xa2\xca\xdc\xb2\xe0\xac\xbe\xe7\xa3\xcf\xe9\x9b\xe3\xd8\x94\xeb\xb0\x96\xe8\x97\x92\xe7}\x9a\xe7N\xa3\xd6$\xab\xcf\x11\xac\x1b\xc6\xe0&\xc9\xe9G\xd0\xeaj\xd0\xe2z\xda\xe0\x8a\xe1\xdd\xae\xe9\xcd\xc9\xeb\xc2\xe3\xe5\xc8\xf2\xd9\xdd\xe9\xbe\xcb\xeb\x9b\xbe\xe8n\xc6\xdeM\xcd\xd80\xd6\xd2\x14\xcd\x07\xe0\xdc"\xe8\xe4.\xeb\xeaJ\xea\xd8t\xe2\xd0\x85\xe7\xc9\x9a\xeb\xca\xc7\xed\xe1\xda\xe4\xeb\xec\xd3\xf5\xed\xbf\xee\xe4\x94\xed\xdcj\xe6\xd5Q\xeb\xd6>\xe5\xc5\x1b\xf0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'


currentCookies={}
passwords={}

def addUser(id,pwd):
  passwords[id]=pwd

class SimpleWebInterface(BaseHTTPServer.BaseHTTPRequestHandler):
  server_version='SimpleWebInterface/1.0'
  serveFiles=[]
  serveDirs=[]

  pendingHeaders=[]

  def addHeader(self,key,value):
    self.pendingHeaders.append((key,value))

  def getUserFromId(self,id,pwd):
    if (id in passwords and pwd==passwords[id]) or len(passwords)==0:
      while 1:
        value=''.join([random.choice(string.ascii_lowercase) for i in range(20)])
        if value not in currentCookies: break
      self.addHeader('Set-Cookie','id='+value)
      currentCookies[value]=id
      return id
    return None
  def getUserFromCookie(self):
    if self.headers.has_key('cookie'):
      cookie=self.headers['cookie']
      for c in cookie.split(';'):
        if '=' in c:
          name,val=c.split('=',1)
          name=name.strip()
          val=val.strip()
          if name=='id' and val in currentCookies:
            return currentCookies[val]
    if testingUser!=None: return testingUser
    return None
  def logOut(self):
    for k,v in currentCookies.items():
      if v==self.user:
        del currentCookies[k]
    self.addHeader('Set-Cookie:','id=0')
  def createLoginForm(self,target=None,**data):
    if self.attemptedLogin:
      message='Invalid password.  Try again.'
    else:
      message='Enter your user name and password:'
    data=''.join(['<input type=hidden name=%s value=%s>'%kv for kv in data.items() if kv[1]!=None])

    if target==None:
      target=self.currentArgs[0]

    # for some reason, using POST here doesn't work under IE6, but does
    #  under Firefox and Safari.
    method='GET'
    if self.headers.has_key('User-Agent'):
      agent=self.headers['User-Agent']
      if 'MSIE' not in agent: method='POST'

    return """<form action="%s" method=%s>%s<br>
    <label for=swi_id>Username: </label><input type=text name=swi_id><br>
    <label for=swi_pwd>Password: </label><input type=password name=swi_pwd>
    %s
    <input type=submit value="Log In">
    </form>"""%(target,method,message,data)


  def swi_favicon_ico(self):
    return ('image/ico',favicon)


  def do_POST(self):
    del self.pendingHeaders[:]
    data=self.rfile.read(int(self.headers['Content-Length']))
    if 'multipart/form-data' in self.headers['Content-Type']:
      db=makeDBFromMultipart(data)
    else:
      db=makeDBFromLine(data)
    if '?' in self.path:
      self.path,data=self.path.split('?',1)
      db2=makeDBFromLine(data)
      db.update(db2)

    if self.path=='': self.path='/'
    self.path=fixText(self.path)
    args=self.path[1:].split('/')
    self.handleRequest(args,db)

  def do_GET(self):
    del self.pendingHeaders[:]
    if self.path=='': self.path='/'
    self.path=fixText(self.path)

    db={}
    if '?' in self.path:
      self.path,data=self.path.split('?',1)
      db=makeDBFromLine(data)

    args=self.path[1:].split('/')
    self.handleRequest(args,db)

  def handleRequest(self,args,db):
    self.currentArgs=args
    if args[0]=='':
      command='swi'
    else: command='swi_%s'%args[0]
    command=command.replace('.','_')

    if hasattr(self,command):
      ctype='text/html'
      self.user=self.getUserFromCookie()
      self.attemptedLogin=False
      if 'swi_id' in db and 'swi_pwd' in db:
        self.attemptedLogin=True
        self.user=self.getUserFromId(db['swi_id'],db['swi_pwd'])
        del db['swi_id']
        del db['swi_pwd']
      elif len(passwords)==0:
        self.user=self.getUserFromId('','')

      try:
        text=getattr(self,command)(*args[1:],**db)
      except:
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        print >>self.wfile,"<html><body><pre>"
        text=StringIO.StringIO()
        traceback.print_exc(file=text)
        text=text.getvalue()
        text=text.replace('<','<')
        text=text.replace('>','>')
        print >>self.wfile,"%s</pre></body></html>"%text
        print text
      else:
        if type(text)==type(()):
          ctype,text=text
        self.send_response(200)
        self.send_header('Content-type',ctype)
        for k,v in self.pendingHeaders:
          self.send_header(k,v)
        self.end_headers()
        print >>self.wfile,text
    elif self.path[1:] in self.serveFiles:
      self.sendFile(self.path[1:])
    elif self.path=='/robots.txt':
      self.send_response(200)
      self.send_header('Content-type','text/text')
      self.end_headers()
      print >>self.wfile,"User-agent: *\nDisallow: /"
    else:
      for d in self.serveDirs:
        if self.path[1:].startswith(d+'/'):
          self.sendFile(self.path[1:])
          return
      self.send_response(200)
      self.send_header('Content-type','text/html')
      self.end_headers()
      print >>self.wfile,"<html><body>Invalid request:<pre>args=%s</pre><pre>db=%s</pre></body></html>"%(args,db)
  def sendFile(self,path):
    self.send_response(200)
    type,enc=mimetypes.guess_type(path)
    self.send_header('Content-type',type)
    self.send_header('Content-encoding',enc)
    self.end_headers()
    print >>self.wfile,file(path,'rb').read()




def start(klass, port=80, asynch=True, addr=''):
    if asynch:
        server=HTTPServer((addr,port),klass)
    else:
        server=BaseHTTPServer.HTTPServer((addr,port),klass)
    server.serve_forever()

def browser(port=80):
    thread.start_new_thread(webbrowser.open,('http://localhost:%d'%port,))


if __name__=='__main__':
  class Demo(SimpleWebInterface):

    # http://localhost:8080/
    def swi(self):
      return 'Hello world!'

    # http://localhost:8080/test1?a=data
    # http://localhost:8080/test1/data
    def swi_test1(self,a='value'):
      return 'The value of a is "%s"'%a

    # http://localhost:8080/test2?a=data&b=data2
    # http://localhost:8080/test2/data/data2
    # http://localhost:8080/test2?b=data2
    def swi_test2(self,a='value',b='value'):
      return 'The value of a is "%s" and b is "%s"'%(a,b)

    # http://localhost:8080/page
    def swi_page(self):
      if self.user==None:
        return self.createLoginForm()
      return '''
      Hello, %s.
      <p>Click <a href="otherPage">here</a> to go to another page.
      <p>Click <a href="logout">here</a> to log out.
      '''%self.user

    # http://localhost:8080/otherPage
    def swi_otherPage(self):
      if self.user==None:
        return self.createLoginForm()
      return '''
      You are now at the other page, %s.
      <p>Click <a href="page">here</a> to go back to the first page.
      <p>Click <a href="logout">here</a> to log out.
      '''%self.user

    # http://localhost:8080/logout
    def swi_logout(self):
      self.logOut()
      return '''
      You are now logged out.
      <p>Click <a href="page">here</a> to the first page.
      <p>Click <a href="otherPage">here</a> to go to another page.
      '''
  addUser('terry','password')
  start(Demo,8080)

