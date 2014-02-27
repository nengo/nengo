"""
PyTag
Version: 2005-06-11
Author: Terry Stewart terry.stewart@gmail.com http://terrystewart.ca

This software is released under the GNU General Public License.
See http://www.gnu.org/copyleft/gpl.html for more details.

This is a utility for easily generating valid XHTML from within
a Python program.  It is inspired by Nevow's <http://nevow.com> 'Stan'
module, which is pretty much exactly the same as this one.  I just
wrote my own version because I felt like it.

Here is a quick example of what you can do:

from pytag import T
print T.html[
  T.head[T.title["This is the title"]],
  T.body[
    T.h1["Heading"],
    T.hr,
    T.p(align='center')[
      "Hello and welcome to my ",
      T.em["nifty"],
      " web page."
      ]
    ]
  ]

As you can see, this is a rather odd (ab)use of the [] and () syntax.
You create tags by saying T.tagname (using any tagname you want).

You set the attributes of the tag by "calling" it.
   T.tag(name=value, othername=othervalue)
You set the contents of the tag using the square brackets.
   T.tag['some stuff',T.othertag,'can go here']
You can do them both at the same time, so you can define an
html link like this:
   T.a(href='page.html')['click here']
or
   T.a['click here'](href='page.html')

You can even do this to simplify the syntax for common tags:

from pytag import *
print html[
  head[title["This is the title"]],
  body[
    h1["Heading"],
    hr,
    p(align='center')[
      "Hello and welcome to my ",
      em["nifty"],
      " web page."
      ]
    ]
  ]


If you add things more than once, they get appended together

  myList=T.ol
  for i in range(10):
    mylist[T.li['this is item #%d'%i]]
  print myList

You can also use the Python list comletion syntax to do something
like that

  print T.ol[[T.li['this is item #%d'%i for i in range(10)]]]

(note the currently-required double-square brackets in that example.
It is possible that that will become optional when list completions
become more general in newer versions of Python)

One other capability is to create template tags.  These are just like
normal tags, but when you add things to them they create a new
instance of the tag, rather than accumulating.

  myP=T.p(align='center',size='+1')
  myP.lock()

  print T.body[
    T.h1['Header'],
    myP['Here is paragraph 1'],
    myP['Here is paragraph 2'],
    ]

For further documentation, see http://terrystewart.ca/pytag.html

**** History
- 2005-06-11: Initial Release
"""



import copy

def _flatten(args):
  for arg in args:
    if type(arg) in (type(()),type([])):
      for x in arg:
        for f in _flatten(x): yield f
    else: yield arg

class Tag:
  def __init__(self,name):
    self.name=name
    self.attr={}
    self.content=[]
    self.locked=False
  def lock(self):
    self.locked=True
    return self
  def __getitem__(self,content):
    if self.locked:
      self=copy.deepcopy(self)
      self.locked=False
    if type(content) not in (type(()),type([])):
      self.content.append(content)
    else:
      self.content.extend(_flatten(content))
    return self
  def __call__(self,**attr):
    if self.locked: 
      self=copy.deepcopy(self)
      self.locked=False
    for k,v in attr.items():
      v=str(v)
      v='"%s"'%v
      self.attr[k]=v
    return self
  def __str__(self):
    attr=''
    if len(self.attr)>0:
      attr=' '+' '.join(['%s=%s'%kv for kv in self.attr.items()])
    if len(self.content)==0:
      return '<%s%s />'%(self.name,attr)
    else:
      content=''.join([unicode(x) for x in self.content])
      return '<%s%s>%s</%s>'%(self.name,attr,content,self.name)

class _TagMaker:
  def __getattr__(self,key):
    return Tag(key)
T=_TagMaker()


_tagList="""html head title body p em strong ol li ul blockquote br
center cite code dd dt dl h1 h2 h3 h4 h5 h6 hr img meta small big a
form input option select optgroup textarea style
sub sup tt
table tr td th
""".split()
for t in _tagList:
  exec('%s=T.%s;%s.lock()'%(t,t,t))


def _pytagtest():
  print p['Hello world']

  print body[T.p['Hello']]

  print html[
    head[T.title["This is the title"]],
    body[
      h1["Heading"],
      p(align='center')[
        "Hello and welcome to my ",
        em["nifty"],
        " web page."
        ],
      br,
      "Click ",a(href="page.html")['here']," to go somewhere else."
      ]
    ]

  print ol[[li[x] for x in range(10)]]

  cp=p(align='center')
  cp.lock()
  print p[cp(a='b')['p 1'],cp['p 2']]



if __name__=='__main__':
  _pytagtest()
