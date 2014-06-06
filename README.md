# nengo_gui

HTML5 graphical interface for Nengo.  This works best in the [Chrome Browser]
(https://www.google.com/intl/en_ca/chrome/browser/?&brand=CHMA&utm_campaign=en&utm_source=en-ha-ca-bk&utm_medium=ha).

Starting from scratch:

1. clone this ``nengo_gui`` repository from githumb

1. Install ``nengo_gui`` from the command-line in its directory: ``python setup.py develop``

1. Download the latest development version of Nengo 1.4 and unzip it: 
http://ctnsrv.uwaterloo.ca:8080/jenkins/job/Nengo/lastSuccessfulBuild/artifact/nengo-latest.zip

1. Make sure you have rpyc installed: ``pip install rpyc`` on the command line if you don't.

1. In the Nengo 1.4 directory run ``nengo-cl python/rpyc/scripts/rpyc_classic.py``
(Now the javaviz interactive visualizer server is running, which let's you see the model output)

1. From the command line type ``nengo_gui``
(Now the gui server is running to let you see and edit the model)

1. To connect the model to the interactive visualizer, put this at the bottom of your script:

```
   import javaviz   
   javaviz.View(model)   # this is all you need to do!
```
<ol start=8>
<li>The button in the top right of the Chrome window should light up.  Click it to interact with the running 
model. Right click on the background of the visualizer to pick things to see, right click on the things to pick
data to plot. The rest is all very intuitive :).</li>
<ol>


