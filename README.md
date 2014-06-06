nengo_gui
=========

HTML5 graphical interface for Nengo.  This works best in the <a href="
https://www.google.com/intl/en_ca/chrome/browser/?&brand=CHMA&utm_campaign=en&utm_source=en-ha-ca-bk&utm_medium=ha">
Chrome Browser</a>.

Starting from scratch:

#. clone this ``nengo_gui`` repository from githumb

#. Install ``nengo_gui`` from the command-line in its directory:

```
   python setup.py develop
```
#. Download the latest development version of Nengo 1.4 and unzip it: 
http://ctnsrv.uwaterloo.ca:8080/jenkins/job/Nengo/lastSuccessfulBuild/artifact/nengo-latest.zip

#. Make sure you have rpyc installed: ``pip install rpyc`` on the command line if you don't.

#. In the Nengo 1.4 directory run ``nengo-cl python/rpyc/scripts/rpyc_classic.py``
(Now the javaviz interactive visualizer server is running, which let's you see the model output)

#. From the command line type ``nengo_gui``
(Now the gui server is running to let you see and edit the model)

#. To connect the model to the interactive visualizer, put this at the bottom of your script:

```
   import javaviz   
   javaviz.View(model)   # this is all you need to do!
```

#. The button in the top right of the Chrome window should light up.  Click it to interact with the running 
model. Right click on the background of the visualizer to pick things to see, right click on the things to pick
data to plot. The rest is all very intuitive :).


