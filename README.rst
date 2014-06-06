nengo_gui
=========

HTML5 graphical interface for Nengo.

Starting from scratch:

1. clone this ``nengo_gui`` repository from githumb

2. Install ``nengo_gui`` from the command-line in its directory:

.. code:: bash

   python setup.py develop
   
3. Download the latest development version of Nengo 1.4 and unzip it: 
http://ctnsrv.uwaterloo.ca:8080/jenkins/job/Nengo/lastSuccessfulBuild/artifact/nengo-latest.zip
   
4. Make sure you have rpyc installed: ``pip install rpyc`` on the command line if you don't.

5. In the Nengo 1.4 directory run ``nengo-cl python/rpyc/scripts/rpyc_classic.py``
(Now the javaviz interactive visualizer server is running, which let's you see the model output)

6. From the command line type ``nengo_gui``
(Now the gui server is running to let you see and edit the model)

7. To connect the model to the interactive visualizer, put this at the bottom of your script:

.. code:: bash

   import javaviz   
   javaviz.View(model)   # this is all you need to do!

8. The button in the top right of the Chrome window should light up.  Click it to interact with the running 
model. Right click on the background of the visualizer to pick things to see, right click on the things to pick
data to plot. The rest is all very intuitive :).


