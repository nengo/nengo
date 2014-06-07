*********
nengo_gui
*********

HTML5 graphical interface for Nengo. This works best in the
`Chrome Browser <https://www.google.com/intl/en_ca/chrome/browser/>`_. Really.  Get Chrome.

Installation
============

To get started, run the following in a terminal window.

.. code:: bash

   # Start in some directory that will contain source code.
   # A common place is $HOME/Code.

   # --- Installing Nengo 1.4

   # Download the latest dev version of Nengo 1.4
   wget http://ctnsrv.uwaterloo.ca:8080/jenkins/job/Nengo/lastSuccessfulBuild/artifact/nengo-latest.zip
   # Unzip it
   unzip nengo-latest.zip
   # Rename the extracted folder to nengo-latest
   mv nengo-[^l]* nengo-latest
   # Remove nengo-latest.zip
   rm nengo-latest.zip

   # --- Installing nengo_gui

   # Clone the nengo_gui project
   git clone https://github.com/ctn-waterloo/nengo_gui.git
   # Install it
   cd nengo_gui
   pip install rpyc
   pip install --editable .

Interactive usage
=================

To run models and see the data generated as the model runs,
you must first start the ``javaviz`` interactive visualizer.

   #double click on javaviz-server.bat (or just javaviz-server) in the nengo-latest folder

Basic usage
===========

To view a Nengo model, run ``nengo_gui`` from the command line.  You can load scripts from the nengo_gui/scripts
folder from within the gui.

Using ``nengo_gui``
-------------------

If you're using ``nengo_gui``, then click the top right button to run
the interactive visualizer. (You can do this right away with the default network that loads
as a quick example).  If you don't
see the visualizer window, it may be hidden in the background; find it
in your OS's window bar.

In the interactive visualizer, right click on the background of the
visualizer to pick things to see; right click on the things to pick
data to plot. The rest is all very intuitive :smiley:. 

Alternative usage
-----------------
You can pass nengo_gui a script to visualize, if desired.

.. code:: bash

   nengo_gui my_nengo_script.py

Using Javaviz in a script
-------------------------

You can run ``javaviz`` straight from a script.
Before

.. code:: python

   sim = nengo.Simulator(model)
   sim.run(2.0)

After

.. code:: python

   if 0:  # Change to 1 to run non-interactively
       sim = nengo.Simulator(model)
       sim.run(2.0)
   if 1:  # Change to 0 to run non-interacively
       from nengo_gui import javaviz
       jv = javaviz.View(model)
       sim = nengo.Simulator(model)
       jv.update_model(sim)
       jv.view()
       sim.run(10000) #run for 10000 seconds (effectively forever)
       
