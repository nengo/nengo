*********
nengo_gui
*********

HTML5 graphical interface for Nengo. This works best in the
`Chrome Browser <https://www.google.com/intl/en_ca/chrome/browser/>`_, Safari, or Opera.
We are still working on some compatibility issues in Firefox.

Installation
============

The Nengo GUI system consists of two parts: a browser-based code editor that shows you the
structure of the network you are building as you create it, and a separate Java-based
visualizer that shows the behaviour of the model as you run it.

To install the editor, you need to download and install the code in this repository.
For most operating systems, here are the commands needed:

.. code:: bash
   git clone https://github.com/ctn-waterloo/nengo_gui.git
   cd nengo_gui
   python setup.py develop
   
Note that this assumes you already have `Nengo <https://github.com/ctn-waterloo/nengo/>`_ installed.

The editor will work on its own, but it is more useful with the visualizer as well.
To install this, we need to download the old Java-based version of Nengo.  Here
are the instructions:

 - Download `this file <http://ctnsrv.uwaterloo.ca:8080/jenkins/job/Nengo/lastSuccessfulBuild/artifact/nengo-latest.zip>`_.
 - Unzip the file

Note that you do have to have `Java <http://java.com/>`_ installed on your computer for this to work.  Finally, you
need to install `RPyC <http://rpyc.readthedocs.org/>`_, which allows the editor and the visualizer to communicate.

.. code:: bash
   pip install rpyc
   


Running Nengo GUI
=================

First, you must start the javaviz-server, a program that will sit in the background and handle the
visualization system.  Do this by going to the directory where you unzipped the old Java-based version
of nengo and running javaviz-server.bat (on Windows) or javaviz-server (on Unix/Mac).  A text window should
pop up and a line like ``INFO:SLAVE/18812:server started on [127.0.0.1]:18812`` should appear.

Finally, we run the Nengo GUI using the command ``nengo_gui``.  You can run this from the command prompt,
or by double-clicking on the ``nengo_gui`` executable.  On Windows, this is likely in a directory such as
``C:\Python27\Scripts\nengo_gui.exe``.  





Basic usage
===========

To view a Nengo model, run ``nengo_gui`` from the command line.  You can load scripts from the nengo_gui/scripts
folder from within the gui.

Using ``nengo_gui``
-------------------

If you're using ``nengo_gui``, then click the top right button to run
the interactive visualizer. (You can do this right away with the default network that loads, 
as a quick example).  If you don't
see the visualizer window, it may be hidden in the background; find it
in your OS's window bar.

In the interactive visualizer, right click on the background of the
visualizer to pick things to see; right click on the things to pick
data to plot. Items can be dragged to be moved around and resized
by the edges.  Plots can be customized by right-clicking on them.
The play button in the bottom-right starts the simulation.

Alternative usage
-----------------
You can pass nengo_gui a script to visualize, if desired.

.. code:: bash

   nengo_gui my_nengo_script.py

Advanced Usage: Javaviz without Nengo GUI
-------------------------------------------

You can also just use the visualizer and bypass the editor completely
(if you don't want to see your network as you build it).  Here is what
you would do in a normal Python script (i.e. one where you are NOT
using the Nengo GUI editor).

.. code:: python

   import nengo_gui
   jv = nengo_gui.javaviz.View(model)
   sim = nengo.Simulator(model)
   jv.update_model(sim)
   jv.view()
   while True:
       sim.run(1) 
       
