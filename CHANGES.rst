***************
Release History
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - API changes
   - Improvements
   - Behavioural changes
   - Bugfixes
   - Documentation

2.1.0 (unreleased)
==================

**API changes**

- Spiking ``LIF`` neuron models now accept an additional argument,
  ``min_voltage``. Voltages are clipped such that they do not drop below
  this value (previously, this was fixed at 0).
  (`#666 <https://github.com/nengo/nengo/pull/666>`_)
- ``Process`` objects can now be passed directly as node outputs,
  making them easier to use. The ``Process`` interface is also improved
  and is currently the same as the ``Synapse`` interface. However,
  further improvements are pending, and the current implementation
  SHOULD NOT BE RELEASED!
  (`#652 <https://github.com/nengo/nengo/pull/652>`_)
- The ``PES`` learning rule no longer accepts a connection as an argument.
  Instead, error information is transmitted by making a connection to the
  learning rule object (e.g.,
  ``nengo.Connection(error_ensemble, connection.learning_rule)``.
  (`#344 <https://github.com/nengo/nengo/issues/344>`_,
  `#642 <https://github.com/nengo/nengo/pull/642>`_)
- The ``modulatory`` attribute has been removed from ``nengo.Connection``.
  This was only used for learning rules to this point, and has been removed
  in favor of connecting directly to the learning rule.
  (`#642 <https://github.com/nengo/nengo/pull/642>`_)
- Connection weights can now be probed with ``nengo.Probe(conn, 'weights')``,
  and these are always the weights that will change with learning
  regardless of the type of connection. Previously, either ``decoders`` or
  ``transform`` may have changed depending on the type of connection;
  it is now no longer possible to probe ``decoders`` or ``transform``.
  (`#729 <https://github.com/nengo/nengo/pull/729>`_)
- The ``spa.State`` object replaces the old ``spa.Memory`` and ``spa.Buffer``.
  These old modules are deprecated and will be removed in 2.2.
  (`#796 <https://github.com/nengo/nengo/pull/796>`_)
- A version of the AssociativeMemory SPA module is now available as a
  stand-alone network in ``nengo.networks``. The AssociativeMemory SPA module
  also has an updated argument list.
  (`#702 <https://github.com/nengo/nengo/pull/702>`_)
- The ``Product`` and ``InputGatedMemory`` networks no longer accept a
  ``config`` argument. (`#814 <https://github.com/nengo/nengo/pull/814>`_)
- The ``EnsembleArray`` network's ``neuron_nodes`` argument is deprecated.
  Instead, call the new ``add_neuron_input`` or ``add_neuron_output`` methods.
  (`#868 <https://github.com/nengo/nengo/pull/868>`_)
- The ``nengo.log`` utility function now takes a string ``level`` parameter
  to specify any logging level, instead of the old binary ``debug`` parameter.
  Cache messages are logged at DEBUG instead of INFO level.
  (`#883 <https://github.com/nengo/nengo/pull/883>`_)

**Behavioural changes**

- The sign on the ``PES`` learning rule's error has been flipped to conform
  with most learning rules, in which error is minimized. The error should be
  ``actual - target``. (`#642 <https://github.com/nengo/nengo/pull/642>`_)
- The ``PES`` rule's learning rate is invariant to the number of neurons
  in the presynaptic population. The effective speed of learning should now
  be unaffected by changes in the size of the presynaptic population.
  Existing learning networks may need to be updated; to achieve identical
  behavior, scale the learning rate by ``pre.n_neurons / 100``.
  (`#643 <https://github.com/nengo/nengo/issues/643>`_)
- The ``probeable`` attribute of all Nengo objects is now implemented
  as a property, rather than a configurable parameter.
  (`#671 <https://github.com/nengo/nengo/pull/671>`_)
- Node functions receive ``x`` as a copied NumPy array (instead of a readonly
  view).
  (`#716 <https://github.com/nengo/nengo/issues/716>`_,
  `#722 <https://github.com/nengo/nengo/pull/722>`_)
- The SPA Compare module produces a scalar output (instead of a specific
  vector).
  (`#775 <https://github.com/nengo/nengo/issues/775>`_,
  `#782 <https://github.com/nengo/nengo/pull/782>`_)

**Improvements**

- Added ``PES.pre_tau`` attribute, which sets the time constant on a lowpass
  filter of the presynaptic activity.
  (`#643 <https://github.com/nengo/nengo/issues/643>`_)
- ``EnsembleArray.add_output`` now accepts a list of functions
  to be computed by each ensemble.
  (`#562 <https://github.com/nengo/nengo/issues/562>`_,
  `#580 <https://github.com/nengo/nengo/pull/580>`_)
- ``LinearFilter`` now has an ``analog`` argument which can be set
  through its constructor. Linear filters with digital coefficients
  can be specified by setting ``analog`` to ``False``.
  (`#819 <https://github.com/nengo/nengo/pull/819>`_)
- Added ``SqrtBeta`` distribution, which describes the distribution
  of semantic pointer elements.
  (`#414 <https://github.com/nengo/nengo/issues/414>`_,
  `#430 <https://github.com/nengo/nengo/pull/430>`_)
- Added ``Triangle`` synapse, which filters with a triangular FIR filter.
  (`#660 <https://github.com/nengo/nengo/pull/660>`_)
- Added ``utils.connection.eval_point_decoding`` function, which
  provides a connection's static decoding of a list of evaluation points.
  (`#700 <https://github.com/nengo/nengo/pull/700>`_)
- Resetting the Simulator now resets all Processes, meaning the
  injected random signals and noise are identical between runs,
  unless the seed is changed (which can be done through
  ``Simulator.reset``).
  (`#582 <https://github.com/nengo/nengo/pull/582>`_,
  `#616 <https://github.com/nengo/nengo/pull/616>`_,
  `#652 <https://github.com/nengo/nengo/pull/652>`_)
- An exception is raised if SPA modules are not properly assigned to an SPA
  attribute.
  (`#730 <https://github.com/nengo/nengo/issues/730>`_,
  `#791 <https://github.com/nengo/nengo/pull/791>`_)
- The ``Product`` network is now more accurate.
  (`#651 <https://github.com/nengo/nengo/pull/651>`_)
- Numpy arrays can now be used as indices for slicing objects.
  (`#754 <https://github.com/nengo/nengo/pull/754>`_)
- ``Config.configures`` now accepts multiple classes rather than
  just one. (`#842 <https://github.com/nengo/nengo/pull/842>`_)
- Added ``add`` method to ``spa.Actions``, which allows
  actions to be added after module has been initialized.
  (`#861 <https://github.com/nengo/nengo/issues/861>`_,
  `#862 <https://github.com/nengo/nengo/pull/862>`_)
- Added SPA wrapper for circular convolution networks, ``spa.Bind``
  (`#849 <https://github.com/nengo/nengo/pull/849>`_)
- Added the ``Voja`` (Vector Oja) learning rule type, which updates an
  ensemble's encoders to fire selectively for its inputs. (see
  ``examples/learning/learn_associations.ipynb``).
  (`#727 <https://github.com/nengo/nengo/issues/727>`_)

**Bug fixes**

- Fixed issue where setting ``Connection.seed`` through the constructor had
  no effect. (`#724 <https://github.com/nengo/nengo/issues/725>`_)
- Fixed issue in which learning connections could not be sliced.
  (`#632 <https://github.com/nengo/nengo/issues/632>`_)
- Fixed issue when probing scalar transforms.
  (`#667 <https://github.com/nengo/nengo/issues/667>`_,
  `#671 <https://github.com/nengo/nengo/pull/671>`_)
- Fix for SPA actions that route to a module with multiple inputs.
  (`#714 <https://github.com/nengo/nengo/pull/714>`_)
- Corrected the ``rmses`` values in ``BuiltConnection.solver_info`` when using
  ``NNls`` and ``Nnl2sL2`` solvers, and the ``reg`` argument for ``Nnl2sL2``.
  (`#839 <https://github.com/nengo/nengo/pull/839>`_)
- ``spa.Vocabulary.create_pointer`` now respects the specified number of
  creation attempts, and returns the most dissimilar pointer if none can be
  found below the similarity threshold.
  (`#817 <https://github.com/nengo/nengo/pull/817>`_)

2.0.2 (October 13, 2015)
========================

2.0.2 is a bug fix release to ensure that Nengo continues
to work with more recent versions of Jupyter
(formerly known as the IPython notebook).

**Behavioural changes**

- The IPython notebook progress bar has to be activated with
  ``%load_ext nengo.ipynb``.
  (`#693 <https://github.com/nengo/nengo/pull/693>`_)

**Improvements**

- Added ``[progress]`` section to ``nengorc`` which allows setting
  ``progress_bar`` and ``updater``.
  (`#693 <https://github.com/nengo/nengo/pull/693>`_)

**Bug fixes**

- Fix compatibility issues with newer versions of IPython,
  and Jupyter. (`#693 <https://github.com/nengo/nengo/pull/693>`_)

2.0.1 (January 27, 2015)
========================

**Behavioural changes**

- Node functions receive ``t`` as a float (instead of a NumPy scalar)
  and ``x`` as a readonly NumPy array (instead of a writeable array).
  (`#626 <https://github.com/nengo/nengo/issues/626>`_,
  `#628 <https://github.com/nengo/nengo/pull/628>`_)

**Improvements**

- ``rasterplot`` works with 0 neurons, and generates much smaller PDFs.
  (`#601 <https://github.com/nengo/nengo/pull/601>`_)

**Bug fixes**

- Fix compatibility with NumPy 1.6.
  (`#627 <https://github.com/nengo/nengo/pull/627>`_)

2.0.0 (January 15, 2015)
========================

Initial release of Nengo 2.0!
Supports Python 2.6+ and 3.3+.
Thanks to all of the contributors for making this possible!
