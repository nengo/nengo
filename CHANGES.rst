***************
Release History
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Deprecated
   - Removed
   - Fixed

3.0.0 (unreleased)
==================

**Added**

- Added progress bar support for Jupyter Lab >=0.32.
  (`#1428 <https://github.com/nengo/nengo/pull/1428>`__,
  `#1087 <https://github.com/nengo/nengo/issues/1087>`__)
- We now warn that the progress bar is not supported in Jupyter Notebook <5.
  (`#1428 <https://github.com/nengo/nengo/pull/1428>`__,
  `#1426 <https://github.com/nengo/nengo/issues/1426>`__)
- Added support for convolutional connections.
  (`#1481 <https://github.com/nengo/nengo/pull/1481>`__)
- Added version tracking to documentation, so that documentation from old
  versions remains available.
  (`#1488 <https://github.com/nengo/nengo/pull/1488>`__)
- Added support for sparse connections.
  (`#1532 <https://github.com/nengo/nengo/pull/1532>`__)
- Added a ``fail_fast`` setting to test operators when they are first
  added to the model. See `configuration options
  <https://www.nengo.ai/nengo/nengorc.html#configuration-options>`__
  for details. (`#1532 <https://github.com/nengo/nengo/pull/1532>`__)
- Added a ``--memory`` option for pytest that prints the total memory
  consumed by the tests when they complete (Linux and Mac OS X only).
  (`#640 <https://github.com/nengo/nengo/pull/640>`__)
- Added a bit precision setting to change the number of bits allocated
  to each value tracked by Nengo.
  (`#640 <https://github.com/nengo/nengo/pull/640>`__)

**Changed**

- Python 2 is no longer supported. The oldest supported Python version is 3.5.
  (`#1520 <https://github.com/nengo/nengo/pull/1520>`__,
  `python3statement.org <https://python3statement.org/>`__)
- Nengo no longer supports Python 3.4.
  Official 3.4 support ended in March 2019.
  (`PEP-429 <https://www.python.org/dev/peps/pep-0429/>`__,
  `#1514 <https://github.com/nengo/nengo/pull/1514>`__)
- Replaced the ``dt`` argument to ``Simulator.trange`` with ``sample_every``
  because ``dt`` would return values that the simulator had not simulated.
  ``dt`` is now an alias for ``sample_every`` and will be removed in the future.
  (`#1368 <https://github.com/nengo/nengo/issues/1368>`_,
  `#1384 <https://github.com/nengo/nengo/pull/1384>`_)
- Dense connection transforms (this includes all previously supported values
  for ``Connection.transform``) will now be represented internally as
  ``nengo.Dense`` objects. Arrays/scalars can still be passed as ``transform``
  values, and they will be automatically converted to the equivalent
  ``nengo.Dense`` object. Retrieving the value of ``my_conn.transform`` will
  return that ``Dense`` object. The original input array can be retrieved
  through ``my_conn.transform.init``.
  (`#1481 <https://github.com/nengo/nengo/pull/1481>`__)
- ``nengo.solvers.NoSolver(w, weights=True)`` now expects ``w`` to have shape
  ``(pre.n_neurons, function_d)``,
  rather than ``pre.n_neurons, post.n_neurons)``. That is, with ``NoSolver``
  you are always specifying the values for the decoders, and encoders/transform
  will be applied automatically to those decoders (as occurs with
  all other solvers). Note that this does not affect
  ``NoSolver(..., weights=False)`` (the default).
  (`#1481 <https://github.com/nengo/nengo/pull/1481>`__)
- Increased minimum NumPy version to 1.11.0. See our
  `instructions for installing NumPy
  <https://www.nengo.ai/nengo/getting_started.html#installing-numpy>`__
  if you need to upgrade.
  (`#1481 <https://github.com/nengo/nengo/pull/1481>`__)
- Solvers are now explicitly marked as compositional or non-compositional
  depending on whether they must act on full connection weight matrices
  when solving for weights.
  (`#1507 <https://github.com/nengo/nengo/pull/1507>`__)
- Solvers no longer take encoders as an argument. Instead, encoders will
  be applied to the targets before the solve function for compositional solvers
  and applied by the Transform builder for non-compositional solvers.
  (`#1507 <https://github.com/nengo/nengo/pull/1507>`__)
- Example Jupyter notebooks have been upgraded to notebook format 4.
  (`#1440 <https://github.com/nengo/nengo/pull/1440>`_)
- Switched documentation to new
  `nengo-sphinx-theme <https://github.com/nengo/nengo-sphinx-theme>`_.
  (`#1489 <https://github.com/nengo/nengo/pull/1489>`__)
- The ``settled_firingrate`` function has been moved from
  ``nengo.utils.neurons`` to ``nengo.neurons``.
  (`#1187 <https://github.com/nengo/nengo/pull/1187>`_)
- Added new pytest config option, ``nengo_test_unsupported`` (replacing the
  previous ``Simulator.unsupported`` functionality).
  (`#1521 <https://github.com/nengo/nengo/pull/1521>`_)
- Switched to nengo-bones templating system for TravisCI config/scripts.
  (`#1514 <https://github.com/nengo/nengo/pull/1514>`_)
- The ``NeuronType.current`` and ``NeuronType.rates`` methods now document
  the supported shapes of parameters and return values.
  (`#1437 <https://github.com/nengo/nengo/pull/1437>`__)
- PES learning updates are now applied on the next timestep rather than
  the current one.
  (`#1398 <https://github.com/nengo/nengo/pull/1398>`_)
- The ``NdarrayParam`` now accepts a ``dtype`` argument to check that
  data assigned to that parameter matches the given Numpy ``dtype``.
  ``DistOrArrayParam`` accepts an analogous ``sample_dtype`` argument.
  (`#1532 <https://github.com/nengo/nengo/pull/1532>`__)
- We no longer test operators when they are initially added to the model,
  which speed up build times slightly. To re-enable this testing,
  enable the ``fail_fast`` RC setting.
  (`#1532 <https://github.com/nengo/nengo/pull/1532>`__)
- ``LinearFilter`` now uses state space representations internally,
  which is faster and potentially more accurate.
  (`#1535 <https://github.com/nengo/nengo/pull/1535>`__)
- The default value of ``y0`` in ``Synapse.filt`` is now 0 instead of
  the initial value of the input signal. This allows unstable filters
  (e.g., integrators) to be used with ``filt``.
  (`#1535 <https://github.com/nengo/nengo/pull/1535>`__)

**Deprecated**

- Deprecated the ``nengo.spa`` module. Use the
  `Nengo SPA <https://www.nengo.ai/nengo-spa/index.html>`__
  project instead.
  (`#1465 <https://github.com/nengo/nengo/pull/1465>`_)
- The ``A`` and ``B`` inputs to the ``Product`` and ``CircularConvolution``
  networks are officially deprecated. Use ``input_a`` and ``input_b`` instead.
  (`#887 <https://github.com/nengo/nengo/issues/887>`__,
  `#1179 <https://github.com/nengo/nengo/pull/1179>`__)
- ``nengo.utils.compat`` will be removed in the next minor release.
  (`#1520 <https://github.com/nengo/nengo/pull/1520>`_)

**Removed**

- Networks no longer accept the ``net`` argument. To set network arguments
  like ``label``, pass them as keyword arguments instead.
  (`#1179 <https://github.com/nengo/nengo/pull/1179>`__)
- Removed ``generate_graphviz`` utility function. It can now be found in
  `nengo_extras <https://github.com/nengo/nengo_extras/>`__.
  (`#1187 <https://github.com/nengo/nengo/pull/1187>`_)
- Removed functions for estimating firing rates from spikes. They can now
  be found in `nengo_extras <https://github.com/nengo/nengo_extras/>`__.
  (`#1187 <https://github.com/nengo/nengo/pull/1187>`_)
- Removed the ``probe_all`` function. It can now be found in
  `nengo_extras <https://github.com/nengo/nengo_extras/>`__.
  (`#1187 <https://github.com/nengo/nengo/pull/1187>`_)
- ``PES.correction`` is no longer probeable.
  (`#1398 <https://github.com/nengo/nengo/pull/1398>`_)

**Fixed**

- ``FrozenObjects`` can control parameter initialization order when copying,
  which fixed a bug encountered when copying convolutional connections.
  (`#1493 <https://github.com/nengo/nengo/pull/1493>`__)
- Fixed an issue in which reshaped signals were not having their offset
  values preserved, causing issues with some node functions.
  (`#1474 <https://github.com/nengo/nengo/pull/1474>`__)
- Better error message when Node output function does not match the
  given ``size_in``/``size_out``.
  (`#1452 <https://github.com/nengo/nengo/issues/1452>`_,
  `#1434 <https://github.com/nengo/nengo/pull/1434>`_)
- Several objects had elements missing from their string representations.
  These strings are now automatically generated and tested to be complete.
  (`#1472 <https://github.com/nengo/nengo/pull/1472>`__)
- Fixed the progress bar in recent Jupyter Lab versions.
  (`#1499 <https://github.com/nengo/nengo/issues/1499>`_,
  `#1500 <https://github.com/nengo/nengo/pull/1500>`_)
- Some higher-order ``LinearFilter`` synapses had unnecessary delays
  that have now been removed.
  (`#1535 <https://github.com/nengo/nengo/pull/1535>`__)

2.8.0 (June 9, 2018)
====================

**Added**

- Added a warning when setting ``gain`` and ``bias`` along with either of
  ``max_rates`` or ``intercepts``, as the latter two parameters are ignored.
  (`#1431 <https://github.com/nengo/nengo/issues/1431>`_,
  `#1433 <https://github.com/nengo/nengo/pull/1433>`_)

**Changed**

- Learning rules can now be sliced when providing error input.
  (`#1365 <https://github.com/nengo/nengo/issues/1365>`_,
  `#1385 <https://github.com/nengo/nengo/pull/1385>`_)
- The order of parameters in learning rules has changed such that
  ``learning_rate`` always comes first.
  (`#1095 <https://github.com/nengo/nengo/pull/1095>`__)
- Learning rules take ``pre_synapse``, ``post_synapse``, and ``theta_synapse``
  instead of ``pre_tau``, ``post_tau``, and ``theta_tau`` respectively.
  This allows arbitrary ``Synapse`` objects to be used as filters on
  learning signals.
  (`#1095 <https://github.com/nengo/nengo/pull/1095>`__)

**Deprecated**

- The ``nengo.ipynb`` IPython extension and the ``IPython2ProgressBar``
  have been deprecated and replaced by the ``IPython5ProgressBar``.
  This progress bar will be automatically activated in IPython and
  Jupyter notebooks from IPython version 5.0 onwards.
  (`#1087 <https://github.com/nengo/nengo/issues/1087>`_,
  `#1375 <https://github.com/nengo/nengo/pull/1375>`_)
- The ``pre_tau``, ``post_tau``, and ``theta_tau`` parameters
  for learning rules are deprecated. Instead, use ``pre_synapse``,
  ``post_synapse``, and ``theta_synapse`` respectively.
  (`#1095 <https://github.com/nengo/nengo/pull/1095>`__)

**Removed**

- Removed ``nengo.utils.docutils`` in favor of using
  `nbsphinx <https://nbsphinx.readthedocs.io>`_.
  (`#1349 <https://github.com/nengo/nengo/pull/1349>`_)

2.7.0 (March 7, 2018)
=====================

**Added**

- Added ``amplitude`` parameter to ``LIF``, ``LIFRate``,
  and ``RectifiedLinear``  which scale the output amplitude.
  (`#1325 <https://github.com/nengo/nengo/pull/1325>`_,
  `#1391 <https://github.com/nengo/nengo/pull/1391>`__)
- Added the ``SpikingRectifiedLinear`` neuron model.
  (`#1391 <https://github.com/nengo/nengo/pull/1391>`__)

**Changed**

- Default values can no longer be set for
  ``Ensemble.n_neurons`` or ``Ensemble.dimensions``.
  (`#1372 <https://github.com/nengo/nengo/pull/1372>`__)
- If the simulator seed is not specified, it will now be set
  from the network seed if a network seed is specified.
  (`#980 <https://github.com/nengo/nengo/issues/980>`__,
  `#1386 <https://github.com/nengo/nengo/pull/1386>`__)

**Fixed**

- Fixed an issue in which signals could not be pickled,
  making it impossible to pickle ``Model`` instances.
  (`#1135 <https://github.com/nengo/nengo/pull/1135>`_)
- Better error message for invalid return values in ``nengo.Node`` functions.
  (`#1317 <https://github.com/nengo/nengo/pull/1317>`_)
- Fixed an issue in which accepting and passing ``(*args, **kwargs)``
  could not be used in custom solvers.
  (`#1358 <https://github.com/nengo/nengo/issues/1358>`_,
  `#1359 <https://github.com/nengo/nengo/pull/1359>`_)
- Fixed an issue in which the cache would not release its index lock
  on abnormal termination of the Nengo process.
  (`#1364 <https://github.com/nengo/nengo/pull/1364>`_)
- Fixed validation checks that prevented the default
  from being set on certain parameters.
  (`#1372 <https://github.com/nengo/nengo/pull/1372>`__)
- Fixed an issue with repeated elements in slices in which
  a positive and negative index referred to the same dimension.
  (`#1395 <https://github.com/nengo/nengo/pull/1395>`_)
- The ``Simulator.n_steps`` and ``Simulator.time`` properties
  now return scalars, as was stated in the documentation.
  (`#1406 <https://github.com/nengo/nengo/pull/1406>`_)
- Fixed the ``--seed-offset`` option of the test suite.
  (`#1409 <https://github.com/nengo/nengo/pull/1409>`_)

2.6.0 (October 6, 2017)
=======================

**Added**

- Added a ``NoSolver`` solver that can be used to manually pass in
  a predefined set of decoders or weights to a connection.
  (`#1352 <https://github.com/nengo/nengo/pull/1352>`_)
- Added a ``Piecewise`` process, which replaces the now deprecated
  ``piecewise`` function.
  (`#1036 <https://github.com/nengo/nengo/issues/1036>`_,
  `#1100 <https://github.com/nengo/nengo/pull/1100>`_,
  `#1355 <https://github.com/nengo/nengo/pull/1355/>`_,
  `#1362 <https://github.com/nengo/nengo/pull/1362>`_)

**Changed**

- The minimum required version of NumPy has been raised to 1.8.
  (`#947 <https://github.com/nengo/nengo/issues/947>`_)
- Learning rules can now have a learning rate of 0.
  (`#1356 <https://github.com/nengo/nengo/pull/1356>`_)
- Running the simulator for zero timesteps will now issue a warning,
  and running for negative time will error.
  (`#1354 <https://github.com/nengo/nengo/issues/1354>`_,
  `#1357 <https://github.com/nengo/nengo/pull/1357>`_)

**Fixed**

- Fixed an issue in which the PES learning rule could not be used
  on connections to an ``ObjView`` when using a weight solver.
  (`#1317 <https://github.com/nengo/nengo/pull/1317>`_)
- The progress bar that can appear when building a large model
  will now appear earlier in the build process.
  (`#1340 <https://github.com/nengo/nengo/pull/1340>`_)
- Fixed an issue in which ``ShapeParam`` would always store ``None``.
  (`#1342 <https://github.com/nengo/nengo/pull/1342>`_)
- Fixed an issue in which multiple identical indices in a slice were ignored.
  (`#947 <https://github.com/nengo/nengo/issues/947>`_,
  `#1361 <https://github.com/nengo/nengo/pull/1361>`_)

**Deprecated**

- The ``piecewise`` function in ``nengo.utils.functions`` has been deprecated.
  Please use the ``Piecewise`` process instead.
  (`#1100 <https://github.com/nengo/nengo/pull/1100>`_)

2.5.0 (July 24, 2017)
=====================

**Added**

- Added a ``n_neurons`` property to ``Network``, which gives the
  number of neurons in the network, including all subnetworks.
  (`#435 <https://github.com/nengo/nengo/issues/435>`_,
  `#1186 <https://github.com/nengo/nengo/pull/1186>`_)
- Added a new example showing how adjusting ensemble tuning curves can
  improve function approximation.
  (`#1129 <https://github.com/nengo/nengo/pull/1129>`_)
- Added a minimum magnitude option to ``UniformHypersphere``.
  (`#799 <https://github.com/nengo/nengo/pull/799>`_)
- Added documentation on RC settings.
  (`#1130 <https://github.com/nengo/nengo/pull/1130>`_)
- Added documentation on improving performance.
  (`#1119 <https://github.com/nengo/nengo/issues/1119>`_,
  `#1130 <https://github.com/nengo/nengo/pull/1130>`_)
- Added ``LinearFilter.combine`` method to
  combine two ``LinearFilter`` instances.
  (`#1312 <https://github.com/nengo/nengo/pull/1312>`_)
- Added a method to all neuron types to compute ensemble
  ``max_rates`` and ``intercepts`` given ``gain`` and ``bias``.
  (`#1334 <https://github.com/nengo/nengo/pull/1334>`_)

**Changed**

- Learning rules now have a ``size_in`` parameter and attribute,
  allowing both integers and strings to define the dimensionality
  of the learning rule. This replaces the ``error_type`` attribute.
  (`#1307 <https://github.com/nengo/nengo/pull/1307>`_,
  `#1310 <https://github.com/nengo/nengo/pull/1310>`_)
- ``EnsembleArray.n_neurons`` now gives the total number of neurons
  in all ensembles, including those in subnetworks.
  To get the number of neurons in each ensemble,
  use ``EnsembleArray.n_neurons_per_ensemble``.
  (`#1186 <https://github.com/nengo/nengo/pull/1186>`_)
- The `Nengo modelling API document
  <https://www.nengo.ai/nengo/frontend_api.html>`_
  now has summaries to help navigate the page.
  (`#1304 <https://github.com/nengo/nengo/pull/1304>`_)
- The error raised when a ``Connection`` function returns ``None``
  is now more clear.
  (`#1319 <https://github.com/nengo/nengo/pull/1319>`_)
- We now raise an error when a ``Connection`` transform is set to ``None``.
  (`#1326 <https://github.com/nengo/nengo/pull/1326>`_)

**Fixed**

- Probe cache is now cleared on simulator reset.
  (`#1324 <https://github.com/nengo/nengo/pull/1324>`_)
- Neural gains are now always applied after the synapse model.
  Previously, this was the case for decoded connections
  but not neuron-to-neuron connections.
  (`#1330 <https://github.com/nengo/nengo/pull/1330>`_)
- Fixed a crash when a lock cannot be acquired while shrinking the cache.
  (`#1335 <https://github.com/nengo/nengo/issues/1335>`_,
  `#1336 <https://github.com/nengo/nengo/pull/1336>`_)

2.4.0 (April 18, 2017)
======================

**Added**

- Added an optimizer that reduces simulation time for common types of models.
  The optimizer can be turned off by passing ``optimize=False`` to ``Simulator``.
  (`#1035 <https://github.com/nengo/nengo/pull/1035>`_)
- Added the option to not normalize encoders by setting
  ``Ensemble.normalize_encoders`` to ``False``.
  (`#1191 <https://github.com/nengo/nengo/issues/1191>`_,
  `#1267 <https://github.com/nengo/nengo/pull/1267>`_)
- Added the ``Samples`` distribution to allow raw NumPy arrays
  to be passed in situations where a distribution is required.
  (`#1233 <https://github.com/nengo/nengo/pull/1233>`_)

**Changed**

- We now raise an error when an ensemble is assigned a negative gain.
  This can occur when solving for gains with intercepts greater than 1.
  (`#1212 <https://github.com/nengo/nengo/issues/1212>`_,
  `#1231 <https://github.com/nengo/nengo/issues/1231>`_,
  `#1248 <https://github.com/nengo/nengo/pull/1248>`_)
- We now raise an error when a ``Node`` or ``Direct`` ensemble
  produces a non-finite value.
  (`#1178 <https://github.com/nengo/nengo/issues/1178>`_,
  `#1280 <https://github.com/nengo/nengo/issues/1280>`_,
  `#1286 <https://github.com/nengo/nengo/pull/1286>`_)
- We now enforce that the ``label`` of a network must be a string or ``None``,
  and that the ``seed`` of a network must be an int or ``None``.
  This helps avoid situations where the seed would mistakenly
  be passed as the label.
  (`#1277 <https://github.com/nengo/nengo/pull/1277>`_,
  `#1275 <https://github.com/nengo/nengo/issues/1275>`_)
- It is now possible to pass NumPy arrays in the ``ens_kwargs`` argument of
  ``EnsembleArray``. Arrays are wrapped in a ``Samples`` distribution internally.
  (`#691 <https://github.com/nengo/nengo/issues/691>`_,
  `#766 <https://github.com/nengo/nengo/issues/766>`_,
  `#1233 <https://github.com/nengo/nengo/pull/1233>`_)
- The default refractory period (``tau_ref``) for the ``Sigmoid`` neuron type
  has changed to 2.5 ms (from 2 ms) for better compatibility with the
  default maximum firing rates of 200-400 Hz.
  (`#1248 <https://github.com/nengo/nengo/pull/1248>`_)
- Inputs to the ``Product`` and ``CircularConvolution`` networks have been
  renamed from ``A`` and ``B`` to ``input_a`` and ``input_b`` for consistency.
  The old names are still available, but should be considered deprecated.
  (`#887 <https://github.com/nengo/nengo/issues/887>`_,
  `#1296 <https://github.com/nengo/nengo/pull/1296>`_)

**Fixed**

- Properly handle non C-contiguous node outputs.
  (`#1184 <https://github.com/nengo/nengo/issues/1184>`_,
  `#1185 <https://github.com/nengo/nengo/pull/1185>`_)

**Deprecated**

- The ``net`` argument to networks has been deprecated. This argument existed
  so that network components could be added to an existing network instead of
  constructing a new network. However, this feature is rarely used,
  and makes the code more complicated for complex networks.
  (`#1296 <https://github.com/nengo/nengo/pull/1296>`_)

2.3.1 (February 18, 2017)
=========================

**Added**

- Added documentation on config system quirks.
  (`#1224 <https://github.com/nengo/nengo/pull/1224>`_)
- Added ``nengo.utils.network.activate_direct_mode`` function to make it
  easier to activate direct mode in networks where some parts require neurons.
  (`#1111 <https://github.com/nengo/nengo/issues/1111>`_,
  `#1168 <https://github.com/nengo/nengo/pull/1168>`_)

**Fixed**

- The matrix multiplication example will now work with matrices of any size
  and uses the product network for clarity.
  (`#1159 <https://github.com/nengo/nengo/pull/1159>`_)
- Fixed instances in which passing a callable class as a function could fail.
  (`#1245 <https://github.com/nengo/nengo/pull/1245>`_)
- Fixed an issue in which probing some attributes would be one timestep
  faster than other attributes.
  (`#1234 <https://github.com/nengo/nengo/issues/1234>`_,
  `#1245 <https://github.com/nengo/nengo/pull/1245>`_)
- Fixed an issue in which SPA models could not be copied.
  (`#1266 <https://github.com/nengo/nengo/issues/1266>`_,
  `#1271 <https://github.com/nengo/nengo/pull/1271>`_)
- Fixed an issue in which Nengo would crash if other programs
  had locks on Nengo cache files in Windows.
  (`#1200 <https://github.com/nengo/nengo/issues/1200>`_,
  `#1235 <https://github.com/nengo/nengo/pull/1235>`_)

**Changed**

- Integer indexing of Nengo objects out of range raises an ``IndexError``
  now to be consistent with standard Python behaviour.
  (`#1176 <https://github.com/nengo/nengo/issues/1176>`_,
  `#1183 <https://github.com/nengo/nengo/pull/1183>`_)
- Documentation that applies to all Nengo projects has been moved to
  https://www.nengo.ai/.
  (`#1251 <https://github.com/nengo/nengo/pull/1251>`_)

2.3.0 (November 30, 2016)
=========================

**Added**

- It is now possible to probe ``scaled_encoders`` on ensembles.
  (`#1167 <https://github.com/nengo/nengo/pull/1167>`_,
  `#1117 <https://github.com/nengo/nengo/issues/1117>`_)
- Added ``copy`` method to Nengo objects. Nengo objects can now be pickled.
  (`#977 <https://github.com/nengo/nengo/issues/977>`_,
  `#984 <https://github.com/nengo/nengo/pull/984>`_)
- A progress bar now tracks the build process
  in the terminal and Jupyter notebook.
  (`#937 <https://github.com/nengo/nengo/issues/937>`_,
  `#1151 <https://github.com/nengo/nengo/pull/1151>`_)
- Added ``nengo.dists.get_samples`` function for convenience
  when working with distributions or samples.
  (`#1181 <https://github.com/nengo/nengo/pull/1181>`_,
  `docs <https://www.nengo.ai/nengo/frontend_api.html#nengo.dists.get_samples>`_)

**Changed**

- Access to probe data via ``nengo.Simulator.data`` is now cached,
  making repeated access much faster.
  (`#1076 <https://github.com/nengo/nengo/issues/1076>`_,
  `#1175 <https://github.com/nengo/nengo/pull/1175>`_)

**Deprecated**

- Access to ``nengo.Simulator.model`` is deprecated. To access static data
  generated during the build use ``nengo.Simulator.data``. It provides access
  to everything that ``nengo.Simulator.model.params`` used to provide access to
  and is the canonical way to access this data across different backends.
  (`#1145 <https://github.com/nengo/nengo/issues/1145>`_,
  `#1173 <https://github.com/nengo/nengo/pull/1173>`_)

2.2.0 (September 12, 2016)
==========================

**API changes**

- It is now possible to pass a NumPy array to the ``function`` argument
  of ``nengo.Connection``. The values in the array are taken to be the
  targets in the decoder solving process, which means that the ``eval_points``
  must also be set on the connection.
  (`#1010 <https://github.com/nengo/nengo/pull/1010>`_)
- ``nengo.utils.connection.target_function`` is now deprecated, and will
  be removed in Nengo 3.0. Instead, pass the targets directly to the
  connection through the ``function`` argument.
  (`#1010 <https://github.com/nengo/nengo/pull/1010>`_)

**Behavioural changes**

- Dropped support for NumPy 1.6. Oldest supported NumPy version is now 1.7.
  (`#1147 <https://github.com/nengo/nengo/pull/1147>`_)

**Improvements**

- Added a ``nengo.backends`` entry point to make the reference simulator
  discoverable for other Python packages. In the future all backends should
  declare an entry point accordingly.
  (`#1127 <https://github.com/nengo/nengo/pull/1127>`_)
- Added ``ShapeParam`` to store array shapes.
  (`#1045 <https://github.com/nengo/nengo/pull/1045>`_)
- Added ``ThresholdingPreset`` to configure ensembles for thresholding.
  (`#1058 <https://github.com/nengo/nengo/issues/1058>`_,
  `#1077 <https://github.com/nengo/nengo/pull/1077>`_,
  `#1148 <https://github.com/nengo/nengo/pull/1148>`_)
- Tweaked ``rasterplot`` so that spikes from different neurons don't overlap.
  (`#1121 <https://github.com/nengo/nengo/pull/1121>`_)

**Documentation**

- Added a page explaining the config system and preset configs.
  (`#1150 <https://github.com/nengo/nengo/pull/1150>`_)

**Bug fixes**

- Fixed some situations where the cache index becomes corrupt by
  writing the updated cache index atomically (in most cases).
  (`#1097 <https://github.com/nengo/nengo/issues/1097>`_,
  `#1107 <https://github.com/nengo/nengo/pull/1107>`_)
- The synapse methods ``filt`` and ``filtfilt`` now support lists as input.
  (`#1123 <https://github.com/nengo/nengo/pull/1123>`_)
- Added a registry system so that only stable objects are cached.
  (`#1054 <https://github.com/nengo/nengo/issues/1054>`_,
  `#1068 <https://github.com/nengo/nengo/pull/1068>`_)
- Nodes now support array views as input.
  (`#1156 <https://github.com/nengo/nengo/issues/1156>`_,
  `#1157 <https://github.com/nengo/nengo/pull/1157>`_)

2.1.2 (June 27, 2016)
=====================

**Bug fixes**

- The DecoderCache is now more robust when used improperly, and no longer
  requires changes to backends in order to use properly.
  (`#1112 <https://github.com/nengo/nengo/pull/1112>`_)

2.1.1 (June 24, 2016)
=====================

**Improvements**

- Improved the default ``LIF`` neuron model to spike at the same rate as the
  ``LIFRate`` neuron model for constant inputs. The older model has been
  moved to `nengo_extras <https://github.com/nengo/nengo-extras>`_
  under the name ``FastLIF``.
  (`#975 <https://github.com/nengo/nengo/pull/975>`_)
- Added ``y0`` attribute to ``WhiteSignal``, which adjusts the phase of each
  dimension to begin with absolute value closest to ``y0``.
  (`#1064 <https://github.com/nengo/nengo/pull/1064>`_)
- Allow the ``AssociativeMemory`` to accept Semantic Pointer expressions as
  ``input_keys`` and ``output_keys``.
  (`#982 <https://github.com/nengo/nengo/pull/982>`_)

**Bug fixes**

- The DecoderCache is used as context manager instead of relying on the
  ``__del__`` method for cleanup. This should solve problems with the
  cache's file lock not being removed. It might be necessary to
  manually remove the ``index.lock`` file in the cache directory after
  upgrading from an older Nengo version.
  (`#1053 <https://github.com/nengo/nengo/pull/1053>`_,
  `#1041 <https://github.com/nengo/nengo/issues/1041>`_,
  `#1048 <https://github.com/nengo/nengo/issues/1048>`_)
- If the cache index is corrupted, we now fail gracefully by invalidating
  the cache and continuing rather than raising an exception.
  (`#1110 <https://github.com/nengo/nengo/pull/1110>`_,
  `#1097 <https://github.com/nengo/nengo/issues/1097>`_)
- The ``Nnls`` solver now works for weights. The ``NnlsL2`` solver is
  improved since we clip values to be non-negative before forming
  the Gram system.
  (`#1027 <https://github.com/nengo/nengo/pull/1027>`_,
  `#1019 <https://github.com/nengo/nengo/issues/1019>`_)
- Eliminate memory leak in the parameter system.
  (`#1089 <https://github.com/nengo/nengo/issues/1089>`_,
  `#1090 <https://github.com/nengo/nengo/pull/1090>`_)
- Allow recurrence of the form ``a=b, b=a`` in basal ganglia SPA actions.
  (`#1098 <https://github.com/nengo/nengo/issues/1098>`_,
  `#1099 <https://github.com/nengo/nengo/pull/1099>`_)
- Support a greater range of Jupyter notebook and ipywidgets versions with the
  the ``ipynb`` extensions.
  (`#1088 <https://github.com/nengo/nengo/pull/1088>`_,
  `#1085 <https://github.com/nengo/nengo/issues/1085>`_)

2.1.0 (April 27, 2016)
======================

**API changes**

- A new class for representing stateful functions called ``Process``
  has been added. ``Node`` objects are now process-aware, meaning that
  a process can be used as a node's ``output``. Unlike non-process
  callables, processes are properly reset when a simulator is reset.
  See the ``processes.ipynb`` example notebook, or the API documentation
  for more details.
  (`#590 <https://github.com/nengo/nengo/pull/590>`_,
  `#652 <https://github.com/nengo/nengo/pull/652>`_,
  `#945 <https://github.com/nengo/nengo/pull/945>`_,
  `#955 <https://github.com/nengo/nengo/pull/955>`_)
- Spiking ``LIF`` neuron models now accept an additional argument,
  ``min_voltage``. Voltages are clipped such that they do not drop below
  this value (previously, this was fixed at 0).
  (`#666 <https://github.com/nengo/nengo/pull/666>`_)
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
- Reorganised the Associative Memory code, including removing many extra
  parameters from ``nengo.networks.assoc_mem.AssociativeMemory`` and modifying
  the defaults of others.
  (`#797 <https://github.com/nengo/nengo/pull/797>`_)
- Add ``close`` method to ``Simulator``. ``Simulator`` can now be used
  used as a context manager.
  (`#857 <https://github.com/nengo/nengo/issues/857>`_,
  `#739 <https://github.com/nengo/nengo/issues/739>`_,
  `#859 <https://github.com/nengo/nengo/pull/859>`_)
- Most exceptions that Nengo can raise are now custom exception classes
  that can be found in the ``nengo.exceptions`` module.
  (`#781 <https://github.com/nengo/nengo/pull/781>`_)
- All Nengo objects (``Connection``, ``Ensemble``, ``Node``, and ``Probe``)
  now accept a ``label`` and ``seed`` argument if they didn't previously.
  (`#958 <https://github.com/nengo/nengo/pull/859>`_)
- In ``nengo.synapses``, ``filt`` and ``filtfilt`` are deprecated. Every
  synapse type now has ``filt`` and ``filtfilt`` methods that filter
  using the synapse.
  (`#945 <https://github.com/nengo/nengo/pull/945>`_)
- ``Connection`` objects can now accept a ``Distribution`` for the transform
  argument; the transform matrix will be sampled from that distribution
  when the model is built.
  (`#979 <https://github.com/nengo/nengo/pull/979>`_).

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
- Bias nodes in ``spa.Cortical``, and gate ensembles and connections in
  ``spa.Thalamus`` are now stored in the target modules.
  (`#894 <https://github.com/nengo/nengo/issues/894>`_,
  `#906 <https://github.com/nengo/nengo/pull/906>`_)
- The ``filt`` and ``filtfilt`` functions on ``Synapse`` now use the initial
  value of the input signal to initialize the filter output by default. This
  provides more accurate filtering at the beginning of the signal, for signals
  that do not start at zero.
  (`#945 <https://github.com/nengo/nengo/pull/945>`_)

**Improvements**

- Added ``Ensemble.noise`` attribute, which injects noise directly into
  neurons according to a stochastic ``Process``.
  (`#590 <https://github.com/nengo/nengo/pull/590>`_)
- Added a ``randomized_svd`` subsolver for the L2 solvers. This can be much
  quicker for large numbers of neurons or evaluation points.
  (`#803 <https://github.com/nengo/nengo/pull/803>`_)
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
  (`#582 <https://github.com/nengo/nengo/issues/582>`_,
  `#616 <https://github.com/nengo/nengo/issues/616>`_,
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
  (`#727 <https://github.com/nengo/nengo/pull/727>`_)
- Added a clipped exponential distribution useful for thresholding, in
  particular in the AssociativeMemory.
  (`#779 <https://github.com/nengo/nengo/pull/779>`_)
- Added a cosine similarity distribution, which is the distribution of the
  cosine of the angle between two random vectors. It is useful for setting
  intercepts, in particular when using the ``Voja`` learning rule.
  (`#768 <https://github.com/nengo/nengo/pull/768>`_)
- ``nengo.synapses.LinearFilter`` now has an ``evaluate`` method to
  evaluate the filter response to sine waves of given frequencies. This can
  be used to create Bode plots, for example.
  (`#945 <https://github.com/nengo/nengo/pull/945>`_)
- ``nengo.spa.Vocabulary`` objects now have a ``readonly`` attribute that
  can be used to disallow adding new semantic pointers. Vocabulary subsets
  are read-only by default.
  (`#699 <https://github.com/nengo/nengo/pull/699>`_)
- Improved performance of the decoder cache by writing all decoders
  of a network into a single file.
  (`#946 <https://github.com/nengo/nengo/pull/946>`_)

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
- Probing a Connection's output now returns the output of that individual
  Connection, rather than the input to the Connection's post Ensemble.
  (`#973 <https://github.com/nengo/nengo/issues/973>`_,
  `#974 <https://github.com/nengo/nengo/pull/974>`_)
- Fixed thread-safety of using networks and config in ``with`` statements.
  (`#989 <https://github.com/nengo/nengo/pull/989>`_)
- The decoder cache will only be used when a seed is specified.
  (`#946 <https://github.com/nengo/nengo/pull/946>`_)

2.0.4 (April 27, 2016)
======================

**Bug fixes**

- Cache now fails gracefully if the ``legacy.txt`` file cannot be read.
  This can occur if a later version of Nengo is used.

2.0.3 (December 7, 2015)
========================

**API changes**

- The ``spa.State`` object replaces the old ``spa.Memory`` and ``spa.Buffer``.
  These old modules are deprecated and will be removed in 2.2.
  (`#796 <https://github.com/nengo/nengo/pull/796>`_)

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
