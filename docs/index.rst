.. Chromo documentation master file, created by
   sphinx-quickstart on Tue Sep  6 13:19:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Chromo's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



Analysis and Debuging functions
-------------------------------

.. py:function:: Compare_ExperimentSets(experimentSet1, experimentSet2)

   Compares two experiment sets and allows user to find differences.

   :param experimentSet1: First experiment set to compare.
   :type experimentSet1: ExperimentSet
   :param experimentSet2: Second experiment set to compare.
   :type experimentSet1: ExperimentSet

.. py:function:: Loss_Function_Analysis(experimentClusterComp, component='Sac', xstart = 0, ystart = 0, xend = 5002, yend = 5002, xstep = 50, ystep = 50, porosityStart = 0.2, porosotyEnd = 1, porosityStep = 0.1)

   Allows user to analyze results of Lev2_Loss_Function with specified input values.

   :param experimentClusterComp: Cluster of components to analyze.
   :type experimentSet1: ExperimentCluster
   :param component: Specified name of component to analyze.
   :type experimentSet1: str
   :param xstart: Start of interval of Henry constant values for analysis.
   :type xstart: int
   :param ystart: Start of interval of Dispersion coeficient values for analysis.
   :type ystart: int
   :param xend: End of interval of Henry constant values for analysis.
   :type xend: int
   :param yend: End of interval of Dispersion coeficient values for analysis.
   :type yend: int
   :param xstep: Step of Henry constant values from interval for analysis.
   :type xstep: int
   :param ystep: Step of Dispersion coeficient values from interval for analysis.
   :type ystep: int
   :param porositystart: Start of interval of porosity values for analysis.
   :type porositystart: float
   :param porosityend: End of interval of porosity values for analysis.
   :type porosityend: float
   :param porositystep: Step of porosity values from interval for analysis.
   :type porositystep: float