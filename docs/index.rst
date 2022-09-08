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

.. py:class::ExperimentSet

   Class representing group of experiments

   :attribute experiments: List of Experiment objects
   :type experiments: list[Experiment]
   :attribute metadata: Information about experiment set
   :type metadata: Metadata

.. py:class::Experiment

   Class representing experiment

   :attribute experimentCondition: Conditions of the experiment
   :type experiments: ExperimentCondition
   :attribute experimentComponents: Components of the experiment
   :type experiments: list[ExperimentComponent]
   :attribute metadata: Information about experiment
   :type metadata: Metadata

.. py:class::ExperimentCondition

   Class representing conditions of a experiment

   :attribute columnDiameter: Diameter of column used in experiment
   :type columnDiameter: float
   :attribute columnLength: Length of column used in experiment
   :type columnLength: float
   :attribute feedVolume: Volume of feed in experiment
   :type feedVolume: float
   :attribute feedTime: Time of feed in experiment
   :type feedTime: float
   :attribute flowRate: Flow rate in experiment
   :type flowRate: float

.. py:class::ExperimentComponent

   Class representing component in a experiment

   :attribute concentrationTime: Table of times and concentrations measured in experiment
   :type concentrationTime: pandas.DataFrame()
   :attribute feedConcentration: Concentration of component in feed
   :type feedConcentration: float
   :attribute name: Name of the component
   :type name: str
   :attribute experiment: Reference to experiment with this component
   :type experiment: Experiment

.. py:class::ExperimentClusters

   Class representing clusters of components from multiple experiments

   :attribute clusters: Dictionary of component clusters
   :type clusters: dict[str:list[ExperimentComponents]]
   :attribute metadata: Information about clusters
   :type metadata: Metadata

.. py:class::Metadata

   Class representing information about other objects

   :attribute date: Relevant date
   :type date: str
   :attribute description: Description of the object
   :type description: str
   :attribute path: Path to the source experiment(s)
   :type path: str

.. py:class::Operator

   Class managing the workflow of the program and comunication with the user

   .. py:method::Start()

      Starts and manages the workflow

   .. py:method::Setting_Parameters()

      Asks user to input relevant parameters

      :returns: List of parameters
      :rtype: list[float]

   .. py:method::Load_Experiment_Set(path)

      Creates an ExperimentSet object from excel files

      :param path: Path to the files
      :type path: str
      :returns: ExperimentSet object
      :rtype: ExperimentSet

   .. py:method::Cluster_By_Component(experimentSet)

      Creates a ExperimentClusters object based on component

      :param experimentSet: Experiment set to cluster
      :type experimentSet: ExperimentSet
      :returns: ExperimentClusters object
      :rtype: ExperimentClusters

   .. py:method::Cluster_By_Condition(experimentSet)

      Creates a ExperimentClusters object based on component and conditon of the experiment

      :param experimentSet: Experiment set to cluster
      :type experimentSet: ExperimentSet
      :returns: ExperimentClusters object
      :rtype: ExperimentClusters

   .. py:method::Cluster_By_Condition2(experimentSet)

      Similar as Cluster_By_Condition, different cluster implementation

      :param experimentSet: Experiment set to cluster
      :type experimentSet: ExperimentSet
      :returns: ExperimentClusters object
      :rtype: ExperimentClusters

   .. py:method::Cluster_Match(comp1, comp2, tolerance = 0.05)

      Method helping clustering methods to decide if two components are similar enough

      :param comp1: First component for comparison
      :type comp1: ExperimentComponent
      :param comp2: Second component for comparison
      :type comp2: ExperimentComponent
      :param tolerance: Relative tolerance for similarity decision
      :type tolerance: float
      :returns: Wether the two components are similar
      :rtype: bool

   .. py:method::Create_Key(comp)

      Method helping clustering methods to create key for dictinary

      :param comp: Component from which the key is created from
      :type comp: ExperimentComponent
      :returns: Key for clusters dictionary
      :rtype: str



Analysis and Debugging functions
--------------------------------

.. py:function:: Compare_ExperimentSets(experimentSet1, experimentSet2)

   Compares two experiment sets and allows user to find differences.

   :param ExperimentSet experimentSet1: First experiment set to compare.
   :type experimentSet1: :py:class:ExperimentSet
   :param experimentSet2: Second experiment set to compare.
   :type experimentSet2: :py:class:ExperimentSet

.. py:function:: Loss_Function_Analysis(experimentClusterComp, component='Sac', xstart = 0, ystart = 0, xend = 5002, yend = 5002, xstep = 50, ystep = 50, porosityStart = 0.2, porosotyEnd = 1, porosityStep = 0.1)

   Allows user to analyze results of Lev2_Loss_Function with specified input values.

   :param ExperimentCluster experimentClusterComp: Cluster of components to analyze.
   :type experimentClusterComp: :py:class:ExperimentCluster
   :param component: Specified name of component to analyze.
   :type component: str
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
