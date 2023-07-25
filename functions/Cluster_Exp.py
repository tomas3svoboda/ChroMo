def Cluster_Exp(experimentSet):
    """retired"""
    experimentClusterComp = 1 + experimentSet
    experimentClusterCompCond = 1 + experimentSet + experimentClusterComp

    return [experimentClusterComp, experimentClusterCompCond]

