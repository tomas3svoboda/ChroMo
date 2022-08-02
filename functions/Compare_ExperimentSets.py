import matplotlib.pyplot as plt
import pandas as pd


def Compare_ExperimentSets(experimentSet1, experimentSet2):
    flag = True
    if len(experimentSet1.experiments) != len(experimentSet2.experiments):
        print("Different number of experiments.")
        return
    for indexExp in range(len(experimentSet1.experiments)):
        if len(experimentSet1.experiments[indexExp].experimentComponents) != len(
                experimentSet2.experiments[indexExp].experimentComponents):
            print("experiment[" + str(indexExp) + "] - different number of components")
            return
        for indexComp in range(len(experimentSet1.experiments[indexExp].experimentComponents)):
            comp1 = experimentSet1.experiments[indexExp].experimentComponents[indexComp]
            comp2 = experimentSet2.experiments[indexExp].experimentComponents[indexComp]
            if not comp1.concentrationTime.equals(comp2.concentrationTime):
                flag = False
                print("experiment[" + str(indexExp) + "].experimentComponent[" + str(indexComp) + "] - not matching")
                while True:
                    i = input("Print?[Y - yes, N - no, E - exit]")
                    if i == "Y":
                        newDF = pd.concat([comp1.concentrationTime, comp2.concentrationTime], axis=1)
                        pd.set_option('display.max_rows', None)
                        print(newDF)
                        comp1.concentrationTime.plot.line(x=0)
                        comp2.concentrationTime.plot.line(x=0)
                        plt.show()
                        break
                    if i == "N":
                        break
                    if i == "E":
                        return
    if flag:
        print("They are the same!")
    return