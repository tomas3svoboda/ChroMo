import matplotlib.pyplot as plt

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
                        tmpDF = comp1.concentrationTime.copy()
                        tmpDF['Time_2'] = comp2.concentrationTime.loc[:, 'Time']
                        tmpDF[comp2.name + '_2'] = comp2.concentrationTime.loc[:, comp2.name]
                        print(tmpDF)
                        comp1.concentrationTime.plot.line(x='Time')
                        comp2.concentrationTime.plot.line(x='Time')
                        plt.show()
                        break
                    if i == "N":
                        break
                    if i == "E":
                        return
    if flag:
        print("They are the same!")
    return