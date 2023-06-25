import matplotlib.pyplot as plt
import pandas as pd

# Function compares two experiment sets
def Compare_ExperimentSets(experimentSet1, experimentSet2):
    flag = True

    # Check if the number of experiments in the sets is equal
    if len(experimentSet1.experiments) != len(experimentSet2.experiments):
        print("Different number of experiments.")
        return

    # Iterate over experiments
    for indexExp in range(len(experimentSet1.experiments)):
        # Check if the number of components in the experiments is equal
        if len(experimentSet1.experiments[indexExp].experimentComponents) != len(
                experimentSet2.experiments[indexExp].experimentComponents):
            print("experiment[" + str(indexExp) + "] - different number of components")
            return

        # Iterate over components
        for indexComp in range(len(experimentSet1.experiments[indexExp].experimentComponents)):
            comp1 = experimentSet1.experiments[indexExp].experimentComponents[indexComp]
            comp2 = experimentSet2.experiments[indexExp].experimentComponents[indexComp]

            # Check if the concentration-time dataframes of the components match
            if not comp1.concentrationTime.equals(comp2.concentrationTime):
                flag = False
                print("experiment[" + str(indexExp) + "].experimentComponent[" + str(indexComp) + "] - not matching")

                while True:
                    i = input("Print?[Y - yes, N - no, E - exit]")
                    if i == "Y":
                        # Concatenate and display the concentration-time dataframes
                        newDF = pd.concat([comp1.concentrationTime, comp2.concentrationTime], axis=1)
                        pd.set_option('display.max_rows', None)
                        print(newDF)

                        # Plot concentration-time lines for both components
                        comp1.concentrationTime.plot.line(x=0)
                        comp2.concentrationTime.plot.line(x=0)
                        plt.show()
                        break
                    if i == "N":
                        break
                    if i == "E":
                        return

    # Check if all components match
    if flag:
        print("They are the same!")

    return