import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import walk
import os
import datetime
import time
import json
from objects.ExperimentSet import ExperimentSet
from objects.ExperimentComponent import ExperimentComponent
from objects.Experiment import Experiment
from objects.ExperimentClusters import ExperimentClusters
from functions.Fit_Gauss import Fit_Gauss
from functions.Ret_Time_Cor import Ret_Time_Cor
from functions.Remote_DP_Elim import Remote_DP_Elim
from functions.Deep_Copy_ExperimentSet import Deep_Copy_ExperimentSet
from functions.Mass_Balance_Cor import Mass_Balance_Cor
from functions.Select_Iso_Exp import Select_Iso_Exp
from functions.Loss_Function_Analysis import Loss_Function_Analysis
from functions.Loss_Function_Analysis_Simple import Loss_Function_Analysis_Simple
from functions.Model_Analysis import Model_Analysis
from functions.Solver_Analysis import Solver_Analysis
from functions.Iso_Decision import Iso_Decision
from functions.Compare_ExperimentSets import Compare_ExperimentSets
from functions.Bilevel_Optim import Bilevel_Optim
from functions.solvers.Solver_Choice import Solver_Choice
from functions.Handle_File_Creation import Handle_File_Creation
from functions.Dead_Volume_Adjustment import Dead_Volume_Adjustment
from functions.Dead_Volume_Preprocess import Dead_Volume_Preprocess
from functions.singleLossFunctions.Multi_Loss_Function_Wrapper import Multi_Loss_Function_Wrapper
from functions.handle_Optim_Settings import handle_Optim_Settings


"""
Time measuring decorator
"""
def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        if 'log_time' in kwargs:
            name = kwargs.get('log_name', method.__name__.upper())
            kwargs['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.22f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


class Operator:
    """Class orchestrating optimization workflow and command line user interface"""
    def Start(self):
        path = input('Enter path to Experiment set: ')
        experimentSet = self.Load_Experiment_Set(path)
        gaussSelection = input("Replace experiment data with gauss curve?[Y - yes, N - no]")
        currentExperimentSet = Dead_Volume_Preprocess(experimentSet)
        if gaussSelection == "Y":
            currentExperimentSet = Fit_Gauss(currentExperimentSet)
            self.Save_Graphs_To_Directory(experimentSet, currentExperimentSet, path)
            print("Graphs of Gauss Curves Created.")
        else:
            print("Fitting Gauss Curve skipped.")
        retTimeSelection = input("Correct for retention time?[Y - yes, N - no]")
        if retTimeSelection == "Y":
            retTimeThresholdSelection = float(input("Select zero threshold for retention time correction."))
            experimentClusterExp = self.Cluster_By_Experiment(currentExperimentSet)
            currentExperimentSet = Ret_Time_Cor(currentExperimentSet, experimentClusterExp, retTimeThresholdSelection, True)
            print("File with Time Corrections Created.")
        else:
            print("Retention Time Correction skipped.")
        massBalanceSelection = input("Correct for mass balance?[Y - yes, N - no]")
        if massBalanceSelection == "Y":
            currentExperimentSet = Mass_Balance_Cor(currentExperimentSet, True)
            print("File with Mass Corrections Created.")
        else:
            print("Mass Balance Correction skipped.")
        experimentClusterComp = self.Cluster_By_Component(currentExperimentSet)
        lossFunctionChoices = ['Simple', 'Squares', 'LogSimple', 'LogSquares']
        lossFunctionString = ", ".join(lossFunctionChoices)
        lossFunctionSelection = input("Sellect loss function [" + lossFunctionString + "]")
        solverChoices = ['Lin', 'Nonlin']
        solverString = ", ".join(solverChoices)
        solverSelection = input("Sellect solver [" + solverString + "]")
        factorSelectionString = "1 - Fc = 1\n" + \
                                "2 - Fc = 1 / maximalOutputConc\n" + \
                                "3 - Fc = 1 / maximalOutputConc^2\n" + \
                                "4 - Fc = 1 / feedConc\n" + \
                                "5 - Fc = 1 / feedConc^2\n" + \
                                "6 - Fc = 1 / feedMass\n" + \
                                "7 - Fc = 1 / feedMass^2\n"
        factorSelection = input("Select factor:\n" + factorSelectionString)
        compSelectString = ", ".join(experimentClusterComp.clusters.keys())
        intervalSelection = input("Print graphs to help select intervals?[Y - yes, N - no]")
        while intervalSelection == "Y":
            porosity = float(input("Select Porosity: "))
            satur = 0
            if solverSelection == "Nonlin":
                satur = float(input("Select Saturation Coefficient: "))
            Kstart = float(input("Select Start for K interval: "))
            Kend = float(input("Select End for K interval: "))
            Kstep = float(input("Select Step for K interval: "))
            Dstart = float(input("Select Start for D interval: "))
            Dend = float(input("Select End for D interval: "))
            Dstep = float(input("Select Step for D interval: "))
            comp = input("Select Component [" + compSelectString + "]: ")
            next(Loss_Function_Analysis_Simple(experimentClusterComp, comp, path, Kstart, Dstart, Kend, Dend, Kstep, Dstep, porosity, satur, lossFunctionSelection, factorSelection, solverSelection))
            intervalSelection = input("Continue with graphs?[Y - yes, N - no]")


        modelAnalysisSelection = input("Print solver graph?[Y - yes, N - no]")
        while modelAnalysisSelection == "Y":
            component = input("Select Component [" + compSelectString + "]: ")
            experimentString = "Select Experiment:\n"
            index = 0
            for comp in experimentClusterComp.clusters[component]:
                index += 1
                head, tail = os.path.split(comp.experiment.metadata.path)
                experimentString += str(index) + " - " + tail + "\n"
            experimentSelection = int(input(experimentString))
            paramSelection = "Y"
            while paramSelection == "Y":
                porosity = float(input("Select Porosity: "))
                satur = 0
                if solverSelection == "Nonlin":
                    satur = float(input("Select Saturation Coefficient: "))
                K = float(input("Select K: "))
                D = float(input("Select D: "))
                params = [porosity, K, D, satur]
                Model_Analysis(experimentClusterComp.clusters[component][experimentSelection-1], solverSelection, params)
                paramSelection = input("Try different parameters?[Y - yes, N - no]")
            modelAnalysisSelection = input("Continue with graphs?[Y - yes, N - no]")

        porosityDict = dict()
        KDQDict = dict()
        porosityStart = float(input("Select Start for Porosity interval: "))
        porosityEnd = float(input("Select End for Porosity interval: "))
        porosityDict["init"] = float(input("Select an Initial guess for Porosity interval: "))
        porosityDict["range"] = [porosityStart, porosityEnd]
        for key in experimentClusterComp.clusters:
            tmpDict = dict()
            KStart = float(input("Select Start for K interval for component " + key + ": "))
            KEnd = float(input("Select End for K interval for component " + key + ": "))
            tmpDict["kinit"] = float(input("Select Initial guess for K interval for component " + key + ": "))
            tmpDict["krange"] = [KStart, KEnd]
            DStart = float(input("Select Start for D interval for component " + key + ": "))
            DEnd = float(input("Select End for D interval for component " + key + ": "))
            tmpDict["dinit"] = float(input("Select Initial guess for D interval for component " + key + ": "))
            tmpDict["drange"] = [DStart, DEnd]
            if solverSelection == "Nonlin":
                QStart = float(input("Select Start for Q interval for component " + key + ": "))
                QEnd = float(input("Select End for Q interval for component " + key + ": "))
                tmpDict["qinit"] = float(input("Select Initial guess for Q interval for component " + key + ": "))
                tmpDict["qrange"] = [QStart, QEnd]
            else:
                tmpDict["qinit"] = 0
                tmpDict["qrange"] = [0, 0]
            KDQDict[key] = tmpDict
        result = Bilevel_Optim(currentExperimentSet, experimentClusterComp, porosityDict, KDQDict, lossFunctionSelection, factorSelection, solverSelection)
        self.Save_Result(result, path)
        chromatogramSelection = input("Create Chromatogram?[Y - yes, N - no]")
        if chromatogramSelection == "Y":
            directory = "Chromatograms"
            dirPath = os.path.join(path, directory)
            Handle_File_Creation(dirPath, True)
            while chromatogramSelection == "Y":
                comp = input("Select Component [" + compSelectString + "]: ")
                conditionSelectionString = "Select one of the following conditions:\n"
                for index, compObj in enumerate(experimentClusterComp.clusters[comp]):
                    conditionSelectionString += str(index + 1) + \
                                                " - Column diameter: " + \
                                                str(compObj.experiment.experimentCondition.columnDiameter) + \
                                                ", Column length: " + \
                                                str(compObj.experiment.experimentCondition.columnLength) + \
                                                ", Feed time: " + \
                                                str(compObj.experiment.experimentCondition.feedTime) + \
                                                ", Feed Volume: " + \
                                                str(compObj.experiment.experimentCondition.feedVolume) + \
                                                ", Dead Volume: " + \
                                                str(compObj.experiment.experimentCondition.deadVolume) + \
                                                ", Flow rate: " + \
                                                str(compObj.experiment.experimentCondition.flowRate) + \
                                                "\n"
                conditionSelection = int(input(conditionSelectionString)) - 1
                saturVal = 0
                if solverSelection == "Nonlin":
                    saturVal = result["compparams"][comp][2]
                selectedComp = experimentClusterComp.clusters[comp][conditionSelection]
                solverOutput = Solver_Choice(solverSelection,
                                             [result["porosity"], result["compparams"][comp][0], result["compparams"][comp][1], saturVal],
                                             selectedComp)[0][:, -1]
                df = experimentClusterComp.clusters[comp][conditionSelection].concentrationTime
                minTime = df.iat[0, 0]
                maxTime = df.iat[-1, 0]
                time = np.linspace(minTime, maxTime, 3000)
                resultDF = pd.DataFrame({'time': time, 'concentration': solverOutput})
                fileName = "chromatogram_" + comp + "_" + str(conditionSelection)
                filePath = dirPath + "\\" + fileName + ".csv"
                resultDF.to_csv(filePath, index=False, compression=None)
                chromatogramSelection = input("More Chromatograms?[Y - yes, N - no]")

    def Web_Start(self, experimentSet, gauss, retCorr, massBal, lossFunc, solver, factor, Lvl1ParamDict, Lvl2ParamDict, spacialDiff, timeDiff, time, optimId, retThreshold, lvl1optim, lvl2optim, optimType, fixporosity) :
        currentExperimentSet = self.Preprocess(experimentSet, gauss, retCorr, massBal, retThreshold)
        experimentClusterComp = self.Cluster_By_Component(currentExperimentSet)
        lossFunctionSelection = lossFunc
        solverSelection = solver
        factorSelection = factor
        result = Bilevel_Optim(currentExperimentSet, experimentClusterComp, Lvl1ParamDict, Lvl2ParamDict, lossFunctionSelection, factorSelection, solverSelection, spacialDiff, timeDiff, time, optimId, lvl1optim, lvl2optim, optimType, fixporosity)
        if retCorr:
            result["shifts"] = {}
            for exp in currentExperimentSet.experiments:
                head, tail = os.path.split(exp.metadata.path)
                result["shifts"][tail] = exp.shift
        if massBal:
            result["originalFeedTimes"] = {}
            result["newFeedTimes"] = {}
            for exp in currentExperimentSet.experiments:
                head, tail = os.path.split(exp.metadata.path)
                result["originalFeedTimes"][tail] = exp.experimentCondition.originalFeedTime
                result["newFeedTimes"][tail] = exp.experimentCondition.feedTime
        result["optimparams"]["gauss"] = gauss
        result["optimparams"]["retCorr"] = retCorr
        result["optimparams"]["retThreshold"] = retThreshold
        result["optimparams"]["massBal"] = massBal
        return result



    #Start for testing purposes

    @timeit
    def Start_For_Testing(self):

        # !!!!!!!!!!!!! Tady vyplň !!!!!!!!!!!!!!!
        path = "C:\\Users\\Adam\\ChroMo\\data\\LossFunctionExperimentSet"
        experimentSet = self.Load_Experiment_Set(path)
        experimentSet = self.Preprocess(experimentSet, True, True, True, 0.005)
        experimentCluster = self.Cluster_By_Component(experimentSet)
        porosity = 0.4
        porosity_bounds = (0.1, 0.9)
        K_init_gal = 3
        K_bounds_gal = (0.1, 500)
        A_init_gal = 2
        A_bounds_gal = (0.1, 100)
        B_gal = 0.0007
        K_init_man = 3
        K_bounds_man = (0.1, 500)
        A_init_man = 2
        A_bounds_man = (0.1, 100)
        B_man = 0.0007
        loss_function_type = 'Squares' # 'Simple', 'LogSimple', 'LogSquares'
        solver = 'Lin' # 'Nonlin
        factor = 1
        optimInfo = {}
        optimInfo["algorithm"] = "2" # "1" = bruteforce, "2" = neldermead, "3" = shgo
        optimInfo["settings"] = {}
        optimInfo["settings"]["maxiter"] = 0
        optimInfo["settings"]["maxfev"] = 0
        optimInfo["settings"]["xatol"] = 0
        optimInfo["settings"]["fatol"] = 0
        optimInfo["settings"]["aptive"] = False

        spacialDiff = 30
        timeDiff = 3000
        time = 10800

        # settings for brureforce...
        # optimInfo["settings"]["Ns"] = 0

        # settings for shgo...
        # optimInfo["settings"]["n"] = 100
        # optimInfo["settings"]["iters"] = 1
        # optimInfo["settings"]["maxev"] = 0
        # optimInfo["settings"]["maxiter"] = 0
        # optimInfo["settings"]["maxfev"] = 0
        # optimInfo["settings"]["maxtime"] = 0
        # optimInfo["settings"]["f_tol"] = 0
        # optimInfo["settings"]["f_min"] = 0

        B = [B_man, B_gal]

        fixPorosity = False
        fixA = False

        if not fixPorosity and not fixA:
            params = [porosity, K_init_gal, A_init_gal, K_init_man, A_init_man]
            bnds = [porosity_bounds, K_bounds_gal, A_bounds_gal, K_bounds_man, A_bounds_man]
            args = (B, loss_function_type, experimentCluster, solver, factor, spacialDiff, timeDiff, time)
        elif fixPorosity and not fixA:
            params = [K_init_gal, A_init_gal, K_init_man, A_init_man]
            bnds = [K_bounds_gal, A_bounds_gal, K_bounds_man, A_bounds_man]
            args = (B, loss_function_type, experimentCluster, solver, factor, spacialDiff, timeDiff, time, porosity)
        elif not fixPorosity and fixA:
            params = [porosity, K_init_gal, K_init_man]
            bnds = [porosity_bounds, K_bounds_gal, K_bounds_man]
            args = (B, loss_function_type, experimentCluster, solver, factor, spacialDiff, timeDiff, time, 0, [A_init_gal, A_init_man])
        else:
            params = [K_init_gal, K_init_man]
            bnds = [K_bounds_gal, K_bounds_man]
            args = (B, loss_function_type, experimentCluster, solver, factor, spacialDiff, timeDiff, time, porosity, [A_init_gal, A_init_man])

        result = handle_Optim_Settings(Multi_Loss_Function_Wrapper, params, args, bnds, optimInfo)
        if optimInfo["algorithm"] == "1":
            paramResult = result[0]
            lossFuncValResult = result[1]
        else:
            paramResult = result.x
            lossFuncValResult = result.fun
        print(paramResult)
        print(lossFuncValResult)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        #Nonlin_Solver()
        #path = "C:\\Users\\z004d8nt\\PycharmProjects\\ChoMo\\data\\TestExperimentSet1"
        #Single_Loss_Function(experimentSet.experiments[0])
        #experimentSetCopy = Fit_Gauss(experimentSetCopy)
        #Compare_ExperimentSets(experimentSet, experimentSetCopy)

        #comp = experimentSetCopy.experiments[0].experimentComponents[0]
        #cond = experimentSetCopy.experiments[0].experimentCondition
        #print(cond.flowRate, cond.columnLength, cond.columnDiameter, cond.feedVolume, comp.feedConcentration)
        #res = Lin_Solver(cond.flowRate, cond.columnLength, cond.columnDiameter, cond.feedVolume, comp.feedConcentration, 0.52 ,12000,  8000, debugPrint=True, debugGraph=True)
        # C:\Users\Adam\ChroMo\data\LossFunctionSingleExperiment
        # C:\Users\Adam\ChroMo\data\LossFunctionExperimentSet

        '''for experiment in experimentSet.experiments:
            head, tail = os.path.split(experiment.metadata.path)
            print(tail)
            print(experiment.experimentCondition.feedTime)
            for component in experiment.experimentComponents:
                head, tail = os.path.split(experiment.metadata.path)
                filename, extesion = os.path.splitext(tail)
                filepath = head + "\\" + filename + "_" + component.name + "_preprocessed.csv"
                component.concentrationTime.to_csv(filepath, index=False, compression=None)
            '''

        #Solver_Analysis(experimentSet, ["Glc", "Sac", "ManOH"], [[0.2, 10, 10], [0.2, 10, 10], [0.2, 10, 10]], "Lin")
        '''solution = Solution()
        for exp in experimentSet.experiments:
            for comp in exp.experimentComponents:
                solution.Add_Result(comp.name, comp.experiment.metadata.path, random.random(), random.random(), random.random())
        solution.Export_To_CSV("C:\\Users\\Adam\\ChroMo\\testSolution.csv")'''
        '''experimentSetCopy = Deep_Copy_ExperimentSet(experimentSet)
        experimentSetGauss = Fit_Gauss(experimentSetCopy)
        experimentSetCompCond = self.Cluster_By_Condition2(experimentSetGauss)
        experimentSetCor2 = Ret_Time_Cor(experimentSetGauss, experimentSetCompCond)
        experimentSetCor3 = Mass_Balance_Cor(experimentSetCor2)
        #Compare_ExperimentSets(experimentSet, experimentSetCor3)
        #expIso = Select_Iso_Exp(experimentSetCopy, experimentClusterComp)
        #testRes = Iso_Decision(expIso, [0.3, 15, 15])
        #print(testRes)
        experimentClusterComp = self.Cluster_By_Component(experimentSetCor2)
        Loss_Function_Analysis(experimentClusterComp, component='Glc', ystart=0, yend=76, ystep=1.5, xstart=0, xend=201, xstep=4, porosityStart=0.4, porosityStep=0.2, lossFunctionChoice="Squares", logScale=True)
        Loss_Function_Analysis(experimentClusterComp, component='Glc', ystart=0, yend=76, ystep=1.5, xstart=0, xend=401, xstep=8, porosityStart=0.6, porosityStep=0.2, lossFunctionChoice="Squares", logScale=True)
        '''
        #experimentClusterCompCond = self.Cluster_By_Condition2(experimentSetCopy)
        #experimentSetCopy = Ret_Time_Cor(experimentSetCopy, experimentClusterCompCond)
        #experimentSetCopy = Mass_Balance_Cor(experimentSetCopy, experimentSetCopy)

        #comp = experimentSetCopy.experiments[0].experimentComponents[2].concentrationTime
        #cond = experimentSetCopy.experiments[0].experimentCondition
        #plt.scatter(comp['time'],comp['Fru'])
        #plt.show()
        #comp.plot(x ='Time', y='Sac')
        #plt.show()
        #t = np.linspace(0, 10800, 5000)
        #plt.plot(t, res[:, -1])
        #plt.show()
        #result = Bilevel_Optim(experimentSetCopy, experimentClusterComp)
        #result = Lev2_Optim([150, 235, 16, 3, 150e-3], experimentClusterComp.clusters['Sac'])
        #print(result)
        #expIso = Select_Iso_Exp(experimentSetCopy, experimentClusterComp)
        #experimentSetCor1 = Mass_Balance_Cor(experimentSet, experimentSet)

    def Save_Result(self, result, parentDir):
        directory = "Result"
        dirPath = os.path.join(parentDir, directory)
        Handle_File_Creation(dirPath, True)
        fileName = "Result.txt"
        filePath = os.path.join(dirPath, fileName)
        file = Handle_File_Creation(filePath)
        file.write("Final Porosity: " + str(result["porosity"]) + "\nFinal Lv2 Loss Function Value: " + str(result["lv1lossfunctionval"]) + "\n\n")
        for comp in result["compparams"]:
            file.write("For component " + comp + ":\n")
            file.write("    Final K value: " + str(result["compparams"][comp][0]) + "\n")
            file.write("    Final D value: " + str(result["compparams"][comp][1]) + "\n")
            if len(result["compparams"][comp]) > 2 and result["compparams"][comp][2] != 0:
                file.write("    Final Q value: " + str(result["compparams"][comp][2]) + "\n")
            file.write("    Final Lv2 Loss Function Value: " + str(result["lv2lossfunctionvals"][comp]) + "\n\n")
        file.close()

    def Save_Graphs_To_Directory(self, experimentSet, experimentSetGauss, parentDir):
        directory = "Gauss_graphs"
        path = os.path.join(parentDir, directory)
        Handle_File_Creation(path, True)
        figNum = 1
        for exp1, expG in zip(experimentSet.experiments, experimentSetGauss.experiments):
            head, tail = os.path.split(exp1.metadata.path)
            filename, extesion = os.path.splitext(tail)
            for comp1, compG in zip(exp1.experimentComponents, expG.experimentComponents):
                plt.figure(figNum)
                figNum += 1
                time1 = comp1.concentrationTime.iloc[:, 0].to_numpy()
                conc1 = comp1.concentrationTime.iloc[:, 1].to_numpy()
                timeG = compG.concentrationTime.iloc[:, 0].to_numpy()
                concG = compG.concentrationTime.iloc[:, 1].to_numpy()
                plt.plot(time1, conc1, label = "original data")
                plt.plot(timeG, concG, label = "Gauss curve")
                graphName = path + "//Gauss_" + filename + "_" + comp1.name + ".png"
                plt.legend()
                plt.savefig(graphName)
        plt.close("all")

    def Load_Experimet_JSON(self, experimentSet, jsonString):
        jsonDict = json.loads(jsonString)
        description = jsonDict["description"]
        experimentDate = jsonDict["experimentDate"]
        columnLength = float(jsonDict["columnLength"])
        columnDiameter = float(jsonDict["columnDiameter"])
        flowRate = float(jsonDict["flowRate"])
        deadVolume = float(jsonDict["deadVolume"])
        feedVolume = float(jsonDict["feedVolume"])
        experiment = Experiment()
        experiment.metadata.date = experimentDate
        experiment.metadata.description = description
        experiment.metadata.path = jsonDict["name"]
        experiment.experimentCondition.flowRate = float(flowRate)
        experiment.experimentCondition.deadVolume = float(deadVolume)
        experiment.experimentCondition.feedVolume = float(feedVolume)
        experiment.experimentCondition.columnLength = float(columnLength)
        experiment.experimentCondition.columnDiameter = float(columnDiameter)
        for compDict in jsonDict["components"]:
            experimentComponent = ExperimentComponent()
            experimentComponent.concentrationTime = pd.read_json(compDict["concentrationTime"], orient="split")
            experimentComponent.name = compDict["name"]
            experimentComponent.feedConcentration = float(compDict["feedConcentration"])
            experimentComponent.experiment = experiment
            experiment.experimentComponents.append(experimentComponent)
        experimentSet.experiments.append(experiment)


    '''def Load_Experiment_Set(self, path):
        experimentSet = ExperimentSet()
        experimentSet.metadata.path = path
        experimentSet.metadata.date = datetime.date.today().strftime("%m/%d/%Y")
        filenames = next(walk(path), (None, None, []))[2]
        for file in filenames:
            if not (file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".csv")):
                continue
            df = pd.read_excel(path + '/' + file)
            description = df.iat[0, 3]
            date = df.iat[2, 3]
            columnLength = float(df.iat[0, 1])
            columnDiameter = float(df.iat[1, 1])
            flowRate = float(df.iat[2, 1])
            feedVolume = float(df.iat[3, 1])
            deadVolume = float(df.iat[4, 1])
            columnNames = df.iloc[[7]].to_numpy()[0]
            feedConcentrations = df.iloc[[6]].replace(',','.', regex=True).to_numpy()[0][1:]
            df.drop([0, 1, 2, 3, 4, 5, 6, 7], axis=0, inplace=True)
            df.columns = columnNames
            while(not isinstance(df.columns[-1], str) ):
                df.drop(columns=df.columns[-1],  axis=1,  inplace=True)
            df = df.replace(',','.', regex=True).astype(float)
            experiment = Experiment()
            experiment.metadata.date = date
            experiment.metadata.description = description
            experiment.metadata.path = path + "\\" + file
            experiment.experimentCondition.flowRate = float(flowRate)
            experiment.experimentCondition.feedVolume = float(feedVolume)
            experiment.experimentCondition.deadVolume = float(deadVolume)
            experiment.experimentCondition.columnLength = float(columnLength)
            experiment.experimentCondition.columnDiameter = float(columnDiameter)
            for index in range(df.columns[1:].size):
                experimentComponent = ExperimentComponent()
                experimentComponent.concentrationTime = df.iloc[:, [0, 1 + index]].astype(float)
                experimentComponent.concentrationTime['Time'] = experimentComponent.concentrationTime['Time'].apply(lambda x: x*60)
                experimentComponent.concentrationTime.reset_index(inplace=True, drop=True)
                experimentComponent.name = columnNames[1+index]
                experimentComponent.feedConcentration = float(feedConcentrations[index])
                experimentComponent.experiment = experiment
                experiment.experimentComponents.append(experimentComponent)
            experimentSet.experiments.append(experiment)
        return experimentSet'''

    def Load_Experiment_Set(self, path):
        experimentSet = ExperimentSet()
        experimentSet.metadata.path = path
        experimentSet.metadata.date = datetime.date.today().strftime("%m/%d/%Y")
        filenames = next(os.walk(path), (None, None, []))[2]
        for file in filenames:
            if not (file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".csv")):
                continue
            if file.startswith("~$"):
                print(f"Skipping temporary file: {file}")
                continue
            print(f"Loading file: {file}")
            df = pd.read_excel(os.path.join(path, file))
            description = df.iat[0, 3]
            date = df.iat[2, 3]
            columnLength = float(df.iat[0, 1])
            columnDiameter = float(df.iat[1, 1])
            flowRate = float(df.iat[2, 1])
            feedVolume = float(df.iat[3, 1])
            deadVolume = float(df.iat[4, 1])
            columnNames = df.iloc[[7]].to_numpy()[0]
            feedConcentrations = df.iloc[[6]].replace(',','.', regex=True).to_numpy()[0][1:]
            df.drop([0, 1, 2, 3, 4, 5, 6, 7], axis=0, inplace=True)
            df.columns = columnNames
            while not isinstance(df.columns[-1], str):
                df.drop(columns=df.columns[-1], axis=1, inplace=True)
            df = df.replace(',', '.', regex=True).astype(float)
            experiment = Experiment()
            experiment.metadata.date = date
            experiment.metadata.description = description
            experiment.metadata.path = os.path.join(path, file)
            experiment.experimentCondition.flowRate = float(flowRate)
            experiment.experimentCondition.feedVolume = float(feedVolume)
            experiment.experimentCondition.deadVolume = float(deadVolume)
            experiment.experimentCondition.columnLength = float(columnLength)
            experiment.experimentCondition.columnDiameter = float(columnDiameter)
            for index in range(df.columns[1:].size):
                experimentComponent = ExperimentComponent()
                experimentComponent.concentrationTime = df.iloc[:, [0, 1 + index]].astype(float)
                experimentComponent.concentrationTime['Time'] = experimentComponent.concentrationTime['Time'].apply(lambda x: x * 60)
                experimentComponent.concentrationTime.reset_index(inplace=True, drop=True)
                experimentComponent.name = columnNames[1 + index]
                experimentComponent.feedConcentration = float(feedConcentrations[index])
                experimentComponent.experiment = experiment
                experiment.experimentComponents.append(experimentComponent)
            experimentSet.experiments.append(experiment)
        return experimentSet

    def Preprocess(self, experimentSet, gauss, retCorr, massBal, retThreshold = 0):
        currExperimentSet = Dead_Volume_Preprocess(experimentSet)
        if gauss:
            currExperimentSet = Fit_Gauss(currExperimentSet)
        if retCorr:
            experimentClusterExp = self.Cluster_By_Experiment(currExperimentSet)
            currExperimentSet = Ret_Time_Cor(currExperimentSet, experimentClusterExp, threshold=retThreshold)
        if massBal:
            currExperimentSet = Mass_Balance_Cor(currExperimentSet)
        return currExperimentSet

    def Cluster_By_Component(self, experimentSet):
        componentDict = {}
        for experiment in experimentSet.experiments:
            for component in experiment.experimentComponents:
                if component.name in componentDict:
                    componentDict[component.name].append(component)
                else:
                    componentDict[component.name] = list()
                    componentDict[component.name].append(component)
        clusters = ExperimentClusters()
        clusters.metadata = experimentSet.metadata
        clusters.clusters = componentDict
        clusters.metadata.description += "\nClusters by component"
        return clusters

    def Cluster_By_Condition(self, experimentSet):
        clusterByCondition = ExperimentClusters()
        clusterByCondition.metadata = experimentSet.metadata
        clusterByCondition.metadata.description += "\nClusters by condition"
        for experiment in experimentSet.experiments:
            for component in experiment.experimentComponents:
                foundFlag = False
                for key, value in clusterByCondition.clusters.items():
                    if self.Cluster_Match(value[0], component):
                        value.append(component)
                        foundFlag = True
                if not foundFlag:
                    clusterByCondition.clusters[self.Create_Key(component)] = list()
                    clusterByCondition.clusters[self.Create_Key(component)].append(component)
        return clusterByCondition

    def Cluster_By_Condition2(self, experimentSet):
        clusterByCondition = ExperimentClusters()
        clusterByCondition.metadata = experimentSet.metadata
        clusterByCondition.metadata.description += "\nClusters by condition"
        tmpCompList = list()
        for experiment in experimentSet.experiments:
            tmpCompList = tmpCompList + experiment.experimentComponents
        while len(tmpCompList) > 0:
            maxCount = 0
            maxCluster = list()
            for experimentComp in tmpCompList:
                tmpCluster = list()
                tmpCluster.append(experimentComp)
                for experimentComp2 in tmpCompList:
                    if experimentComp != experimentComp2 and self.Cluster_Match(experimentComp, experimentComp2):
                        tmpCluster.append(experimentComp2)
                if len(tmpCluster) > maxCount:
                    maxCount = len(tmpCluster)
                    maxCluster = tmpCluster
            clusterByCondition.clusters[self.Create_Key(maxCluster[0])] = maxCluster
            tmpCompList = [x for x in tmpCompList if x not in maxCluster]
        return clusterByCondition

    def Cluster_By_Experiment(self, experimentSet):
        clusterByExperiment = ExperimentClusters()
        clusterByExperiment.metadata = experimentSet.metadata
        clusterByExperiment.metadata.description += "\nClusters by experiments"
        tmpCompList = experimentSet.experiments
        dictKey = 0
        while len(tmpCompList) > 0:
            maxCount = 0
            maxCluster = list()
            for experiment in tmpCompList:
                tmpCluster = list()
                tmpCluster.append(experiment)
                for experiment2 in tmpCompList:
                    if experiment != experiment2 and self.Cluster_Match_Exp(experiment, experiment2):
                        tmpCluster.append(experiment2)
                if len(tmpCluster) > maxCount:
                    maxCount = len(tmpCluster)
                    maxCluster = tmpCluster
            clusterByExperiment.clusters[dictKey] = [maxCluster]
            tmpCompList = [x for x in tmpCompList if x not in maxCluster]
            dictKey += 1
        for key, value in clusterByExperiment.clusters.items():
            componentDict = {}
            for experiment in value[0]:
                for component in experiment.experimentComponents:
                    if component.name in componentDict:
                        componentDict[component.name].append(component)
                    else:
                        componentDict[component.name] = list()
                        componentDict[component.name].append(component)
            clusterByExperiment.clusters[key].append(componentDict)
        return clusterByExperiment


    """
    Calculates if comp2 is close to comp1 with tolerance(default 0.05)
    """
    def Cluster_Match(self, comp1, comp2, tolerance = 0.01):
        print("Calling Cluster_Match")
        cond1 = comp1.experiment.experimentCondition
        cond2 = comp2.experiment.experimentCondition
        if(comp1.name == comp2.name and
                abs(comp1.feedConcentration - comp2.feedConcentration) < tolerance * comp1.feedConcentration and
                abs(cond1.flowRate - cond2.flowRate) < tolerance * cond1.flowRate and
                abs(cond1.columnDiameter - cond2.columnDiameter) < tolerance * cond1.columnDiameter and
                abs(cond1.columnLength - cond2.columnLength) < tolerance * cond1.columnLength):
            return True
                #abs(cond1.feedVolume - cond2.feedVolume) < tolerance * cond1.feedVolume):
        return False

    """
    Calculates if exp2 is close to exp1 with tolerance(default 0.01)
    """
    def Cluster_Match_Exp(self, exp1, exp2, tolerance = 0.01):
        cond1 = exp1.experimentCondition
        cond2 = exp2.experimentCondition
        if(abs(cond1.flowRate - cond2.flowRate) < tolerance * cond1.flowRate and
           abs(cond1.columnDiameter - cond2.columnDiameter) < tolerance * cond1.columnDiameter and
           abs(cond1.columnLength - cond2.columnLength) < tolerance * cond1.columnLength):
            return True
        return False

    """
    Creates key for cluster dictionary from component
    """
    def Create_Key(self, comp):
        name = comp.name
        feedConc = str(comp.feedConcentration)
        colDia = str(comp.experiment.experimentCondition.columnDiameter)
        colLen = str(comp.experiment.experimentCondition.columnLength)
        feedVol = str(comp.experiment.experimentCondition.feedVolume)
        flowRate = str(comp.experiment.experimentCondition.flowRate)
        return ":".join([name, feedConc, colDia, colLen, feedVol, flowRate])

    def Print_Component_Graph(self, experimentComponent):
        experimentComponent.concentrationTime.plot.line(x='Time')
        plt.show()
