import os
import threading
import time
import datetime
import zipfile
from operator import add
import flask_login
import pandas as pd
import numpy as np
from flask import Flask, flash, request, redirect, url_for, send_file, render_template, json
import mongoengine as me
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
import matplotlib.pyplot as plt
from functions.WebServerStuff.Serialize_File_To_JSON import Serialize_File_To_JSON
from functions.Loss_Function_Analysis_Simple import Loss_Function_Analysis_Simple
from functions.Model_Analysis import Model_Analysis
from functions.solvers.Solver_Choice import Solver_Choice
from objects.Operator import Operator
from objects.ExperimentSet import ExperimentSet
from os import walk
from waitress import serve

def Web_Server():
    """Class implementing all the web server functionalities."""
    matplotlib.use('Agg')

    SOLVER_TIME = 10800
    SOLVER_TIME_DIFF = 3000
    SOLVER_SPACIAL_DIFF = 30

    plotFileCounter = 1
    numberOfRunningOptims = 0
    experimentSet = {}
    clusterComp = {}
    compList = {}
    formInfos = {}
    timers = {}
    operator = Operator()


    BASE_FOLDER = os.getcwd()
    UPLOAD_FOLDER = BASE_FOLDER + '/data/TestUploadFolder'

    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

    api = Flask(__name__)
    api.config['SECRET_KEY'] = 'super secret key'
    api.config['SECURITY_PASSWORD_SALT'] = 'super secret salt'
    api.config['SESSION_TYPE'] = 'development'
    api.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    api.config['MONGODB_DB'] = 'ChroMo'
    api.config['MONGODB_HOST'] = '127.0.0.1'
    api.config['MONGODB_PORT'] = 27017
    login_manager = flask_login.LoginManager()
    login_manager.init_app(api)
    me.connect('ChroMo', host='127.0.0.1', port=27017)

    uploadedFiles = {}
    exporting_threads = {}

    db_mutex = threading.Lock()

    class DBExperiment(me.Document):
        uniquename = me.StringField(required=True, unique=True)
        name = me.StringField(required=True)
        experiment = me.StringField(required=True)

    class DBResult(me.Document):
        thr_id = me.IntField(required=True, unique=True)
        name = me.StringField(required=True)
        experiments = me.ListField(me.StringField())
        results = me.DictField(required=True)
        time = me.StringField()


    class DBUser(me.Document):
        username = me.StringField(required=True, unique=True)
        password_hash = me.StringField()
        results = me.ListField(me.ReferenceField(DBResult))
        experiments = me.ListField(me.ReferenceField(DBExperiment))

    class User(flask_login.UserMixin):
        pass

    tmpcount = DBResult.objects.count()
    threadCounter = 0
    if tmpcount > 0:
        threadCounter = DBResult.objects[tmpcount-1].thr_id + 1

    @login_manager.user_loader
    def user_loader(username):
        for dbuser in DBUser.objects:
            if username == dbuser.username:
                user = User()
                user.id = username
                user.db = dbuser
                return user
        return

    @login_manager.request_loader
    def request_loader(request):
        username = request.form.get('username')
        for dbuser in DBUser.objects:
            if username == dbuser.username:
                user = User()
                user.id = username
                user.db = dbuser
                return user
        return

    @login_manager.unauthorized_handler
    def unauthorized_handler():
        print(flask_login.current_user.id)
        return 'Unauthorized', 401

    class MainWorkThread(threading.Thread):
        nonlocal experimentSet, formInfos, timers

        def __init__(self, user_id, thr_id, dbuser):
            self.user_id = user_id
            self.thr_id = thr_id
            self.dbuser = dbuser
            self.result = "-"
            super().__init__()

        def run(self):
            nonlocal numberOfRunningOptims, db_mutex
            numberOfRunningOptims += 1
            try:
                formInfo = formInfos[self.user_id]
                usedExpSet = experimentSet[self.user_id]
                Lvl1ParamDict = {}
                if formInfo["optimType"] != "singlelevel":
                    Lvl1ParamDict["pinit"] = formInfo["porosity"]
                    if not formInfo["fixporosity"]:
                        Lvl1ParamDict["prange"] = [formInfo["porosityStart"], formInfo["porosityEnd"]]
                    if formInfo["optimType"] == "calcDisper":
                        Lvl1ParamDict["ainit"] = formInfo["A"]
                        Lvl1ParamDict["arange"] = [formInfo["AStart"], formInfo["AEnd"]]
                Lvl2ParamDict = {}
                for comp in compList[self.user_id]:
                    tmpDict = {}
                    if formInfo["optimType"] == "singlelevel":
                        tmpDict["pinit"] = formInfo[comp + "P"]
                        if not formInfo["fixporosity"]:
                            tmpDict["prange"] = [formInfo[comp + "PStart"], formInfo[comp + "PEnd"]]
                    tmpDict["kinit"] = formInfo[comp + "K"]
                    tmpDict["krange"] = [formInfo[comp + "KStart"], formInfo[comp + "KEnd"]]
                    if formInfo["optimType"] != "calcDisper" and formInfo["optimType"] != "calcDisper2":
                        tmpDict["dinit"] = formInfo[comp + "D"]
                        tmpDict["drange"] = [formInfo[comp + "DStart"], formInfo[comp + "DEnd"]]
                    else:
                        tmpDict["b"] = formInfo[comp + "B"]
                        if formInfo["optimType"] == "calcDisper2":
                            tmpDict["ainit"] = formInfo[comp + "A"]
                            tmpDict["arange"] = [formInfo[comp + "AStart"], formInfo[comp + "AEnd"]]
                    if formInfo["solver"] == "Nonlin":
                        tmpDict["qinit"] = formInfo[comp + "Q"]
                        tmpDict["qrange"] = [formInfo[comp + "QStart"], formInfo[comp + "QEnd"]]
                    else:
                        tmpDict["qinit"] = 0
                        tmpDict["qrange"] = [0, 0]
                    Lvl2ParamDict[comp] = tmpDict
                tmp = operator.Web_Start(usedExpSet,
                    formInfo["gauss"], formInfo["retCorr"], formInfo["massBal"], formInfo["lossFunc"],
                    formInfo["solver"], formInfo["factor"], Lvl1ParamDict, Lvl2ParamDict, formInfo["spacialDiff"],
                    formInfo["timeDiff"], formInfo["time"], self.thr_id, formInfo["retCorrThreshold"],
                    formInfo["lvl1optimsettings"], formInfo["lvl2optimsettings"], formInfo["optimType"], formInfo["fixporosity"])
                if formInfo["retCorr"]:
                    formInfo["shifts"] = tmp["shifts"]
                else:
                    formInfo["shifts"] = None
                if formInfo["massBal"]:
                    formInfo["originalFeedTimes"] = tmp["originalFeedTimes"]
                    formInfo["newFeedTimes"] = tmp["newFeedTimes"]
                else:
                    formInfo["originalFeedTimes"] = None
                timer = time.time() - timers[self.thr_id]
                newResult = DBResult(thr_id=self.thr_id, results = tmp, name=self.name, experiments=[os.path.split(exp.metadata.path)[1] for exp in usedExpSet.experiments], time=str(datetime.timedelta(seconds=timer)))
                print("thr_id: ", self.thr_id)
                with db_mutex:
                    newResult.save()
                    self.dbuser.reload()
                    self.dbuser.results.append(newResult)
                    self.dbuser.save()
                self.result = tmp
            except Exception as e:
                print(e)
                self.result = "FAIL"
            numberOfRunningOptims -= 1

    '''class ApiWorkThread(threading.Thread):

        def __init__(self, data, thr_id):
            self.data = data
            self.thr_id = thr_id
            self.result = "-"
            super().__init__()

        def run(self):
            try:
                expSet = ExperimentSet()
                expSet.metadata.path = "database"
                expSet.metadata.date = datetime.date.today().strftime("%m/%d/%Y")
                for exp in self.data['experiments']:
                    operator.Load_Experimet_JSON(expSet, json.dumps(exp))
                gauss = bool(self.data['settings']['gauss'])
                retCorr = bool(self.data['settings']['retCorr'])
                retCorrThreshold = float(self.data['settings']['retCorrThreshold'])
                massBal = bool(self.data['settings']['massBal'])
                lossFunc = str(self.data['settings']['lossFunc'])
                solver = str(self.data['settings']['solver'])
                factor = int(self.data['settings']['factor'])
                porosityStart = float(self.data['settings']['porosityStart'])
                porosityEnd = float(self.data['settings']['porosityEnd'])
                porosity = float(self.data['settings']['porosityInit'])
                Lvl1ParamDict = {}
                if formInfo["optimType"] != "singlelevel":
                    Lvl1ParamDict["pinit"] = formInfo["porosity"]
                    Lvl1ParamDict["prange"] = [formInfo["porosityStart"], formInfo["porosityEnd"]]
                    if formInfo["optimType"] == "calcDisper":
                        Lvl1ParamDict["ainit"] = formInfo["A"]
                        Lvl1ParamDict["arange"] = [formInfo["AStart"], formInfo["AEnd"]]
                KDQDict = {}
                for key, val in self.data['settings']['components'].items():
                    tmpDict = {}
                    tmpDict["kinit"] = val["K"]
                    tmpDict["krange"] = [val["KStart"], val["KEnd"]]
                    tmpDict["dinit"] = val["D"]
                    tmpDict["drange"] = [val["DStart"], val["DEnd"]]
                    if solver == "Nonlin":
                        tmpDict["qinit"] = val["Q"]
                        tmpDict["Qrange"] = [val["QStart"], val["QEnd"]]
                    else:
                        tmpDict["qinit"] = 0
                        tmpDict["qrange"] = [0, 0]
                    KDQDict[key] = tmpDict
                lvl1optimsettings = self.data['settings']['lvl1optimsettings']
                lvl2optimsettings = self.data['settings']['lvl2optimsettings']
                spacialDiff = int(self.data['settings']['spacialDiff'])
                timeDiff = int(self.data['settings']['timeDiff'])
                time = float(self.data['settings']['time'])
                tmp = operator.Web_Start(expSet, gauss, retCorr, massBal, lossFunc,
                        solver, factor, porosityStart, porosityEnd,
                        porosity, KDQDict, spacialDiff,
                        timeDiff, time, self.thr_id, retCorrThreshold,
                        lvl1optimsettings, lvl2optimsettings)
                self.result = tmp
            except Exception as e:
                print(e)
                self.result = "FAIL"'''

    class SimpleThread(threading.Thread):
        nonlocal experimentSet, formInfos, clusterComp, BASE_FOLDER

        def __init__(self, user_id):
            self.user_id = user_id
            self.progress = "-"
            super().__init__()

        def run(self):
            nonlocal plotFileCounter
            try:
                plt.clf()
                formInfo = formInfos[self.user_id]
                if not self.user_id in clusterComp:
                    clusterComp[self.user_id] = operator.Cluster_By_Component(experimentSet[self.user_id])
                currExperimentSet2 = operator.Preprocess(experimentSet[self.user_id], False, False, False, 0)
                experimentClusterComp2 = operator.Cluster_By_Component(currExperimentSet2)
                params = [formInfo["porosity"], formInfo[formInfo["comp2"] + "K"], formInfo[formInfo["comp2"] + "D"],
                          formInfo["saturation"]]
                print("---------Not Preprocessed Output Start---------")
                Model_Analysis(experimentClusterComp2.clusters[formInfo["comp2"]][formInfo["exp" + formInfo["comp2"]]],
                               formInfo["solver"], params, formInfo["spacialDiff"], formInfo["timeDiff"],
                               formInfo["time"], webMode=True, title="Experimental data")
                print("---------Not Preprocessed Output End---------")
                filename = "plot" + str(plotFileCounter) + ".png"
                plotFileCounter += 1
                plt.savefig(BASE_FOLDER + '/functions/WebServerStuff/static/images/' + filename)
                plt.clf()
                currExperimentSet = operator.Preprocess(experimentSet[self.user_id], formInfo["gauss"], formInfo["retCorr"], formInfo["massBal"], formInfo["retCorrThreshold"])
                if formInfo["retCorr"]:
                    formInfo["shifts"] = {}
                    for exp in currExperimentSet.experiments:
                        head, tail = os.path.split(exp.metadata.path)
                        formInfo["shifts"][tail] = exp.shift
                else:
                    formInfo["shifts"] = None
                if formInfo["massBal"]:
                    formInfo["originalFeedTimes"] = {}
                    formInfo["newFeedTimes"] = {}
                    for exp in currExperimentSet.experiments:
                        head, tail = os.path.split(exp.metadata.path)
                        formInfo["originalFeedTimes"][tail] = exp.experimentCondition.originalFeedTime
                        formInfo["newFeedTimes"][tail] = exp.experimentCondition.feedTime
                else:
                    formInfo["originalFeedTimes"] = None
                    formInfo["newFeedTimes"] = None
                experimentClusterComp = operator.Cluster_By_Component(currExperimentSet)
                print("---------Preprocessed Output Start---------")
                Model_Analysis(experimentClusterComp.clusters[formInfo["comp2"]][formInfo["exp" + formInfo["comp2"]]],
                               formInfo["solver"], params, formInfo["spacialDiff"], formInfo["timeDiff"], formInfo["time"],
                               webMode=True, title="Preprocessed data", full=True)
                print("---------Preprocessed Output End---------")
                filenames = []
                fig_nums = plt.get_fignums()
                figs = [plt.figure(n) for n in fig_nums]
                for fig in figs:
                    filename2 = "plot" + str(plotFileCounter) + ".png"
                    filenames.append(filename2)
                    plotFileCounter += 1
                    fig.savefig(BASE_FOLDER + '/functions/WebServerStuff/static/images/' + filename2)
                    fig.clf()
                self.progress = [filename] + filenames
            except Exception as e:
                print(e)
                self.progress = "FAILED"

    class ExportingThread(threading.Thread):
        nonlocal experimentSet, formInfos

        def __init__(self, user_id, thr_id):
            self.progress = "Estimated time remaining: "
            self.user_id = user_id
            self.thr_id = thr_id
            super().__init__()

        def run(self):
            try:
                formInfo = formInfos[self.user_id]
                currExperimentSet = operator.Preprocess(experimentSet[self.user_id], formInfo["gauss"], formInfo["retCorr"], formInfo["massBal"], formInfo["retCorrThreshold"])
                if formInfo["retCorr"]:
                    formInfo["shifts"] = {}
                    for exp in currExperimentSet.experiments:
                        head, tail = os.path.split(exp.metadata.path)
                        formInfo["shifts"][tail] = exp.shift
                else:
                    formInfo["shifts"] = None
                if formInfo["massBal"]:
                    formInfo["originalFeedTimes"] = {}
                    formInfo["newFeedTimes"] = {}
                    for exp in currExperimentSet.experiments:
                        head, tail = os.path.split(exp.metadata.path)
                        formInfo["originalFeedTimes"][tail] = exp.experimentCondition.originalFeedTime
                        formInfo["newFeedTimes"][tail] = exp.experimentCondition.feedTime
                else:
                    formInfo["originalFeedTimes"] = None
                    formInfo["newFeedTimes"] = None
                experimentClusterComp = operator.Cluster_By_Component(currExperimentSet)
                self.generator = Loss_Function_Analysis_Simple(experimentClusterComp, formInfo["comp"], "", formInfo[formInfo["comp"] + "KStart"]
                                          , formInfo[formInfo["comp"] + "DStart"], formInfo[formInfo["comp"] + "KEnd"]
                                          , formInfo[formInfo["comp"] + "DEnd"], formInfo[formInfo["comp"] + "KStep"]
                                          , formInfo[formInfo["comp"] + "DStep"], formInfo["porosity"], formInfo["saturation"]
                                          , formInfo["lossFunc"], formInfo["factor"], formInfo["solver"], formInfo["spacialDiff"]
                                          ,formInfo["timeDiff"], formInfo["time"], True, self.thr_id)
            except Exception as e:
                print(e)
                self.progress = "FAIL"
                return
            for res in self.generator:
                self.progress = res
                if not type(res) is str:
                    break

        def new_angle(self):
            self.progress = next(self.generator)

    class SolverThread(threading.Thread):
        nonlocal experimentSet, formInfos

        def __init__(self, user_id, solver, params, experiment, compIdx, comp):
            self.user_id = user_id
            self.solver = solver
            self.params = params
            self.exp = experiment
            self.compIdx = compIdx
            self.comp = comp
            self.result = "-"
            super().__init__()

        def run(self):
            formInfo = formInfos[self.user_id]
            result = []
            print(self.params)
            if self.compIdx != "all":
                result.append(
                    Solver_Choice(self.solver, self.params, self.exp.experimentComponents[self.compIdx], formInfo["spacialDiff"], formInfo["timeDiff"],
                                  formInfo["time"]))
            else:
                for idx, comp in enumerate(self.exp.experimentComponents):
                        result.append(Solver_Choice(self.solver, self.params[idx], comp, formInfo["spacialDiff"], formInfo["timeDiff"], formInfo["time"]))
            self.result = result


    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @flask_login.login_required
    def fetchExperimentData():
        nonlocal experimentSet
        experimentSet[flask_login.current_user.id] = ExperimentSet()
        experimentSet[flask_login.current_user.id].metadata.path = "database"
        experimentSet[flask_login.current_user.id].metadata.date = datetime.date.today().strftime("%m/%d/%Y")
        for exp in flask_login.current_user.db.experiments:
            operator.Load_Experimet_JSON(experimentSet[flask_login.current_user.id], exp.experiment)


    @api.route('/projects/test2', methods=['GET'])
    @flask_login.login_required
    def get_projects_test2():
        nonlocal experimentSet, compList, formInfos, clusterComp
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        if not "spacialDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
        if not "timeDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
        if not "time" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
        formInfo = formInfos[flask_login.current_user.id]
        if not flask_login.current_user.id in experimentSet:
            fetchExperimentData()
        if not flask_login.current_user.id in clusterComp:
            clusterComp[flask_login.current_user.id] = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
        experimentClusterComp = clusterComp[flask_login.current_user.id]
        compList[flask_login.current_user.id] = experimentClusterComp.clusters.keys()
        compExperimentDict = {}
        for comp in compList[flask_login.current_user.id]:
            expList = []
            for comp2 in experimentClusterComp.clusters[comp]:
                head, tail = os.path.split(comp2.experiment.metadata.path)
                expList.append(tail)
            compExperimentDict[comp] = expList
        return render_template('ParamsTestForm2.html', compList = compList[flask_login.current_user.id], compExpDict = compExperimentDict, formInfo=formInfo, user = flask_login.current_user.id)

    @api.route('/projects/test2', methods=['POST'])
    @flask_login.login_required
    def post_projects_test2():
        nonlocal experimentSet, plotFileCounter, formInfos, threadCounter
        plt.clf()
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        if not "spacialDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
        if not "timeDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
        if not "time" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["gauss"] = bool(request.form.get("gaussTest"))
        formInfo["retCorr"] = bool(request.form.get("retCorrTest"))
        formInfo["retCorrThreshold"] = 0
        if formInfo["retCorr"]:
            formInfo["retCorrThreshold"] = float(request.form.get("retCorrThreshold"))
        formInfo["massBal"] = bool(request.form.get("massBalTest"))
        formInfo["solver"] = str(request.form.get("solverTest"))
        formInfo["porosity"] = float(request.form.get("porosityTest"))
        formInfo["saturation"] = 0
        if formInfo["solver"] == "Nonlin":
            formInfo["saturation"] = float(request.form.get("saturationTest"))
        formInfo["comp2"] = str(request.form.get("componentTest"))
        formInfo["exp" + formInfo["comp2"]] = int(request.form.get("expList" + formInfo["comp2"]))
        formInfo[formInfo["comp2"] + "K"] = float(request.form.get("KTest"))
        formInfo[formInfo["comp2"] + "D"] = float(request.form.get("DTest"))
        formInfo[formInfo["comp2"] + "Q"] = 0
        if formInfo["solver"] == "Nonlin":
            formInfo[formInfo["comp2"] + "Q"] = float(request.form.get("saturationTest"))
        thread_id = threadCounter
        threadCounter += 1
        if not thread_id in exporting_threads:
            print("ID: " + str(thread_id))
            exporting_threads[thread_id] = SimpleThread(flask_login.current_user.id)
            plt.clf()
            exporting_threads[thread_id].start()
        return url_for("post_projects_test2_progress", id=thread_id)

    @api.route('/projects/test2/<id>/progress', methods=['GET'])
    @flask_login.login_required
    def post_projects_test2_progress(id):
        nonlocal exporting_threads, plotFileCounter, formInfos
        formInfo = formInfos[flask_login.current_user.id]
        thread_id = int(id)
        if exporting_threads[thread_id].progress == "-":
            return ""
        else:
            returnString = ""
            for idx, filename in enumerate(exporting_threads[thread_id].progress):
                returnString += render_template('Picture.html', pictureURL=url_for('static', filename='images/' + filename),
                                   alt="chart", width="640", height="480")
                if idx == 0:
                    returnString += render_template('GraphInfo.html', shifts=formInfo["shifts"],
                                    originalFeedTimes=formInfo["originalFeedTimes"],
                                    newFeedTimes=formInfo["newFeedTimes"])
            return returnString

    @api.route('/projects/test2/continue', methods=['POST'])
    @flask_login.login_required
    def post_projects_test2_continue():
        nonlocal formInfos
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        if not "spacialDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
        if not "timeDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
        if not "time" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["gauss"] = bool(request.form.get("gaussTest"))
        formInfo["retCorr"] = bool(request.form.get("retCorrTest"))
        formInfo["retCorrThreshold"] = 0
        if formInfo["retCorr"]:
            formInfo["retCorrThreshold"] = float(request.form.get("retCorrThreshold"))
        formInfo["massBal"] = bool(request.form.get("massBalTest"))
        formInfo["solver"] = str(request.form.get("solverTest"))
        if request.form.get("porosityTest") != "":
            formInfo["porosity"] = float(request.form.get("porosityTest"))
        formInfo["comp2"] = str(request.form.get("componentTest"))
        if formInfo["solver"] == "Nonlin":
            if request.form.get("saturationTest") != "":
                formInfo["saturation"] = float(request.form.get("saturationTest"))
        formInfo["exp" + formInfo["comp2"]] = int(request.form.get("expList" + formInfo["comp2"]))
        if request.form.get("KTest") != "":
            formInfo[formInfo["comp2"] + "K"] = float(request.form.get("KTest"))
        if request.form.get("DTest") != "":
            formInfo[formInfo["comp2"] + "D"] = float(request.form.get("DTest"))
        return redirect(url_for('get_projects_params'))

    @api.route('/projects/test', methods=['GET'])
    @flask_login.login_required
    def get_projects_test():
        nonlocal experimentSet, compList, formInfos, clusterComp
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        if not "spacialDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
        if not "timeDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
        if not "time" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
        formInfo = formInfos[flask_login.current_user.id]
        if not flask_login.current_user.id in experimentSet:
            fetchExperimentData()
        if not flask_login.current_user.id in clusterComp:
            clusterComp[flask_login.current_user.id] = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
        experimentClusterComp = clusterComp[flask_login.current_user.id]
        compList[flask_login.current_user.id] = experimentClusterComp.clusters.keys()
        return render_template('ParamsTestForm.html', compList = compList[flask_login.current_user.id], formInfo=formInfo, user = flask_login.current_user.id)

    @api.route('/projects/test/<id>/newAngle', methods=['GET'])
    @flask_login.login_required
    def post_projects_test_new_angle(id):
        nonlocal exporting_threads, plotFileCounter, formInfos
        formInfo = formInfos[flask_login.current_user.id]
        thread_id = int(id)
        exporting_threads[thread_id].new_angle()
        progress = exporting_threads[thread_id].progress
        filename = "plot" + str(plotFileCounter) + ".png"
        plotFileCounter += 1
        plt.savefig(BASE_FOLDER + '/functions/WebServerStuff/static/images/' + filename)
        returnString = render_template('Picture.html', pictureURL = url_for('static', filename='images/'+filename), alt="chart", width="640", height="480", newAngleUrl=url_for('post_projects_test_new_angle', id=id)) + \
                       render_template('GraphInfo.html', henryConst=progress[0], disperCoef=progress[1], shifts=formInfo["shifts"], originalFeedTimes=formInfo["originalFeedTimes"], newFeedTimes=formInfo["newFeedTimes"] )
        return  returnString

    @api.route('/projects/test/<id>/progress', methods=['GET'])
    @flask_login.login_required
    def post_projects_test_progress(id):
        nonlocal exporting_threads, plotFileCounter, formInfos, BASE_FOLDER
        formInfo = formInfos[flask_login.current_user.id]
        thread_id = int(id)
        if type(exporting_threads[thread_id].progress) is str:
            return str(exporting_threads[thread_id].progress)
        else:
            progress = exporting_threads[thread_id].progress
            filename = "plot" + str(plotFileCounter) + ".png"
            plotFileCounter += 1
            plt.savefig(BASE_FOLDER + '/functions/WebServerStuff/static/images/' + filename)
            returnString = render_template('Picture.html', pictureURL = url_for('static', filename='images/'+filename),
                                           alt="chart", width="640", height="480", newAngleUrl=url_for('post_projects_test_new_angle', id=id),
                                           downloadUrl = url_for('get_projects_test_matrix', id=id)) + \
                           render_template('GraphInfo.html', henryConst=progress[0], disperCoef=progress[1], shifts=formInfo["shifts"],
                                           originalFeedTimes=formInfo["originalFeedTimes"], newFeedTimes=formInfo["newFeedTimes"] )
            return returnString


    @api.route('/projects/test/<id>/matrix', methods=['GET'])
    @flask_login.login_required
    def get_projects_test_matrix(id):
        nonlocal exporting_threads, plotFileCounter
        thread_id = int(id)
        if type(exporting_threads[thread_id].progress) is str:
            return str(exporting_threads[thread_id].progress)
        else:
            progress = exporting_threads[thread_id].progress
            filename = "table" + str(plotFileCounter) + ".csv"
            plotFileCounter += 1
            progress[2].to_csv('functions/WebServerStuff/static/tables/' + filename, index=False)
            return send_file('static/tables/' + filename)


    @api.route('/projects/test', methods=['POST'])
    @flask_login.login_required
    def post_projects_test():
        nonlocal formInfos, exporting_threads, threadCounter
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        if not "spacialDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
        if not "timeDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
        if not "time" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["gauss"] = bool(request.form.get("gaussTest"))
        formInfo["retCorr"] = bool(request.form.get("retCorrTest"))
        formInfo["retCorrThreshold"] = 0
        if formInfo["retCorr"]:
            formInfo["retCorrThreshold"] = float(request.form.get("retCorrThreshold"))
        formInfo["massBal"] = bool(request.form.get("massBalTest"))
        formInfo["lossFunc"] = str(request.form.get("lossFuncTest"))
        formInfo["solver"] = str(request.form.get("solverTest"))
        formInfo["factor"] = int(request.form.get("factorTest"))
        formInfo["porosity"] = float(request.form.get("porosityTest"))
        formInfo["saturation"] = 0
        if formInfo["solver"] == "Nonlin":
            formInfo["saturation"] = float(request.form.get("saturationTest"))
        formInfo["comp"] = str(request.form.get("componentTest"))
        formInfo[formInfo["comp"] + "KStart"] = float(request.form.get("KStartTest"))
        formInfo[formInfo["comp"] + "KEnd"] = float(request.form.get("KEndTest"))
        formInfo[formInfo["comp"] + "KStep"] = float(request.form.get("KStepTest"))
        formInfo[formInfo["comp"] + "DStart"] = float(request.form.get("DStartTest"))
        formInfo[formInfo["comp"] + "DEnd"] = float(request.form.get("DEndTest"))
        formInfo[formInfo["comp"] + "DStep"] = float(request.form.get("DStepTest"))
        thread_id = threadCounter
        threadCounter += 1
        if not thread_id in exporting_threads:
            print("ID: " + str(thread_id))
            exporting_threads[thread_id] = ExportingThread(flask_login.current_user.id, thread_id)
            plt.clf()
            exporting_threads[thread_id].start()
        return url_for("post_projects_test_progress", id=thread_id)

    @api.route('/projects/test/continue', methods=['POST'])
    @flask_login.login_required
    def post_projects_test_continue():
        nonlocal formInfos
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        if not "spacialDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
        if not "timeDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
        if not "time" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["gauss"] = bool(request.form.get("gaussTest"))
        formInfo["retCorr"] = bool(request.form.get("retCorrTest"))
        formInfo["retCorrThreshold"] = 0
        if formInfo["retCorr"]:
            formInfo["retCorrThreshold"] = float(request.form.get("retCorrThreshold"))
        formInfo["massBal"] = bool(request.form.get("massBalTest"))
        formInfo["lossFunc"] = str(request.form.get("lossFuncTest"))
        formInfo["solver"] = str(request.form.get("solverTest"))
        formInfo["factor"] = int(request.form.get("factorTest"))
        if request.form.get("porosityTest") != "":
            formInfo["porosity"] = float(request.form.get("porosityTest"))
        if formInfo["solver"] == "Nonlin":
            if request.form.get("saturationTest") != "":
                formInfo["saturation"] = float(request.form.get("saturationTest"))
        formInfo["comp"] = str(request.form.get("componentTest"))
        if request.form.get("KStartTest") != "":
            formInfo[formInfo["comp"] + "KStart"] = float(request.form.get("KStartTest"))
        if request.form.get("KEndTest") != "":
            formInfo[formInfo["comp"] + "KEnd"] = float(request.form.get("KEndTest"))
        if request.form.get("KStepTest") != "":
            formInfo[formInfo["comp"] + "KStep"] = float(request.form.get("KStepTest"))
        if request.form.get("DStartTest") != "":
            formInfo[formInfo["comp"] + "DStart"] = float(request.form.get("DStartTest"))
        if request.form.get("DEndTest") != "":
            formInfo[formInfo["comp"] + "DEnd"] = float(request.form.get("DEndTest"))
        if request.form.get("DStepTest") != "":
            formInfo[formInfo["comp"] + "DStep"] = float(request.form.get("DStepTest"))
        return redirect(url_for('get_projects_params'))

    @api.route('/projects/params', methods=['GET'])
    @flask_login.login_required
    def get_projects_params():
        nonlocal experimentSet, compList, formInfos, clusterComp
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        if not "spacialDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
        if not "timeDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
        if not "time" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
        formInfo = formInfos[flask_login.current_user.id]
        if not flask_login.current_user.id in experimentSet:
            fetchExperimentData()
        if not flask_login.current_user.id in clusterComp:
            clusterComp[flask_login.current_user.id] = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
        experimentClusterComp = clusterComp[flask_login.current_user.id]
        compList[flask_login.current_user.id] = experimentClusterComp.clusters.keys()
        return render_template('ParamsForm.html', compList = compList[flask_login.current_user.id], formInfo = formInfo, user = flask_login.current_user.id)

    @api.route('/projects/params', methods=['POST'])
    @flask_login.login_required
    def post_projects_params():
        nonlocal experimentSet, compList, formInfos, threadCounter
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        if not "spacialDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
        if not "timeDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
        if not "time" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["expName"] = str(request.form.get("expName"))
        formInfo["gauss"] = bool(request.form.get("gauss"))
        formInfo["retCorr"] = bool(request.form.get("retCorr"))
        formInfo["retCorrThreshold"] = 0
        if formInfo["retCorr"]:
            formInfo["retCorrThreshold"] = float(request.form.get("retCorrThreshold"))
        formInfo["massBal"] = bool(request.form.get("massBal"))
        formInfo["lossFunc"] = str(request.form.get("lossFunc"))
        formInfo["solver"] = str(request.form.get("solver"))
        formInfo['optimType'] = str(request.form.get("optimType"))
        formInfo["factor"] = int(request.form.get("factor"))
        formInfo["fixporosity"] = bool(request.form.get("fixporosity"))
        if not formInfo['optimType'] == "singlelevel":
            if not formInfo["fixporosity"]:
                formInfo["porosityStart"] = float(request.form.get("porosityStart"))
                formInfo["porosityEnd"] = float(request.form.get("porosityEnd"))
            formInfo["porosity"] = float(request.form.get("porosityInit"))
        if formInfo["optimType"] == "calcDisper":
            formInfo["AStart"] = float(request.form.get("AStart"))
            formInfo["AEnd"] = float(request.form.get("AEnd"))
            formInfo["A"] = float(request.form.get("AInit"))
        for comp in compList[flask_login.current_user.id]:
            if formInfo['optimType'] == "singlelevel":
                if not formInfo["fixporosity"]:
                    formInfo[comp + "PStart"] = float(request.form.get(comp + "PStart"))
                    formInfo[comp + "PEnd"] = float(request.form.get(comp + "PEnd"))
                formInfo[comp + "P"] = float(request.form.get(comp + "PInit"))
            formInfo[comp + "KStart"] = float(request.form.get(comp + "KStart"))
            formInfo[comp + "KEnd"] = float(request.form.get(comp + "KEnd"))
            formInfo[comp + "K"] = float(request.form.get(comp + "KInit"))
            if formInfo["optimType"] != "calcDisper" and formInfo["optimType"] != "calcDisper2":
                formInfo[comp + "DStart"] = float(request.form.get(comp + "DStart"))
                formInfo[comp + "DEnd"] = float(request.form.get(comp + "DEnd"))
                formInfo[comp + "D"] = float(request.form.get(comp + "DInit"))
            else:
                formInfo[comp + "B"] = float(request.form.get(comp + "B"))
                if formInfo["optimType"] == "calcDisper2":
                    formInfo[comp + "AStart"] = float(request.form.get(comp + "AStart"))
                    formInfo[comp + "AEnd"] = float(request.form.get(comp + "AEnd"))
                    formInfo[comp + "A"] = float(request.form.get(comp + "AInit"))
            if formInfo["solver"] == "Nonlin":
                formInfo[comp + "QStart"] = float(request.form.get(comp + "QStart"))
                formInfo[comp + "QEnd"] = float(request.form.get(comp + "QEnd"))
                formInfo[comp + "Q"] = float(request.form.get(comp + "QInit"))
        formInfo["lvl1optimsettings"] = {}
        formInfo["lvl2optimsettings"] = {}
        formInfo["lvl1optimsettings"]["algorithm"] = request.form.get("lvl1alg")
        formInfo["lvl2optimsettings"]["algorithm"] = request.form.get("lvl2alg")
        formInfo["lvl1optimsettings"]["settings"] ={}
        formInfo["lvl2optimsettings"]["settings"] ={}
        if formInfo["lvl1optimsettings"]["algorithm"] == "1":
            formInfo["lvl1optimsettings"]["settings"]["Ns"] = request.form.get("lvl1bruteforceNs")
        elif formInfo["lvl1optimsettings"]["algorithm"] == "2":
            formInfo["lvl1optimsettings"]["settings"]["maxiter"] = request.form.get("lvl1neldermeadmaxiter")
            formInfo["lvl1optimsettings"]["settings"]["maxfev"] = request.form.get("lvl1neldermeadmaxfev")
            formInfo["lvl1optimsettings"]["settings"]["xatol"] = request.form.get("lvl1neldermeadxatol")
            formInfo["lvl1optimsettings"]["settings"]["fatol"] = request.form.get("lvl1neldermeadfatol")
            formInfo["lvl1optimsettings"]["settings"]["aptive"] = request.form.get("lvl1neldermeadadaptive")
        elif formInfo["lvl1optimsettings"]["algorithm"] == "3":
            formInfo["lvl1optimsettings"]["settings"]["n"] = request.form.get("lvl1shgon")
            formInfo["lvl1optimsettings"]["settings"]["iters"] = request.form.get("lvl1shgoiters")
            formInfo["lvl1optimsettings"]["settings"]["maxev"] = request.form.get("lvl1shgomaxev")
            formInfo["lvl1optimsettings"]["settings"]["maxiter"] = request.form.get("lvl1shgomaxiter")
            formInfo["lvl1optimsettings"]["settings"]["maxfev"] = request.form.get("lvl1shgomaxfev")
            formInfo["lvl1optimsettings"]["settings"]["maxtime"] = request.form.get("lvl1shgomaxtime")
            formInfo["lvl1optimsettings"]["settings"]["f_tol"] = request.form.get("lvl1shgof_tol")
            formInfo["lvl1optimsettings"]["settings"]["f_min"] = request.form.get("lvl1shgof_min")
        elif formInfo["lvl1optimsettings"]["algorithm"] == "4":
            formInfo["lvl1optimsettings"]["settings"]["maxiter"] = request.form.get("lvl1powellmaxiter")
        if formInfo["lvl2optimsettings"]["algorithm"] == "1":
            formInfo["lvl2optimsettings"]["settings"]["Ns"] = request.form.get("lvl2bruteforceNs")
        elif formInfo["lvl2optimsettings"]["algorithm"] == "2":
            formInfo["lvl2optimsettings"]["settings"]["maxiter"] = request.form.get("lvl2neldermeadmaxiter")
            formInfo["lvl2optimsettings"]["settings"]["maxfev"] = request.form.get("lvl2neldermeadmaxfev")
            formInfo["lvl2optimsettings"]["settings"]["xatol"] = request.form.get("lvl2neldermeadxatol")
            formInfo["lvl2optimsettings"]["settings"]["fatol"] = request.form.get("lvl2neldermeadfatol")
            formInfo["lvl2optimsettings"]["settings"]["aptive"] = request.form.get("lvl2neldermeadadaptive")
        elif formInfo["lvl2optimsettings"]["algorithm"] == "3":
            formInfo["lvl2optimsettings"]["settings"]["n"] = request.form.get("lvl2shgon")
            formInfo["lvl2optimsettings"]["settings"]["iters"] = request.form.get("lvl2shgoiters")
            formInfo["lvl2optimsettings"]["settings"]["maxev"] = request.form.get("lvl2shgomaxev")
            formInfo["lvl2optimsettings"]["settings"]["maxiter"] = request.form.get("lvl2shgomaxiter")
            formInfo["lvl2optimsettings"]["settings"]["maxfev"] = request.form.get("lvl2shgomaxfev")
            formInfo["lvl2optimsettings"]["settings"]["maxtime"] = request.form.get("lvl2shgomaxtime")
            formInfo["lvl2optimsettings"]["settings"]["f_tol"] = request.form.get("lvl2shgof_tol")
            formInfo["lvl2optimsettings"]["settings"]["f_min"] = request.form.get("lvl2shgof_min")
        elif formInfo["lvl2optimsettings"]["algorithm"] == "4":
            formInfo["lvl2optimsettings"]["settings"]["maxiter"] = request.form.get("lvl2powellmaxiter")
        thread_id = threadCounter
        threadCounter += 1
        if not thread_id in exporting_threads:
            print("ID: " + str(thread_id))
            exporting_threads[thread_id] = MainWorkThread(flask_login.current_user.id, thread_id, flask_login.current_user.db)
            exporting_threads[thread_id].name = formInfo["expName"]
            exporting_threads[thread_id].start()
            timers[thread_id] = time.time()
        return redirect(url_for('get_projects_params_result', id=thread_id))

    @api.route('/projects/params/result/<id>', methods=['GET'])
    @flask_login.login_required
    def get_projects_params_result(id):
        nonlocal experimentSet, compList, experimentSet, clusterComp
        if not flask_login.current_user.id in experimentSet:
            fetchExperimentData()
        if not flask_login.current_user.id in clusterComp:
            clusterComp[flask_login.current_user.id] = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
        id = int(id)
        if id in exporting_threads:
            return render_template('ResultPage.html', result = exporting_threads[id].result, user = flask_login.current_user.id, name = exporting_threads[id].name, id = id)
        else:
            dbuser = flask_login.current_user.db
            for result in dbuser.results:
                if result.thr_id == id:
                    expCompDict = {}
                    for exp in result.experiments:
                        expCompDict[exp] = []
                    for key, val in result.results["lossfunctionprogress"].items():
                        for exp in val.keys():
                            expCompDict[exp].append(key)
                    id = int(id)
                    compExperimentDict = {}
                    for key, val in result.results["lossfunctionprogress"].items():
                        compExperimentDict[key] = val.keys()
                    return render_template('ResultPage.html', compList=result.results["bestLvl2Params"].keys(),
                                           compExpDict=compExperimentDict, expDict=expCompDict, result=result.results,
                                           timer=result.time, user=flask_login.current_user.id, name=result.name, id=id)
        return "Result not found"

    @api.route('/projects/params/result/<id>/prograph', methods=['POST'])
    @api.route('/projects/result/<id>/prograph', methods=['POST'])
    @flask_login.login_required
    def post_projects_result_prograph(id):
        # TODO maybe thread this
        nonlocal exporting_threads, plotFileCounter, BASE_FOLDER
        id = int(id)
        if id in exporting_threads:
            result = exporting_threads[id].result
        else:
            dbuser = flask_login.current_user.db
            for res in dbuser.results:
                if res.thr_id == id:
                    result = res.results
        comp = str(request.form.get("componentTest"))
        if comp != "all":
            exp = str(request.form.get("expList" + comp))
        if comp != "all":
            if exp != "all":
                relevantList = result["lossfunctionprogress"][comp][exp]
            else:
                firstKey = list(result["lossfunctionprogress"][comp].keys())[0]
                relevantList = [0] * len(result["lossfunctionprogress"][comp][firstKey])
                for key in result["lossfunctionprogress"][comp]:
                    relevantList = list( map(add, relevantList, result["lossfunctionprogress"][comp][key]) )
        else:
            firstKey = list(result["lossfunctionprogress"].keys())[0]
            firstFirstKey = list(result["lossfunctionprogress"][firstKey].keys())[0]
            relevantList = [0] * len(result["lossfunctionprogress"][firstKey][firstFirstKey])
            for compkey in result["lossfunctionprogress"]:
                for expkey in result["lossfunctionprogress"][compkey]:
                    relevantList = list( map(add, relevantList, result["lossfunctionprogress"][compkey][expkey]) )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Loss function progression")
        ax.set_xlabel("Number of lv1 optimizations")
        ax.set_ylabel("Loss function value")
        ax.plot(relevantList)
        filename = "plot" + str(plotFileCounter) + ".png"
        plotFileCounter += 1
        plt.savefig(BASE_FOLDER + '/functions/WebServerStuff/static/images/' + filename)
        plt.clf()
        return render_template('Picture.html', pictureURL = url_for('static', filename='images/'+filename), alt="chart", width="640", height="480")

    @api.route('/projects/params/result/<id>/rescomp', methods=['POST'])
    @api.route('/projects/result/<id>/rescomp', methods=['POST'])
    @flask_login.login_required
    def post_projects_result_rescomp(id):
        nonlocal exporting_threads, plotFileCounter, experimentSet, threadCounter, clusterComp
        if not flask_login.current_user.id in experimentSet:
            fetchExperimentData()
        id = int(id)
        dbuser = flask_login.current_user.db
        for res in dbuser.results:
            if res.thr_id == id:
                result = res.results
                expList = res.experiments
        expIdx = int(request.form.get("expList2"))
        compForm = request.form.get("componentTest2" + str(expIdx))
        if compForm != "all":
            compIdx = int(compForm)
            comp = list(result["lossfunctionprogress"].keys())[compIdx]
            params = []
            for param in result["bestLvl1Params"]:
                params.append(param)
            for param in result["bestLvl2Params"][comp]:
                params.append(param)
        else:
            compIdx = compForm
            params = []
            for comp in result["lossfunctionprogress"].keys():
                tmp = []
                for param in result["bestLvl1Params"]:
                    tmp.append(param)
                for param in result["bestLvl2Params"][comp]:
                    tmp.append(param)
                params.append(tmp)
            comp = compIdx
        gauss = bool(request.form.get("gauss"))
        retCorr = bool(request.form.get("retCorr"))
        retCorrThreshold = 0
        if retCorr:
            retCorrThreshold = float(request.form.get("retCorrThreshold"))
        massBal = bool(request.form.get("massBal"))
        currExpSet = operator.Preprocess(experimentSet[flask_login.current_user.id], gauss, retCorr, massBal, retCorrThreshold)
        exp = expList[expIdx]
        expObj = currExpSet.get_exp_by_name(exp)
        if expObj:
            thread_id = threadCounter
            threadCounter += 1
            if not thread_id in exporting_threads:
                print("ID: " + str(thread_id))
                exporting_threads[thread_id] = SolverThread(flask_login.current_user.id, result["optimparams"]["solver"], params, expObj, compIdx, comp)
                exporting_threads[thread_id].start()
            return url_for('post_projects_result_rescomp_progress', id=thread_id)
        return "Error: requested experiment was removed"

    @api.route('/projects/params/result/<id>/rescomp/progress', methods=['GET'])
    @api.route('/projects/result/<id>/rescomp/progress', methods=['GET'])
    @flask_login.login_required
    def post_projects_result_rescomp_progress(id):
        nonlocal exporting_threads, plotFileCounter, formInfos, BASE_FOLDER
        formInfo = formInfos[flask_login.current_user.id]
        thread_id = int(id)
        if type(exporting_threads[thread_id].result) is str and exporting_threads[thread_id].result == "-":
            return ""
        else:
            results = exporting_threads[thread_id].result
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if exporting_threads[thread_id].comp == "all":
                for idx, result in enumerate(results):
                    compName = exporting_threads[thread_id].exp.experimentComponents[idx].name
                    df = exporting_threads[thread_id].exp.experimentComponents[idx].concentrationTime
                    modelCurve = result[0][:, -1]
                    time = result[1]
                    ax.set_title("Result comparison")
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Concentration [mg/mL]")
                    ax.plot(time, modelCurve, label=compName+" model")
                    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], marker=',', s=10, label=compName+" experiment")
            else:
                compName = exporting_threads[thread_id].comp
                df = exporting_threads[thread_id].exp.experimentComponents[int(exporting_threads[thread_id].compIdx)].concentrationTime
                modelCurve = results[0][0][:, -1]
                time = results[0][1]
                ax.set_title("Result comparison")
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Concentration [mg/mL]")
                ax.plot(time, modelCurve, label=compName + " model")
                ax.scatter(df.iloc[:, 0], df.iloc[:, 1], marker=',', s=10, label=compName + " experiment")
            plt.legend()
            filename = "plot" + str(plotFileCounter) + ".png"
            plotFileCounter += 1
            plt.savefig(BASE_FOLDER + '/functions/WebServerStuff/static/images/' + filename)
            plt.clf()
            return render_template('Picture.html', pictureURL=url_for('static', filename="images/" + filename), picId="graphImg2", alt="chart", width="640", height="480", downloadUrl=url_for("post_projects_result_rescomp_matrix", id=thread_id))

    @api.route('/projects/params/result/<id>/rescomp/matrix', methods=['GET'])
    @api.route('/projects/result/<id>/rescomp/matrix', methods=['GET'])
    @flask_login.login_required
    def post_projects_result_rescomp_matrix(id):
        nonlocal exporting_threads, plotFileCounter
        thread_id = int(id)
        if type(exporting_threads[thread_id].result) is str:
            return str(exporting_threads[thread_id].result)
        else:
            zipname = 'tables'+ str(plotFileCounter) +'.zip'
            zipf = zipfile.ZipFile('functions/WebServerStuff/static/tables/' + zipname,'w', zipfile.ZIP_DEFLATED)
            plotFileCounter += 1
            for result in exporting_threads[thread_id].result:
                df = pd.DataFrame(result)
                filename = "table" + str(plotFileCounter) + ".csv"
                plotFileCounter += 1
                df.to_csv('functions/WebServerStuff/static/tables/' + filename, index=False)
                zipf.write('functions/WebServerStuff/static/tables/' + filename, filename)
            zipf.close()
            return send_file("static/tables/" + zipname)

    @api.route('/projects/params/result/<id>/progress', methods=['GET'])
    @flask_login.login_required
    def get_projects_params_result_progress(id):
        nonlocal compList
        id = int(id)
        timer = time.time() - timers[id]
        if exporting_threads[id].result == "-":
            return "Time elapsed: " + str(datetime.timedelta(seconds=timer))
        else:
            exporting_threads.pop(id)
            timers.pop(id)
            return "DONE"

    @api.route('/projects/result', methods=['GET'])
    @flask_login.login_required
    def get_projects_result_list():
        nonlocal numberOfRunningOptims
        dbuser = flask_login.current_user.db
        resList = []
        for res in dbuser.results:
            resList.append(res)
        return render_template('ResultList.html', resList=resList, user=flask_login.current_user.id, numOfTasks=numberOfRunningOptims)


    @api.route('/projects/result/<id>/params', methods=['GET'])
    @flask_login.login_required
    def get_projects_result_params(id):
        nonlocal formInfos
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        formInfo = formInfos[flask_login.current_user.id]
        dbuser = flask_login.current_user.db
        optimparams = {}
        for res in dbuser.results:
            if res.thr_id == int(id):
                formInfo["expName"] = res.name
                optimparams = res.results["optimparams"]
                break
        if not optimparams:
            return "Unknown result"
        formInfo["gauss"] = optimparams["gauss"]
        formInfo["retCorr"] = optimparams["retCorr"]
        formInfo["retCorrThreshold"] = optimparams["retThreshold"]
        formInfo["massBal"] = optimparams["massBal"]
        formInfo['lossFunc'] = optimparams["lossFunction"]
        formInfo['optimType'] = optimparams["optimType"]
        formInfo['solver'] = optimparams["solver"]
        formInfo['factor'] = optimparams["factor"]
        formInfo['fixporosity'] = optimparams["fixporosity"]
        if formInfo['optimType'] != "singlelevel":
            if not formInfo["fixporosity"]:
                formInfo["porosityStart"] = optimparams["Lvl1ParamDict"]["prange"][0]
                formInfo["porosityEnd"] = optimparams["Lvl1ParamDict"]["prange"][1]
            formInfo['porosity'] = optimparams["Lvl1ParamDict"]["pinit"]
        if formInfo["optimType"] == "calcDisper":
            formInfo["AStart"] = optimparams["Lvl1ParamDict"]["arange"][0]
            formInfo["AEnd"] = optimparams["Lvl1ParamDict"]["arange"][1]
            formInfo['A'] = optimparams["Lvl1ParamDict"]["ainit"]
        cmpList = []
        for comp in optimparams["Lvl2ParamDict"]:
            cmpList.append(comp)
            formInfo[comp + "KStart"] = optimparams["Lvl2ParamDict"][comp]["krange"][0]
            formInfo[comp + "KEnd"] = optimparams["Lvl2ParamDict"][comp]["krange"][1]
            formInfo[comp + "K"] = optimparams["Lvl2ParamDict"][comp]["kinit"]
            if formInfo["optimType"] != "calcDisper" and formInfo["optimType"] != "calcDisper2":
                formInfo[comp + "DStart"] = optimparams["Lvl2ParamDict"][comp]["drange"][0]
                formInfo[comp + "DEnd"] = optimparams["Lvl2ParamDict"][comp]["drange"][1]
                formInfo[comp + "D"] = optimparams["Lvl2ParamDict"][comp]["dinit"]
            else:
                formInfo[comp + "B"] = optimparams["Lvl2ParamDict"][comp]["b"]
                if formInfo["optimType"] == "calcDisper2":
                    formInfo[comp + "AStart"] = optimparams["Lvl2ParamDict"][comp]["arange"][0]
                    formInfo[comp + "AEnd"] = optimparams["Lvl2ParamDict"][comp]["arange"][1]
                    formInfo[comp + "A"] = optimparams["Lvl2ParamDict"][comp]["ainit"]
            if formInfo["solver"] == "Nonlin":
                formInfo[comp + "QStart"] = optimparams["Lvl2ParamDict"][comp]["qrange"][0]
                formInfo[comp + "QEnd"] = optimparams["Lvl2ParamDict"][comp]["qrange"][1]
                formInfo[comp + "Q"] = optimparams["Lvl2ParamDict"][comp]["qinit"]
        formInfo["lvl1optimsettings"] = optimparams["lvl1optim"]
        formInfo["lvl2optimsettings"] = optimparams["lvl2optim"]
        return redirect(url_for("get_projects_params"))

    @api.route('/projects/result/<id>', methods=['GET'])
    @flask_login.login_required
    def get_projects_result_list_show(id):
        nonlocal compList, experimentSet, clusterComp
        if not flask_login.current_user.id in experimentSet:
            fetchExperimentData()
        if not flask_login.current_user.id in clusterComp:
            clusterComp[flask_login.current_user.id] = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
        dbuser = flask_login.current_user.db
        id = int(id)
        for result in dbuser.results:
            print(result.thr_id)
            if result.thr_id == id:
                expCompDict = {}
                for exp in result.experiments:
                    expCompDict[exp] = []
                for key, val in result.results["lossfunctionprogress"].items():
                    for exp in val.keys():
                        expCompDict[exp].append(key)
                compExperimentDict = {}
                for key, val in result.results["lossfunctionprogress"].items():
                    compExperimentDict[key] = val.keys()
                return render_template('ResultPage.html', compList=result.results["bestLvl2Params"].keys(),
                                           compExpDict=compExperimentDict, expDict=expCompDict, result=result.results,
                                           timer=result.time, user=flask_login.current_user.id, name=result.name, id=id)
        return "Unknown result"

    @api.route('/projects/result/<id>', methods=['DELETE'])
    @flask_login.login_required
    def get_projects_result_list_delete(id):
        nonlocal compList, db_mutex
        id = int(id)
        dbuser = flask_login.current_user.db
        for res in dbuser.results:
            if res.thr_id == id:
                with db_mutex:
                    dbuser.results.remove(res)
                    dbuser.save()
                res.delete()
                return redirect(url_for('get_projects_result_list'))
        return "Unknown result"

    @api.route('/projects/result/<id>/copy', methods=['GET'])
    @flask_login.login_required
    def get_projects_result_list_copy(id):
        nonlocal compList
        id = int(id)
        dbuser = flask_login.current_user.db
        for res in dbuser.results:
            if res.thr_id == id:
                return res.to_json()
        return "Unknown result"

    @api.route('/projects/solver', methods=['GET'])
    @flask_login.login_required
    def get_projects_solver():
        nonlocal formInfos
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        if not "spacialDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
        if not "timeDiff" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
        if not "time" in formInfos[flask_login.current_user.id]:
            formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
        formInfo = formInfos[flask_login.current_user.id]
        return render_template('SolverSettings.html', formInfo=formInfo, user = flask_login.current_user.id)

    @api.route('/projects/solver', methods=['POST'])
    @flask_login.login_required
    def post_projects_solver():
        nonlocal formInfos
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["time"] = float(request.form.get("time"))
        formInfo["spacialDiff"] = int(request.form.get("spacialDiff"))
        formInfo["timeDiff"] = int(request.form.get("timeDiff"))
        return redirect(url_for("upload_file_page"))

    @api.route('/projects', methods=['POST'])
    @flask_login.login_required
    def upload_file():
        nonlocal uploadedFiles, db_mutex
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('upload_file_page'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(api.config['UPLOAD_FOLDER']  + '/' + flask_login.current_user.id, filename))
            jsonString = Serialize_File_To_JSON(BASE_FOLDER + "/data/TestUploadFolder/" + flask_login.current_user.id + "/" + filename)
            newExperiment = DBExperiment(uniquename=flask_login.current_user.id + "/" + filename, name=filename, experiment=jsonString)
            try:
                with db_mutex:
                    newExperiment.save()
                    flask_login.current_user.db.experiments.append(newExperiment)
                    flask_login.current_user.db.save()
                uploadedFiles[flask_login.current_user.id] = [exp.name for exp in flask_login.current_user.db.experiments]
                #uploadedFiles[flask_login.current_user.id] = next(walk(UPLOAD_FOLDER + "\\" + flask_login.current_user.id), (None, None, []))[2]
                fetchExperimentData()
                clusterComp[flask_login.current_user.id] = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
                compList[flask_login.current_user.id] = clusterComp[flask_login.current_user.id].clusters.keys()
                return redirect(url_for('upload_file_page'))
            except me.errors.NotUniqueError:
                return render_template('Upload.html', uploadedFilesLen = len(uploadedFiles[flask_login.current_user.id]), uploadedFiles = uploadedFiles[flask_login.current_user.id], user = flask_login.current_user.id, error="File with that name already exists.")

    @api.route('/projects/<file>', methods=['DELETE'])
    @flask_login.login_required
    def delete_file(file):
        nonlocal uploadedFiles, db_mutex
        dbuser = flask_login.current_user.db
        for exp in dbuser.experiments:
            if exp.uniquename == flask_login.current_user.id + "/" + file:
                with db_mutex:
                    dbuser.experiments.remove(exp)
                    dbuser.save()
                exp.delete()
                os.remove(UPLOAD_FOLDER + "/" + flask_login.current_user.id + "/" + file)
                uploadedFiles[flask_login.current_user.id] = [exp.name for exp in dbuser.experiments]
                print(uploadedFiles[flask_login.current_user.id])
                #uploadedFiles[flask_login.current_user.id] = next(walk(UPLOAD_FOLDER + "\\" + flask_login.current_user.id), (None, None, []))[2]
                fetchExperimentData()
                #experimentSet[flask_login.current_user.id] = operator.Load_Experiment_Set(UPLOAD_FOLDER + "\\" + flask_login.current_user.id)
                clusterComp[flask_login.current_user.id] = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
                compList[flask_login.current_user.id] = clusterComp[flask_login.current_user.id].clusters.keys()
                return redirect(url_for('upload_file_page'))
        return "Unknown result"

    @api.route('/projects/<file>', methods=['GET'])
    @flask_login.login_required
    def get_experiment_string(file):
        nonlocal uploadedFiles
        dbuser = flask_login.current_user.db
        for exp in dbuser.experiments:
            if exp.uniquename == flask_login.current_user.id + "/" + file:
                return exp.experiment
        return "Unknown experiment"


    @api.route('/projects', methods=['GET'])
    @flask_login.login_required
    def upload_file_page():
        nonlocal uploadedFiles
        if not os.path.exists(UPLOAD_FOLDER + "/" + flask_login.current_user.id):
            os.mkdir(UPLOAD_FOLDER + "/" + flask_login.current_user.id)
        uploadedFiles[flask_login.current_user.id] = [exp.name for exp in flask_login.current_user.db.experiments]
        #uploadedFiles[flask_login.current_user.id] = next(walk(UPLOAD_FOLDER + '\\' + flask_login.current_user.id), (None, None, []))[2]
        return render_template('Upload.html', uploadedFilesLen = len(uploadedFiles[flask_login.current_user.id]), uploadedFiles = uploadedFiles[flask_login.current_user.id], user = flask_login.current_user.id)

    @api.route('/logout')
    def logout():
        flask_login.logout_user()
        return redirect(url_for('get_main_page'))

    @api.route('/register', methods=['GET', 'POST'])
    def get_registration():
        nonlocal db_mutex
        if request.method == 'GET':
            return render_template('Registration.html')
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        for user in DBUser.objects:
            if username == user.username:
                return render_template('Registration.html', badRegistration="User already exists.")
        newUser = DBUser(username=username, password_hash=password)
        with db_mutex:
            newUser.save()
        return redirect(url_for('get_main_page'))

    @api.route('/', methods=['GET', 'POST'])
    def get_main_page():
        nonlocal formInfos
        if request.method == 'GET':
            if flask_login.current_user.is_authenticated:
                if not flask_login.current_user.id in formInfos:
                    formInfos[flask_login.current_user.id] = {}
                if not "spacialDiff" in formInfos[flask_login.current_user.id]:
                    formInfos[flask_login.current_user.id]["spacialDiff"] = SOLVER_SPACIAL_DIFF
                if not "timeDiff" in formInfos[flask_login.current_user.id]:
                    formInfos[flask_login.current_user.id]["timeDiff"] = SOLVER_TIME_DIFF
                if not "time" in formInfos[flask_login.current_user.id]:
                    formInfos[flask_login.current_user.id]["time"] = SOLVER_TIME
                return render_template('Index.html', user = flask_login.current_user.id)
            return render_template('Index.html')
        username = request.form['username']
        for dbuser in DBUser.objects:
            if username in dbuser.username and check_password_hash(dbuser.password_hash, request.form['password']):
                user = User()
                user.id = username
                flask_login.login_user(user)
                if not user.id in formInfos:
                    formInfos[user.id] = {}
                if not os.path.exists(UPLOAD_FOLDER + "/" + user.id):
                    os.mkdir(UPLOAD_FOLDER + "/" + user.id)
                return redirect(url_for('upload_file_page'))

        return render_template('Index.html', badLogin=True)

    '''@api.route('/api/experiment', methods=['POST'])
    def api_post_experiment():
        nonlocal threadCounter
        try:
            data = request.json
            thread_id = threadCounter
            threadCounter += 1
            if not thread_id in exporting_threads:
                print("ID: " + str(thread_id))
                exporting_threads[thread_id] = ApiWorkThread(data, thread_id)
                exporting_threads[thread_id].start()
                timers[thread_id] = time.time()
            return (url_for('api_get_result', id=thread_id), 201)
        except Exception as e:
            print(e)
            return (e, 400)

    @api.route('/api/result/<id>', methods=['GET'])
    def api_get_result(id):
        nonlocal threadCounter, db_mutex
        id = int(id)
        timer = time.time() - timers[id]
        if exporting_threads[id].result == "-":
            return "Time elapsed: " + str(datetime.timedelta(seconds=timer))
        else:
            thread = exporting_threads[id]
            timers.pop(id)
            newResult = DBResult(results = thread.result, thr_id=id, name=thread.data['settings']['expName'])
            with db_mutex:
                newResult.save()
            return newResult.to_json()'''

    #api.run(debug=True)
    print("localhost:6969")
    serve(api, listen='*:6969', threads=4)

