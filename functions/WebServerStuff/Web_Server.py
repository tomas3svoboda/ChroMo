import os
import threading
import time
import flask_login
import pandas as pd
from flask_security import Security, MongoEngineUserDatastore, \
    UserMixin, RoleMixin, login_required
from flask_security.utils import hash_password
from flask_mongoengine import MongoEngine
from flask import Flask, flash, request, redirect, url_for, send_file, render_template, json
import mongoengine as me
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
import matplotlib.pyplot as plt
from functions.WebServerStuff.Serialize_File_To_JSON import Serialize_File_To_JSON
from functions.Loss_Function_Analysis_Simple import Loss_Function_Analysis_Simple
from functions.Model_Analysis import Model_Analysis
from objects.Operator import Operator
from objects.ExperimentSet import ExperimentSet
from os import walk

def Web_Server():
    matplotlib.use('Agg')

    plotFileCounter = 1
    threadCounter = 1
    experimentSet = {}
    compList = {}
    formInfos = {}
    operator = Operator()


    UPLOAD_FOLDER = 'C:\\Users\\Adam\\ChroMo\\docu\\TestUploadFolder'
    BASE_FOLDER = 'C:\\Users\\Adam\\ChroMo'

    #UPLOAD_FOLDER = 'C:\\Users\\Z004PTSU\\PycharmProjects\\ChroMo\\docu\\TestUploadFolder'
    #BASE_FOLDER = 'C:\\Users\\Z004PTSU\\PycharmProjects\\ChroMo'

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

    class DBExperiment(me.Document):
        name = me.StringField(required=True)
        experiment = me.StringField(required=True)

    class DBResult(me.Document):
        thr_id = me.IntField(required=True, unique=True)
        name = me.StringField(required=True)
        results = me.DictField(required=True)

    class DBUser(me.Document):
        username = me.StringField(required=True, unique=True)
        password_hash = me.StringField()
        results = me.ListField()
        experiments = me.ListField()


    class User(flask_login.UserMixin):
        pass

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
        nonlocal experimentSet, formInfos

        def __init__(self, user_id):
            self.user_id = user_id
            self.result = "-"
            super().__init__()

        def run(self):
            formInfo = formInfos[self.user_id]
            KDQDict = {}
            for comp in compList[self.user_id]:
                tmpDict = {}
                tmpDict["kinit"] = formInfo[comp + "KInit"]
                tmpDict["krange"] = [formInfo[comp + "KStart"], formInfo[comp + "KEnd"]]
                tmpDict["dinit"] = formInfo[comp + "DInit"]
                tmpDict["drange"] = [formInfo[comp + "DStart"], formInfo[comp + "DEnd"]]
                if formInfo["solver"] == "Nonlin":
                    tmpDict["qinit"] = float(request.form.get(comp + "QInit"))
                    tmpDict["qrange"] = [formInfo[comp + "QStart"], formInfo[comp + "QEnd"]]
                else:
                    tmpDict["qinit"] = 0
                    tmpDict["qrange"] = [0, 0]
                KDQDict[comp] = tmpDict
            self.result = operator.Web_Start(
                    experimentSet[self.user_id], UPLOAD_FOLDER + "\\" + self.user_id,
                    formInfo["gauss"], formInfo["retCorr"], formInfo["massBal"], formInfo["lossFunc"],
                    formInfo["solver"], formInfo["factor"], formInfo["porosityStart"], formInfo["porosityEnd"],
                    formInfo["porosityInit"], KDQDict)

    class SimpleThread(threading.Thread):
        nonlocal experimentSet, formInfos

        def __init__(self, user_id):
            self.user_id = user_id
            self.progress = "-"
            super().__init__()

        def run(self):
            nonlocal plotFileCounter
            formInfo = formInfos[self.user_id]
            experimentClusterComp2 = operator.Cluster_By_Component(experimentSet[self.user_id])
            params = [formInfo["porosity"], formInfo[formInfo["comp2"] + "K"], formInfo[formInfo["comp2"] + "D"],
                      formInfo["saturation"]]
            Model_Analysis(experimentClusterComp2.clusters[formInfo["comp2"]][formInfo["exp" + formInfo["comp2"]] - 1],
                           formInfo["solver"], params, webMode=True, title="Experimental data")
            filename = "plot" + str(plotFileCounter) + ".png"
            plotFileCounter += 1
            plt.savefig('functions/WebServerStuff/static/images/' + filename)
            plt.cla()
            currExperimentSet = operator.Preprocess(experimentSet[self.user_id], formInfo["gauss"], formInfo["retCorr"], formInfo["massBal"])
            experimentClusterComp = operator.Cluster_By_Component(currExperimentSet)
            Model_Analysis(experimentClusterComp.clusters[formInfo["comp2"]][formInfo["exp" + formInfo["comp2"]] - 1],
                           formInfo["solver"], params, webMode=True, title="Preprocessed data")
            filename2 = "plot" + str(plotFileCounter) + ".png"
            plotFileCounter += 1
            plt.savefig('functions/WebServerStuff/static/images/' + filename2)
            plt.cla()
            self.progress = [filename, filename2]

    class ExportingThread(threading.Thread):
        nonlocal experimentSet, formInfos

        def __init__(self, user_id):
            self.progress = "0%"
            self.user_id = user_id
            super().__init__()

        def run(self):
            formInfo = formInfos[self.user_id]
            currExperimentSet = operator.Preprocess(experimentSet[self.user_id], formInfo["gauss"], formInfo["retCorr"], formInfo["massBal"])
            experimentClusterComp = operator.Cluster_By_Component(currExperimentSet)
            self.generator = Loss_Function_Analysis_Simple(experimentClusterComp, formInfo["comp"], "", formInfo[formInfo["comp"] + "KStart"]
                                      , formInfo[formInfo["comp"] + "DStart"], formInfo[formInfo["comp"] + "KEnd"]
                                      , formInfo[formInfo["comp"] + "DEnd"], formInfo[formInfo["comp"] + "KStep"]
                                      , formInfo[formInfo["comp"] + "DStep"], formInfo["porosity"], formInfo["saturation"]
                                      , formInfo["lossFunc"], formInfo["factor"], formInfo["solver"], True)
            for res in self.generator:
                self.progress = res
                if not type(res) is str:
                    break

        def new_angle(self):
            self.progress = next(self.generator)


    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @api.route('/projects/test2', methods=['GET'])
    @flask_login.login_required
    def get_projects_test2():
        nonlocal experimentSet, compList, formInfos
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        formInfo = formInfos[flask_login.current_user.id]
        if not flask_login.current_user.id in experimentSet:
            experimentSet[flask_login.current_user.id] = operator.Load_Experiment_Set(UPLOAD_FOLDER + "\\" + flask_login.current_user.id)
        experimentClusterComp = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
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
        plt.cla()
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["gauss"] = bool(request.form.get("gaussTest"))
        formInfo["retCorr"] = bool(request.form.get("retCorrTest"))
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
        thread_id = threadCounter
        threadCounter += 1
        if not thread_id in exporting_threads:
            print("ID: " + str(thread_id))
            exporting_threads[thread_id] = SimpleThread(flask_login.current_user.id)
            plt.cla()
            exporting_threads[thread_id].start()
        return url_for("post_projects_test2_progress", id=thread_id)

    @api.route('/projects/test2/<id>/progress', methods=['GET'])
    @flask_login.login_required
    def post_projects_test2_progress(id):
        nonlocal exporting_threads, plotFileCounter
        thread_id = int(id)
        if exporting_threads[thread_id].progress == "-":
            return ""
        else:
            return render_template('Picture.html', pictureURL=url_for('static', filename='images/' + exporting_threads[thread_id].progress[0]),
                                   alt="chart", width="640", height="480") + \
                   render_template('Picture.html', pictureURL=url_for('static', filename='images/' + exporting_threads[thread_id].progress[1]),
                                   alt="chart", width="640", height="480")

    @api.route('/projects/test2/continue', methods=['POST'])
    @flask_login.login_required
    def post_projects_test2_continue():
        nonlocal formInfos
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["gauss"] = bool(request.form.get("gaussTest"))
        formInfo["retCorr"] = bool(request.form.get("retCorrTest"))
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
        nonlocal experimentSet, compList, formInfos
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        formInfo = formInfos[flask_login.current_user.id]
        if not flask_login.current_user.id in experimentSet:
            experimentSet[flask_login.current_user.id] = operator.Load_Experiment_Set(UPLOAD_FOLDER + "\\" + flask_login.current_user.id)
        experimentClusterComp = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
        compList[flask_login.current_user.id] = experimentClusterComp.clusters.keys()
        return render_template('ParamsTestForm.html', compList = compList[flask_login.current_user.id], formInfo=formInfo, user = flask_login.current_user.id)

    @api.route('/projects/test/<id>/newAngle', methods=['GET'])
    @flask_login.login_required
    def post_projects_test_new_angle(id):
        nonlocal exporting_threads, plotFileCounter
        thread_id = int(id)
        exporting_threads[thread_id].new_angle()
        progress = exporting_threads[thread_id].progress
        filename = "plot" + str(plotFileCounter) + ".png"
        plotFileCounter += 1
        plt.savefig('functions/WebServerStuff/static/images/' + filename)
        return  "<div>Minimum:<br>Henry constant: " + str(progress[0]) + "<br>Dispersion coefficient: " + str(progress[1]) + "</div>" + \
                render_template('Picture.html', pictureURL = url_for('static', filename='images/'+filename), alt="chart", width="640", height="480", newAngleUrl=url_for('post_projects_test_new_angle', id=id))

    @api.route('/projects/test/<id>/progress', methods=['GET'])
    @flask_login.login_required
    def post_projects_test_progress(id):
        nonlocal exporting_threads, plotFileCounter
        thread_id = int(id)
        if type(exporting_threads[thread_id].progress) is str:
            return str(exporting_threads[thread_id].progress)
        else:
            progress = exporting_threads[thread_id].progress
            filename = "plot" + str(plotFileCounter) + ".png"
            plotFileCounter += 1
            plt.savefig('functions/WebServerStuff/static/images/' + filename)
            return "<div>Minimum:<br>Henry constant: " + str(progress[0]) + "<br>Dispersion coefficient: " + str(progress[1]) + "</div>" + \
                   render_template('Picture.html', pictureURL = url_for('static', filename='images/'+filename), alt="chart", width="640", height="480", newAngleUrl=url_for('post_projects_test_new_angle', id=id), downloadUrl = url_for('get_projects_test_matrix', id=id))

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
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["gauss"] = bool(request.form.get("gaussTest"))
        formInfo["retCorr"] = bool(request.form.get("retCorrTest"))
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
            exporting_threads[thread_id] = ExportingThread(flask_login.current_user.id)
            plt.cla()
            exporting_threads[thread_id].start()
        return url_for("post_projects_test_progress", id=thread_id)

    @api.route('/projects/test/continue', methods=['POST'])
    @flask_login.login_required
    def post_projects_test_continue():
        nonlocal formInfos
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["gauss"] = bool(request.form.get("gaussTest"))
        formInfo["retCorr"] = bool(request.form.get("retCorrTest"))
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
        nonlocal experimentSet, compList, formInfos
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        formInfo = formInfos[flask_login.current_user.id]
        if not flask_login.current_user.id in experimentSet:
            experimentSet[flask_login.current_user.id] = operator.Load_Experiment_Set(UPLOAD_FOLDER + "\\" + flask_login.current_user.id)
        experimentClusterComp = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
        compList[flask_login.current_user.id] = experimentClusterComp.clusters.keys()
        return render_template('ParamsForm.html', compList = compList[flask_login.current_user.id], formInfo = formInfo, user = flask_login.current_user.id)

    @api.route('/projects/params', methods=['POST'])
    @flask_login.login_required
    def post_projects_params():
        nonlocal experimentSet, compList, formInfos, threadCounter
        if not flask_login.current_user.id in formInfos:
            formInfos[flask_login.current_user.id] = {}
        formInfo = formInfos[flask_login.current_user.id]
        formInfo["expName"] = str(request.form.get("expName"))
        formInfo["gauss"] = bool(request.form.get("gauss"))
        formInfo["retCorr"] = bool(request.form.get("retCorr"))
        formInfo["massBal"] = bool(request.form.get("massBal"))
        formInfo["lossFunc"] = str(request.form.get("lossFunc"))
        formInfo["solver"] = str(request.form.get("solver"))
        formInfo["factor"] = int(request.form.get("factor"))
        formInfo["porosityStart"] = float(request.form.get("porosityStart"))
        formInfo["porosityEnd"] = float(request.form.get("porosityEnd"))
        formInfo["porosityInit"] = float(request.form.get("porosityInit"))
        for comp in compList[flask_login.current_user.id]:
            formInfo[comp + "KStart"] = float(request.form.get(comp + "KStart"))
            formInfo[comp + "KEnd"] = float(request.form.get(comp + "KEnd"))
            formInfo[comp + "KInit"] = float(request.form.get(comp + "KInit"))
            formInfo[comp + "DStart"] = float(request.form.get(comp + "DStart"))
            formInfo[comp + "DEnd"] = float(request.form.get(comp + "DEnd"))
            formInfo[comp + "DInit"] = float(request.form.get(comp + "DInit"))
            if formInfo["solver"] == "Nonlin":
                formInfo[comp + "QStart"] = float(request.form.get(comp + "QStart"))
                formInfo[comp + "QEnd"] = float(request.form.get(comp + "QEnd"))
                formInfo[comp + "QInit"] = float(request.form.get(comp + "QInit"))
        thread_id = threadCounter
        threadCounter += 1
        if not thread_id in exporting_threads:
            print("ID: " + str(thread_id))
            exporting_threads[thread_id] = MainWorkThread(flask_login.current_user.id)
            exporting_threads[thread_id].name = formInfo["expName"]
            exporting_threads[thread_id].start()
        return redirect(url_for('get_projects_params_result', id=thread_id))

    @api.route('/projects/params/result/<id>', methods=['GET'])
    @flask_login.login_required
    def get_projects_params_result(id):
        nonlocal experimentSet, compList
        id = int(id)
        if id in exporting_threads:
            return render_template('ResultPage.html', compList = compList[flask_login.current_user.id], result = exporting_threads[id].result, user = flask_login.current_user.id, name = exporting_threads[id].name)
        else:
            dbuser = flask_login.current_user.db
            for result in dbuser.results:
                if result.thr_id == id:
                    return render_template('ResultPage.html', compList = compList[flask_login.current_user.id], result = result.results, user = flask_login.current_user.id, name = result.name)
        return "Result not found"

    @api.route('/projects/params/result/<id>/progress', methods=['GET'])
    @flask_login.login_required
    def get_projects_params_result_progress(id):
        nonlocal compList
        id = int(id)
        if exporting_threads[id].result == "-":
            return "-"
        else:
            thread = exporting_threads[id]
            newResult = DBResult(results = thread.result, thr_id=id, name=thread.name)
            newResult.save()
            dbuser = flask_login.current_user.db
            dbuser.results.append(newResult)
            dbuser.save()
            return render_template('Result.html', compList = compList[flask_login.current_user.id], result = thread.result, user = flask_login.current_user.id)

    @api.route('/projects/result', methods=['GET'])
    @flask_login.login_required
    def get_projects_result_list():
        dbuser = flask_login.current_user.db
        resList = []
        for res in dbuser.results:
            resList.append(res)
        return render_template('ResultList.html', resList=resList)

    @api.route('/projects/result/<id>', methods=['GET'])
    @flask_login.login_required
    def get_projects_result_list_show(id):
        nonlocal compList, experimentSet
        if not flask_login.current_user.id in experimentSet:
            experimentSet[flask_login.current_user.id] = operator.Load_Experiment_Set(UPLOAD_FOLDER + "\\" + flask_login.current_user.id)
        experimentClusterComp = operator.Cluster_By_Component(experimentSet[flask_login.current_user.id])
        compList[flask_login.current_user.id] = experimentClusterComp.clusters.keys()
        id = int(id)
        dbuser = flask_login.current_user.db
        for res in dbuser.results:
            if res.thr_id == id:
                return render_template('ResultPage.html', compList = compList[flask_login.current_user.id], result = res.results, user = flask_login.current_user.id, name = res.name)
        return "Unknown result"

    @api.route('/projects/result/<id>', methods=['DELETE'])
    @flask_login.login_required
    def get_projects_result_list_delete(id):
        nonlocal compList
        id = int(id)
        dbuser = flask_login.current_user.db
        for exp in dbuser.results:
            if exp.thr_id == id:
                dbuser.results.remove(exp)
                exp.delete()
                dbuser.save()
                return redirect(url_for('get_projects_result_list'))
        return "Unknown result"

    @api.route('/projects', methods=['POST'])
    @flask_login.login_required
    def upload_file():
        nonlocal uploadedFiles
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('upload_file_page'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(api.config['UPLOAD_FOLDER']  + '\\' + flask_login.current_user.id, filename))
            newExperiment = DBExperiment(name=filename, experiment=Serialize_File_To_JSON(BASE_FOLDER + "\\docu\\TestUploadFolder\\" + flask_login.current_user.id + "\\" + filename))
            newExperiment.save()
            flask_login.current_user.db.experiments.append(newExperiment)
            flask_login.current_user.db.save()
            uploadedFiles[flask_login.current_user.id] = next(walk(UPLOAD_FOLDER + "\\" + flask_login.current_user.id), (None, None, []))[2]
            return redirect(url_for('upload_file_page'))

    @api.route('/projects', methods=['GET'])
    @flask_login.login_required
    def upload_file_page():
        nonlocal uploadedFiles
        uploadedFiles[flask_login.current_user.id] = next(walk(UPLOAD_FOLDER + '\\' + flask_login.current_user.id), (None, None, []))[2]
        return render_template('Upload.html', uploadedFilesLen = len(uploadedFiles[flask_login.current_user.id]), uploadedFiles = uploadedFiles[flask_login.current_user.id], user = flask_login.current_user.id)

    @api.route('/logout')
    def logout():
        flask_login.logout_user()
        return redirect(url_for('get_main_page'))

    @api.route('/register', methods=['GET', 'POST'])
    def get_registration():
        if request.method == 'GET':
            return render_template('Registration.html')
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        for user in DBUser.objects:
            if username == user.username:
                return render_template('Registration.html', badRegistration="User already exists.")
        newUser = DBUser(username=username, password_hash=password)
        newUser.save()
        return redirect(url_for('get_main_page'))

    @api.route('/', methods=['GET', 'POST'])
    def get_main_page():
        nonlocal formInfos
        if request.method == 'GET':
            if flask_login.current_user.is_authenticated:
                if not flask_login.current_user.id in formInfos:
                    formInfos[flask_login.current_user.id] = {}
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
                if not os.path.exists(UPLOAD_FOLDER + "\\" + user.id):
                    os.mkdir(UPLOAD_FOLDER + "\\" + user.id)
                return redirect(url_for('upload_file_page'))

        return render_template('Index.html', badLogin=True)
    api.run(debug=True)

