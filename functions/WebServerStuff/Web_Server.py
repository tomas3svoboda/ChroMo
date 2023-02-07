import os
from flask import Flask, flash, request, redirect, url_for, send_file, render_template, json
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from functions.WebServerStuff.Serialize_File_To_JSON import Serialize_File_To_JSON
from functions.Loss_Function_Analysis_Simple import Loss_Function_Analysis_Simple
import random
from objects.Operator import Operator
from objects.ExperimentSet import ExperimentSet
from os import walk

def Web_Server():
    projects = {}

    plotFileCounter = 1
    experimentSet = ExperimentSet()
    compList = []
    operator = Operator()

    UPLOAD_FOLDER = 'C:\\Users\\Z004PTSU\\PycharmProjects\\ChroMo\\docu\\TestUploadFolder'
    BASE_FOLDER = 'C:\\Users\\Z004PTSU\\PycharmProjects\\ChroMo'
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

    api = Flask(__name__)
    api.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    uploadedFiles = next(walk(UPLOAD_FOLDER), (None, None, []))[2]

    @api.route('/projects/<id>/deserialized', methods=['GET'])
    def get_deserialized(id):
        operator.Load_Experimet_JSON(experimentSet, projects[id])
        print(experimentSet)
        print(experimentSet.experiments)
        print(experimentSet.experiments[0])
        print(experimentSet.experiments[0].experimentComponents)
        print(experimentSet.experiments[0].experimentComponents[0])
        print(experimentSet.experiments[0].experimentComponents[0].concentrationTime)
        return json.dumps(projects[id])

    @api.route('/projects/<id>', methods=['GET'])
    def get_projects(id):
        return json.dumps(projects[id])

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @api.route('/projects/test', methods=['GET'])
    def get_projects_test():
        nonlocal experimentSet, compList
        experimentSet = operator.Load_Experiment_Set(UPLOAD_FOLDER)
        experimentClusterComp = operator.Cluster_By_Component(experimentSet)
        compList = experimentClusterComp.clusters.keys()
        return render_template('ParamsTestForm.html', compList = compList)

    @api.route('/projects/test', methods=['POST'])
    def post_projects_test():
        nonlocal experimentSet, compList, plotFileCounter
        gauss = bool(request.form.get("gaussTest"))
        retCorr = bool(request.form.get("retCorrTest"))
        massBal = bool(request.form.get("massBalTest"))
        currExperimentSet = operator.Preprocess(experimentSet, gauss, retCorr, massBal)
        lossFunc = str(request.form.get("lossFuncTest"))
        solver = str(request.form.get("solverTest"))
        factor = int(request.form.get("factorTest"))
        porosity = float(request.form.get("porosityTest"))
        saturation = 0
        if solver == "Nonlin":
            saturation = float(request.form.get("saturationTest"))
        comp = str(request.form.get("componentTest"))
        KStart = float(request.form.get("KStartTest"))
        KEnd = float(request.form.get("KEndTest"))
        KStep = float(request.form.get("KStepTest"))
        DStart = float(request.form.get("DStartTest"))
        DEnd = float(request.form.get("DEndTest"))
        DStep = float(request.form.get("DStepTest"))
        experimentClusterComp = operator.Cluster_By_Component(currExperimentSet)
        filename = "plot" + str(plotFileCounter) + ".png"
        plotFileCounter += 1
        Loss_Function_Analysis_Simple(experimentClusterComp, comp, "", KStart, DStart, KEnd, DEnd, KStep, DStep, porosity, saturation, lossFunc, factor, solver, True)
        plt.savefig('functions/WebServerStuff/static/images/' + filename)
        return render_template('ParamsTestForm.html', compList = compList, plotUrl = url_for('static', filename='images/'+filename))

    @api.route('/projects/params', methods=['GET'])
    def get_projects_params():
        nonlocal experimentSet, compList
        experimentSet = operator.Load_Experiment_Set(UPLOAD_FOLDER)
        experimentClusterComp = operator.Cluster_By_Component(experimentSet)
        compList = experimentClusterComp.clusters.keys()
        return render_template('ParamsForm.html', compList = compList)

    @api.route('/projects/params', methods=['POST'])
    def post_projects_params():
        gauss = bool(request.form.get("gauss"))
        retCorr = bool(request.form.get("retCorr"))
        massBal = bool(request.form.get("massBal"))
        lossFunc = str(request.form.get("lossFunc"))
        solver = str(request.form.get("solver"))
        factor = str(request.form.get("factor"))
        porosityStart = float(request.form.get("porosityStart"))
        porosityEnd = float(request.form.get("porosityEnd"))
        porosityInit = float(request.form.get("porosityInit"))
        KDQDict = {}
        for comp in compList:
            tmpDict = dict()
            KStart = float(request.form.get(comp + "KStart"))
            KEnd = float(request.form.get(comp + "KEnd"))
            tmpDict["kinit"] = float(request.form.get(comp + "KInit"))
            tmpDict["krange"] = [KStart, KEnd]
            DStart = float(request.form.get(comp + "DStart"))
            DEnd = float(request.form.get(comp + "DEnd"))
            tmpDict["dinit"] = float(request.form.get(comp + "DInit"))
            tmpDict["drange"] = [DStart, DEnd]
            if solver == "Nonlin":
                QStart = float(request.form.get(comp + "QStart"))
                QEnd = float(request.form.get(comp + "QEnd"))
                tmpDict["qinit"] = float(request.form.get(comp + "QInit"))
                tmpDict["qrange"] = [QStart, QEnd]
            else:
                tmpDict["qinit"] = 0
                tmpDict["qrange"] = [0, 0]
            KDQDict[comp] = tmpDict
        operator.Web_Start(experimentSet, UPLOAD_FOLDER, gauss, retCorr, massBal, lossFunc, solver, factor,
                           porosityStart, porosityEnd, porosityInit, KDQDict)
        return redirect(url_for('get_projects_params'))

    @api.route('/projects', methods=['POST'])
    def upload_file():
        nonlocal uploadedFiles
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('upload_file_page'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(api.config['UPLOAD_FOLDER'], filename))
            i = str(random.randint(0, 999))
            projects[i] = Serialize_File_To_JSON(BASE_FOLDER + "\\docu\\TestUploadFolder\\" + filename)
            uploadedFiles = next(walk(UPLOAD_FOLDER), (None, None, []))[2]
            return redirect(url_for('upload_file_page'))

    @api.route('/projects', methods=['GET'])
    def upload_file_page():
        nonlocal uploadedFiles
        uploadedFiles = next(walk(UPLOAD_FOLDER), (None, None, []))[2]
        return render_template('Upload.html', uploadedFilesLen = len(uploadedFiles), uploadedFiles = uploadedFiles)

    @api.route('/', methods=['GET'])
    def get_main_page():
        return render_template('Index.html')


    api.config['SECRET_KEY'] = 'super secret key'
    api.config['SESSION_TYPE'] = 'development'
    api.run(debug=True)

