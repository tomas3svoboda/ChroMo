import os
from flask import Flask, flash, request, redirect, url_for, json
from werkzeug.utils import secure_filename
from functions.WebServerStuff.Serialize_File_To_JSON import Serialize_File_To_JSON
import time
import random
from objects.Operator import Operator
from objects.ExperimentSet import ExperimentSet

def Web_Server():
    projects = {}

    UPLOAD_FOLDER = 'C:\\Users\\Adam\\ChroMo\\docu\\TestUploadFolder'
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

    api = Flask(__name__)
    api.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @api.route('/projects/<id>/deserialized', methods=['GET'])
    def get_deserialized(id):
        experimentSet = ExperimentSet()
        operator = Operator()
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

    @api.route('/projects', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(api.config['UPLOAD_FOLDER'], filename))
                i = str(random.randint(0, 999))
                projects[i] = Serialize_File_To_JSON("C:\\Users\\Adam\\ChroMo\\docu\\TestUploadFolder\\TestExperiment3.xlsx")
                return redirect(url_for('get_projects', id=i))
        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        '''

    api.run(debug=True)

