import os
import shutil

def Handle_File_Creation(path, dir = False):
    exists = os.path.exists(path)
    if exists:
        head, tail = os.path.split(path)
        fileOrDir = "File "
        if dir:
            fileOrDir = "Directory "
        inp = input(fileOrDir + tail + " already exists and is about to be overwritten. Do you want to proceed?[Y - yes]")
        if inp != "Y":
            print("Shutting down.")
            exit()
        if dir:
            shutil.rmtree(path)
    if not dir:
        file = open(path, "w")
        return file
    if dir:
        os.mkdir(path)
        return True
