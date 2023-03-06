import json
import pandas as pd
import datetime

def Serialize_File_To_JSON(file):
    jsonDict = {}
    jsonDict["name"] = file
    jsonDict["projectDate"] = str(datetime.date.today().strftime("%m/%d/%Y"))
    if not (file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".csv")):
        raise("In Serialize_File_To_JSON: Wrong file type.")
    df = pd.read_excel(file)
    jsonDict["description"] = df.iat[0, 3]
    jsonDict["experimentDate"] = str(df.iat[2, 3])
    jsonDict["columnLength"] = float(df.iat[0, 1])
    jsonDict["columnDiameter"] = float(df.iat[1, 1])
    jsonDict["flowRate"] = float(df.iat[2, 1])
    jsonDict["feedVolume"] = float(df.iat[3, 1])
    jsonDict["deadVolume"] = float(df.iat[4, 1])
    jsonDict["components"] = []
    columnNames = df.iloc[[7]].to_numpy()[0]
    feedConcentrations = df.iloc[[6]].replace(',', '.', regex=True).to_numpy()[0][1:]
    df.drop([0, 1, 2, 3, 4, 5, 6, 7], axis=0, inplace=True)
    df.columns = columnNames
    df = df.replace(',', '.', regex=True).astype(float)
    for index in range(columnNames[1:].size):
        compDict = {}
        concentrationTime = df.iloc[:, [0, 1 + index]].astype(float)
        concentrationTime['Time'] = concentrationTime['Time'].apply(lambda x: x * 60)
        concentrationTime.reset_index(inplace=True, drop=True)
        compDict["name"] = columnNames[1 + index]
        compDict["concentrationTime"] = concentrationTime.to_json(orient="split")
        compDict["feedConcentration"] = float(feedConcentrations[index])
        jsonDict["components"].append(compDict)
    return json.dumps(jsonDict)