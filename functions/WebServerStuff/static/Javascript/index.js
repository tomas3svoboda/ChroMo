
uploaded_files = []

function myFunction() {
  var x = document.getElementById("navDemo");
  if (x.className.indexOf("w3-show") == -1) {
    x.className += " w3-show";
  } else {
    x.className = x.className.replace(" w3-show", "");
  }
}

function formSolverChange(){
    solver = document.getElementById("solver").value
    if(solver == "Nonlin"){
        divs = document.getElementsByClassName("variableParams")
        divs2 = document.getElementsByClassName("nonLinParam")
        inputs = document.querySelectorAll("#nonLinParam > input")
        for(i=0; i < divs.length; i++){
            divs[i].innerHTML = divs[i].innerHTML.replaceAll("Henry","Langmuir")
        }
        for(i=0; i < divs2.length; i++){
            divs2[i].style.display = "block"
        }
        for(x of document.querySelectorAll(".nonLinParam > input")){
            x.setAttribute("required", "")
        }
    }
    else if(solver == "Lin"){
        divs = document.getElementsByClassName("variableParams")
        divs2 = document.getElementsByClassName("nonLinParam")
        for(i=0; i < divs.length; i++){
            divs[i].innerHTML = divs[i].innerHTML.replaceAll("Langmuir","Henry")
        }
        for(i=0; i < divs2.length; i++){
            divs2[i].style.display = "none"
        }
        for(x of document.querySelectorAll(".nonLinParam > input")){
            x.removeAttribute("required")
        }
    }
}

function formSolverChangeForTest(){
    solver = document.getElementById("solverTest").value
    div = document.getElementById("nonLinTest")
    params = document.getElementById("paramsTest")
    if(solver == "Nonlin"){
        div.innerHTML =`
                <label for="saturationTest">Choose a Saturation <div class="required">*</div></label>
                <input type="number" value="saturationTest" id="saturationTest" name="saturationTest" min="0" step="1" required><br>
           `
        params.innerHTML = params.innerHTML.replaceAll("Henry","Langmuir")
    }
    else if(solver == "Lin"){
        div.innerHTML = ""
        params.innerHTML = params.innerHTML.replaceAll("Langmuir","Henry")
    }
}

function formCompChangeForTest2(c = "componentTest", d = "expListDiv"){
    comp = document.getElementById(c).value
    divs = document.getElementsByClassName(d)
    for(i=0; i<divs.length; i++){
        divs[i].style.display = "none"
    }
    if( comp != "all" ){
        div = document.getElementById(d + comp).style.display = "block"
    }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function showGraph(){
    document.getElementById("newAngleButton").style.display = "none"
    document.getElementById("downloadMatrixButton").style.display = "none"
    let form = document.getElementById("mainform")
    let data = new FormData(form)
    for (const key of data.keys()){
        if(data.get(key) == ""){
            if(key == "retCorrThreshold" && !(data.get("retCorrTest") || data.get("retCorr"))){
                continue
            }
            window.alert("Field " + key + " not filled!")
            return
        }
    }
    let resp = await fetch(window.location.href, {
        "method": "POST",
        "body": data,
    })
    let path = await resp.text()
    let content = "Estimated time remaining: "
    do{
        await sleep(500)
        let resp2 = await fetch(path)
        content = await resp2.text()
        document.getElementById("imgdiv").innerHTML = content
        if(!content.startsWith("Estimated time remaining: "))
            document.getElementById("newAngleButton").style.display = "inline-block"
            document.getElementById("downloadMatrixButton").style.display = "inline-block"
    }while(content.startsWith("Estimated time remaining: "))
}

async function newAngle(){
    let url = document.getElementById("graphImg").getAttribute('url')
    let resp = await fetch(url)
    let content = await resp.text()
    document.getElementById("imgdiv").innerHTML = content
}

async function showGraph2(f = "mainform", aurl = "", i = "imgdiv", showEl = ""){
    if(showEl)
        document.getElementById(showEl).style.display = "none"
    let form = document.getElementById(f)
    let data = new FormData(form)
    for (const key of data.keys()){
        if(data.get(key) == ""){
            if(key == "retCorrThreshold" && !(data.get("retCorrTest") || data.get("retCorr"))){
                continue
            }
            window.alert("Field " + key + " not filled!")
            return
        }
    }
    let resp = await fetch(window.location.href + aurl, {
        "method": "POST",
        "body": data,
    })
    let path = await resp.text()
    while(1){
        await sleep(500)
        let resp2 = await fetch(path)
        let content = await resp2.text()
        document.getElementById(i).innerHTML = content
        if(content != ""){
            if(showEl)
                document.getElementById(showEl).style.display = "inline-block"
            break
        }
    }
}


function continueFromTesting(){
    let form = document.getElementById("mainform")
    let data = new FormData(form)
    fetch(window.location.href+"/continue", {
        "method": "POST",
        "body": data,
    }).then(response => response.text()
    ).then(response => document.getElementById("imgdiv").innerHTML = response)
}

async function getResult(){
    while(1){
        let resp = await fetch(window.location.href + "/progress")
        let result = await resp.text()
        document.getElementById("resultDiv").innerHTML = result
        if(!result.startsWith("Time elapsed: ")){
            document.location.reload()
            break
        }
        await sleep(500)
    }
}

function downloadMatrix(picId = "graphImg"){
    let url = document.getElementById(picId).getAttribute('durl')
    document.getElementById('invis_iframe').src = url;
}

function retTimeTresholdInput(){
    let div = document.getElementById("retCorrTresholdDiv")
    let check = document.getElementById("retCorrTest")
    if(check == null)
        check = document.getElementById("retCorr")
    let val = check.checked
    if(val){
        div.style.display = "block"
    }
    else{
        div.style.display = "none"
    }
}

async function showGraph3(f = "mainform", aurl = "/prograph", i = "imgdiv"){
    let form = document.getElementById(f)
    let data = new FormData(form)
    for (const key of data.keys()){
        if(data.get(key) == ""){
            window.alert("Field " + key + " not filled!")
            return
        }
    }
    let resp = await fetch(window.location.href + aurl, {
        "method": "POST",
        "body": data,
    })
    let content = await resp.text()
    document.getElementById(i).innerHTML = content
}

