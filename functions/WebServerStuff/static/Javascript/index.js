
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

function formCompChangeForTest2(){
    comp = document.getElementById("componentTest").value
    divs = document.getElementsByClassName("expListDiv")
    for(i=0; i<divs.length; i++){
        divs[i].style.display = "none"
    }
    div = document.getElementById("expListDiv" + comp).style.display = "block"
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
            window.alert("Field " + key + " not filled!")
            return
        }
    }
    let resp = await fetch(window.location.href, {
        "method": "POST",
        "body": data,
    })
    let path = await resp.text()
    let content = "0%"
    do{
        await sleep(500)
        let resp2 = await fetch(path)
        content = await resp2.text()
        document.getElementById("imgdiv").innerHTML = content
        if(!content.endsWith("%"))
            document.getElementById("newAngleButton").style.display = "inline-block"
            document.getElementById("downloadMatrixButton").style.display = "inline-block"
    }while(content.endsWith("%"))
}

async function newAngle(){
    let url = document.getElementById("graphImg").getAttribute('url')
    let resp = await fetch(url)
    let content = await resp.text()
    document.getElementById("imgdiv").innerHTML = content
}

async function showGraph2(){
    let form = document.getElementById("mainform")
    let data = new FormData(form)
    for (const key of data.keys()){
        if(data.get(key) == ""){
            window.alert("Field " + key + " not filled!")
            return
        }
    }
    let resp = await fetch(window.location.href, {
        "method": "POST",
        "body": data,
    })
    let path = await resp.text()
    while(1){
        await sleep(500)
        let resp2 = await fetch(path)
        let content = await resp2.text()
        document.getElementById("imgdiv").innerHTML = content
        if(content != "")
            break
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
        if(result != "-"){
            document.getElementById("resultDiv").innerHTML = result
            break
        }
        document.getElementById("dotsDiv").innerHTML += "."
        if(document.getElementById("dotsDiv").innerHTML.length > 3)
            document.getElementById("dotsDiv").innerHTML = "."
        await sleep(500)
    }
}

function downloadMatrix(){
    let url = document.getElementById("graphImg").getAttribute('durl')
    document.getElementById('invis_iframe').src = url;
}

