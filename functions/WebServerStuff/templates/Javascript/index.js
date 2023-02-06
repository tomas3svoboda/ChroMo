
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
        console.log(divs)
        for(i=0; i < divs.length; i++){
            console.log(divs[i].classList[1])
            comp = divs[i].classList[1]
            innerHTML = divs[i].innerHTML
            innerHTML = innerHTML.replaceAll("Henry","Langmuir")
            additionalHTML =`
            <div id="saturConst">
                <label for="${comp}QStart">Choose a Saturation coeficient interval for {{comp}}:</label>
                [ <input type="number" value="${comp}QStart" id="${comp}QStart" name="{{comp}}QStart" min="0" step="1">, <input type="number" value="${comp}QEnd" id="${comp}QEnd" name="{{comp}}QEnd" min="0" step="1"> ]<br>
                <label for="${comp}QInit">Choose a Saturation coeficient initial guess for {{comp}}:</label>
                <input type="number" value="${comp}}QInit" id="${comp}QInit" name="{{comp}}QInit" min="0" step="1"><br>
            </div>
            `
            innerHTML = innerHTML + additionalHTML
            divs[i].innerHTML = innerHTML
        }
    }
    else if(solver == "Lin"){
        divs = document.getElementsByClassName("variableParams")
        for(i=0; i < divs.length; i++){
            comp = divs[i].classList[1]
            divs[i].innerHTML = divs[i].innerHTML.replaceAll("Langmuir","Henry")
            document.getElementById("saturConst").remove()
            divs[i].innerHTML = innerHTML
        }
    }
}