
uploaded_files = []

function myFunction() {
  var x = document.getElementById("navDemo");
  if (x.className.indexOf("w3-show") == -1) {
    x.className += " w3-show";
  } else {
    x.className = x.className.replace(" w3-show", "");
  }
}

function add_file(){
    let name = document.getElementById('fileInput').files[0].name;
    uploaded_files.push(name)
}

function show_files(){
    if(uploaded_files.length > 0){
        addedHTML = ""
        for(const file of uploaded_files){
            addedHTML += file + "<br>"
        }
        document.getElementById("uploadedFiles").innerHTML = addedHTML
    }
}