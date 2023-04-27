function submitForm() {
	// Get input values
	var relevantCaseInput = document.getElementById("relevant-case-input").value;
	var relevantStatutesInput = document.getElementById("relevant-statutes-input").value;
	var statutesInput = document.getElementById("statutes-input").value;
	
	// Make AJAX request to server with the inputs
    // // get the value of relevant_case_input
    // var relevant_case_input = document.getElementById('relevant-case-input').value;

    // send the value to the Flask server using AJAX
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/search');
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function() {
        if (xhr.status === 200) {
            // handle successful response from the Flask server
            console.log(xhr.responseText);
        } else {
            // handle error response from the Flask server
            console.log('Error: ' + xhr.statusText);
        }
    };
    xhr.send(JSON.stringify({relevant_case_input: relevant_case_input}));
    
	
	// Display the results in the result section
	var resultSection = document.getElementById("result-section");
	resultSection.innerHTML = "";
	
	if (relevantCaseInput !== "") {
		var relevantCaseResult = "<h2>Relevant Case Search Results</h2><ul class='result-list'><li>Result 1</li><li>Result 2</li><li>Result 3</li></ul>";
		resultSection.innerHTML += relevantCaseResult;
	}
	
	if (relevantStatutesInput !== "") {
		var relevantStatutesResult = "<h2>Relevant Statutes Search Results</h2><ul class='result-list'><li>Result 1</li><li>Result 2</li><li>Result 3</li></ul>";
		resultSection.innerHTML += relevantStatutesResult;
	}
	
	if (statutesInput !== "") {
		var statutesResult = "<h2>Statutes Search Results</h2><ul class='result-list'><li>Result 1</li><li>Result 2</li><li>Result 3</li></ul>";
		resultSection.innerHTML += statutesResult;
	}
}
