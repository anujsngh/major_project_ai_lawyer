import React, { useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

const App = () => {
  // State variables to store user input, search results, and loading state
  const [query, setQuery] = useState('');
  const [task, setTask] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false); // Add a loading state

  // Function to handle user input change
  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  // Function to handle task selection change
  const handleTaskChange = (e) => {
    setTask(e.target.value);
  };

  // Function to handle search/submit button click
  const handleSearch = () => {
    // Set loading state to true before making API call
    setIsLoading(true);
    
    // Call an API endpoint with query and task to retrieve search results
    fetch(`http://localhost:5000/api/search?query=${query}&task=${task}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Error: Failed to fetch search results');
        }
        return response.json();
      })
      .then((data) => {
        setSearchResults(data);
        setIsLoading(false); // Set loading state to false after API response
      })
      .catch((error) => {
        console.error(error);
        setIsLoading(false); // Set loading state to false on error
        toast.error('Failed to fetch search results. Please try again.', {
          position: toast.POSITION.TOP_CENTER
        });
      });
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Legal Search AI</h1>
      <div className="form-container">
        {/* Render a loading spinner or indicator based on loading state */}
        {isLoading ? (
          <div className="loading-spinner">
            {/* Replace with your desired loading spinner or indicator */}
            <p>Loading...</p>
          </div>
        ) : (
          <>
            <label htmlFor="queryInput" className="form-label">Enter your query:</label>
            <input
              type="text"
              id="queryInput"
              className="form-input"
              value={query}
              onChange={handleQueryChange}
            />
            <label htmlFor="taskSelect" className="form-label">Select task:</label>
            <select id="taskSelect" className="form-select" value={task} onChange={handleTaskChange}>
              <option value="precedent">Precedent Retrieval</option>
              <option value="statute">Statute Retrieval</option>
            </select>
            <button onClick={handleSearch} className="form-button">Search</button>
          </>
        )}
      </div>
      <h2 className="results-title">Search Results</h2>
      <ul className="results-list">
        {searchResults.map((result) => (
          <li key={result.docement_id} className="result-item">
            {/* <h3 className="result-title">{result.title}</h3> */}
            {/* <p className="result-summary">{result.summary}</p> */}
            <p className="result-summary">{result.content}</p>
            <button className="result-button">View Document</button>
          </li>
        ))}
      </ul>
      <ToastContainer /> {/* Add ToastContainer to the app */}
    </div>
  );
};

export default App;
