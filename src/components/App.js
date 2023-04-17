import React, { useState } from 'react';
import './App.css';

const App = () => {
  // State variables to store user input and search results
  const [query, setQuery] = useState('');
  const [task, setTask] = useState('');
  const [searchResults, setSearchResults] = useState([]);

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
    // Call an API endpoint with query and task to retrieve search results
    fetch(`http://localhost:5000/api/search?query=${query}&task=${task}`)
      .then((response) => response.json())
      .then((data) => setSearchResults(data));
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Legal Search AI</h1>
      <div className="form-container">
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
      </div>
      <h2 className="results-title">Search Results</h2>
      <ul className="results-list">
        {searchResults.map((result) => (
          <li key={result.id} className="result-item">
            <h3 className="result-title">{result.title}</h3>
            <p className="result-summary">{result.summary}</p>
            <button className="result-button">View Document</button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default App;
