import React from 'react';
import './LoadingSpinner.css';

function LoadingSpinner() {
  return (
    <div className="loading-spinner">
      <div className="spinner"></div>
      <p>Analyzing image...</p>
    </div>
  );
}

export default LoadingSpinner;