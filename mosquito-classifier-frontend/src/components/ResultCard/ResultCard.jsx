import React from 'react';
import './ResultCard.css';

function ResultCard({ result }) {
  if (!result || !result.success) {
    return (
      <div className="result-card error">
        <h3>❌ Prediction Failed</h3>
        <p>{result?.message || 'An error occurred during prediction'}</p>
      </div>
    );
  }

  const { predicted_class, confidence, probabilities } = result;
  const isConfident = confidence > 0.7;

  return (
    <div className="result-card">
      <div className="result-header">
        <h2>🔬 Analysis Results</h2>
      </div>

      <div className="result-prediction">
        <div className="predicted-species">
          <span className="species-label">Predicted Species:</span>
          <span className={`species-name ${predicted_class}`}>
            {predicted_class === 'aegypti' ? 'Aedes aegypti' : 'Aedes albopictus'}
          </span>
        </div>

        <div className="confidence-badge">
          <span className={`confidence ${isConfident ? 'high' : 'low'}`}>
            {(confidence * 100).toFixed(1)}% Confidence
          </span>
        </div>
      </div>

      <div className="probabilities">
        <h3>Probability Distribution</h3>
        
        <div className="probability-bar">
          <div className="probability-label">
            <span>🦟 Aedes aegypti</span>
            <span>{(probabilities.aegypti * 100).toFixed(1)}%</span>
          </div>
          <div className="progress-bar">
            <div 
              className="progress-fill aegypti"
              style={{ width: `${probabilities.aegypti * 100}%` }}
            ></div>
          </div>
        </div>

        <div className="probability-bar">
          <div className="probability-label">
            <span>🦟 Aedes albopictus</span>
            <span>{(probabilities.albopictus * 100).toFixed(1)}%</span>
          </div>
          <div className="progress-bar">
            <div 
              className="progress-fill albopictus"
              style={{ width: `${probabilities.albopictus * 100}%` }}
            ></div>
          </div>
        </div>
      </div>

      {!isConfident && (
        <div className="warning-box">
          <span className="warning-icon">⚠️</span>
          <p>Low confidence prediction. Consider uploading a clearer image for better accuracy.</p>
        </div>
      )}

      <div className="species-info-box">
        <h4>{predicted_class === 'aegypti' ? 'About Aedes aegypti' : 'About Aedes albopictus'}</h4>
        {predicted_class === 'aegypti' ? (
          <ul>
            <li>Also known as Yellow Fever Mosquito</li>
            <li>Primary vector for dengue, Zika, and chikungunya</li>
            <li>Lyre-shaped white markings on thorax</li>
            <li>Active during daytime</li>
          </ul>
        ) : (
          <ul>
            <li>Also known as Asian Tiger Mosquito</li>
            <li>Secondary vector for dengue and chikungunya</li>
            <li>Single white stripe down center of head and back</li>
            <li>Aggressive daytime biter</li>
          </ul>
        )}
      </div>
    </div>
  );
}

export default ResultCard;