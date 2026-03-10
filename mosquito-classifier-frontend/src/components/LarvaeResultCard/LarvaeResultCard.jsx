import React from 'react';
import './LarvaeResultCard.css';

function LarvaeResultCard({ result }) {
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
  const isAedes = predicted_class === 'Aedes aegypti';

  return (
    <div className="result-card">
      <div className="result-header">
        <h2>🔬 Larvae Analysis Results</h2>
      </div>

      <div className="result-prediction">
        <div className="predicted-species">
          <span className="species-label">Predicted Species:</span>
          <span className={`species-name ${isAedes ? 'aegypti' : 'albopictus'}`}>
            {predicted_class}
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
            <span>{((probabilities['Aedes aegypti'] || 0) * 100).toFixed(1)}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill aegypti"
              style={{ width: `${(probabilities['Aedes aegypti'] || 0) * 100}%` }}
            ></div>
          </div>
        </div>

        <div className="probability-bar">
          <div className="probability-label">
            <span>🦟 Culex quinquefasciatus</span>
            <span>{((probabilities['Culex quinquefasciatus'] || 0) * 100).toFixed(1)}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill albopictus"
              style={{ width: `${(probabilities['Culex quinquefasciatus'] || 0) * 100}%` }}
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
        <h4>{isAedes ? 'About Aedes aegypti Larvae' : 'About Culex quinquefasciatus Larvae'}</h4>
        {isAedes ? (
          <ul>
            <li>Also known as Yellow Fever Mosquito larvae</li>
            <li>Vector for Dengue, Zika, and Chikungunya</li>
            <li>Large siphon with a single pair of hair tufts</li>
            <li>Found in clean artificial water containers</li>
          </ul>
        ) : (
          <ul>
            <li>Also known as Southern House Mosquito larvae</li>
            <li>Vector for West Nile Virus and Lymphatic Filariasis</li>
            <li>Long slender siphon with multiple hair tufts</li>
            <li>Thrives in stagnant or polluted water</li>
          </ul>
        )}
      </div>
    </div>
  );
}

export default LarvaeResultCard;