import React, { useState } from 'react';
import ImageUpload from '../../components/ImageUpload/ImageUpload';
import LoadingSpinner from '../../components/LoadingSpinner/LoadingSpinner';
import ResultCard from '../../components/ResultCard/ResultCard';
import api from '../../services/api';
import './Predict.css';

function Predict() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null); // Clear previous results
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      alert('Please select an image first');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const predictionResult = await api.predictMosquito(selectedFile);
      setResult(predictionResult);
    } catch (error) {
      setResult({
        success: false,
        message: error.response?.data?.detail || 'Prediction failed. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setLoading(false);
  };

  return (
    <div className="predict-page">
      <div className="predict-container">
        <div className="predict-header">
          <h1>🔬 Mosquito Species Prediction</h1>
          <p>Upload a clear image of a mosquito to identify its species</p>
        </div>

        <ImageUpload onImageSelect={handleImageSelect} preview={preview} />

        {preview && !loading && !result && (
          <div className="action-buttons">
            <button className="btn btn-primary btn-large" onClick={handlePredict}>
              🔍 Analyze Image
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              🔄 Reset
            </button>
          </div>
        )}

        {loading && <LoadingSpinner />}

        {result && (
          <>
            <ResultCard result={result} />
            <div className="action-buttons">
              <button className="btn btn-primary" onClick={handleReset}>
                📤 Upload Another Image
              </button>
            </div>
          </>
        )}

        <div className="tips-box">
          <h3>📸 Tips for Best Results</h3>
          <ul>
            <li>Use a clear, well-lit image</li>
            <li>Ensure the mosquito is in focus</li>
            <li>Capture distinctive features (thorax markings, stripes)</li>
            <li>Avoid blurry or distant shots</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Predict;