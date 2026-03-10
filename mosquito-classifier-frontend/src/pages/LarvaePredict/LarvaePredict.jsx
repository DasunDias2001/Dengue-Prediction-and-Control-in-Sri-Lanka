import React, { useState } from 'react';
import ImageUpload from '../../components/ImageUpload/ImageUpload';
import LoadingSpinner from '../../components/LoadingSpinner/LoadingSpinner';
import LarvaeResultCard from '../../components/LarvaeResultCard/LarvaeResultCard';
import api from '../../services/api';
import '../Predict/Predict.css';

function LarvaePredict() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      alert('Please select an image first');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const predictionResult = await api.predictLarvae(selectedFile);
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
          <h1>🔬 Mosquito Larvae Identification</h1>
          <p>Upload a clear image of mosquito larvae to identify its species</p>
        </div>

        <div className="larvae-species-banner">
          <div className="larvae-species-item">
            <span className="larvae-dot aedes"></span>
            <div>
              <strong>Aedes aegypti</strong>
              <small>Yellow Fever Mosquito · Dengue, Zika, Chikungunya</small>
            </div>
          </div>
          <div className="larvae-species-item">
            <span className="larvae-dot culex"></span>
            <div>
              <strong>Culex quinquefasciatus</strong>
              <small>Southern House Mosquito · West Nile Virus, Filariasis</small>
            </div>
          </div>
        </div>

        <ImageUpload onImageSelect={handleImageSelect} preview={preview} />

        {preview && !loading && !result && (
          <div className="action-buttons">
            <button className="btn btn-primary btn-large" onClick={handlePredict}>
              🔍 Identify Larvae
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              🔄 Reset
            </button>
          </div>
        )}

        {loading && <LoadingSpinner />}

        {result && (
          <>
            <LarvaeResultCard result={result} />
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
            <li>Use a clear, well-lit image of the larvae</li>
            <li>Capture the siphon (breathing tube) if possible</li>
            <li>Place larvae against a plain light background</li>
            <li>Avoid blurry or distant shots</li>
          </ul>
        </div>
      </div>

      <style>{`
        .larvae-species-banner {
          display: flex;
          gap: 16px;
          margin-bottom: 24px;
          flex-wrap: wrap;
        }
        .larvae-species-item {
          display: flex;
          align-items: center;
          gap: 10px;
          background: #f8f9fa;
          border: 1px solid #e9ecef;
          border-radius: 10px;
          padding: 12px 18px;
          flex: 1;
          min-width: 220px;
        }
        .larvae-species-item div {
          display: flex;
          flex-direction: column;
        }
        .larvae-species-item small {
          color: #6c757d;
          font-size: 12px;
          margin-top: 2px;
        }
        .larvae-dot {
          width: 14px;
          height: 14px;
          border-radius: 50%;
          flex-shrink: 0;
        }
        .larvae-dot.aedes  { background: #dc3545; }
        .larvae-dot.culex  { background: #28a745; }
      `}</style>
    </div>
  );
}

export default LarvaePredict;