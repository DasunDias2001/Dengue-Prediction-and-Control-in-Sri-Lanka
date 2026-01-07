import React from 'react';
import { Link } from 'react-router-dom';
import './HowItWorks.css';

function HowItWorks() {
  return (
    <div className="how-it-works-page">
      <div className="how-container">
        {/* Header */}
        <section className="how-header">
          <h1>How It Works</h1>
          <p>Understanding the technology behind MosquitoAI</p>
        </section>

        {/* Process Steps */}
        <section className="process-section">
          <h2>🔄 The Classification Process</h2>
          <div className="steps-container">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>📤 Upload Image</h3>
                <p>
                  Start by uploading a clear photograph of a mosquito. Our system accepts 
                  JPG, PNG, and other common image formats. For best results, ensure the 
                  mosquito is in focus and well-lit.
                </p>
              </div>
            </div>

            <div className="step-arrow">↓</div>

            <div className="step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>🔍 Image Preprocessing</h3>
                <p>
                  The uploaded image is automatically resized to 224×224 pixels and 
                  normalized. This ensures consistency and optimal performance of our 
                  neural network model.
                </p>
              </div>
            </div>

            <div className="step-arrow">↓</div>

            <div className="step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>🧠 AI Analysis</h3>
                <p>
                  Our deep learning model, trained on over 1,200 mosquito images, analyzes 
                  distinctive features like thorax markings, body stripes, and morphological 
                  characteristics to identify the species.
                </p>
              </div>
            </div>

            <div className="step-arrow">↓</div>

            <div className="step">
              <div className="step-number">4</div>
              <div className="step-content">
                <h3>📊 Results</h3>
                <p>
                  Receive instant predictions with confidence scores for both Aedes aegypti 
                  and Aedes albopictus. Our system provides probability distributions and 
                  species information to help you understand the results.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Technology Stack */}
        <section className="tech-stack-section">
          <h2>⚙️ Technology Stack</h2>
          <div className="tech-stack-grid">
            <div className="stack-card">
              <h3>🖼️ Frontend</h3>
              <ul>
                <li>React.js</li>
                <li>React Router</li>
                <li>Axios</li>
                <li>Modern CSS3</li>
              </ul>
            </div>

            <div className="stack-card">
              <h3>🔧 Backend</h3>
              <ul>
                <li>FastAPI (Python)</li>
                <li>Uvicorn Server</li>
                <li>RESTful API</li>
                <li>CORS Support</li>
              </ul>
            </div>

            <div className="stack-card">
              <h3>🤖 AI/ML</h3>
              <ul>
                <li>TensorFlow 2.16</li>
                <li>Keras</li>
                <li>CNN Architecture</li>
                <li>Transfer Learning</li>
              </ul>
            </div>

            <div className="stack-card">
              <h3>📊 Processing</h3>
              <ul>
                <li>OpenCV</li>
                <li>NumPy</li>
                <li>Image Augmentation</li>
                <li>Data Preprocessing</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Model Architecture */}
        <section className="model-section">
          <h2>🏗️ Model Architecture</h2>
          <div className="model-info">
            <div className="architecture-diagram">
              <div className="layer">Input Layer (224×224×3)</div>
              <div className="arrow-down">↓</div>
              <div className="layer">Convolutional Layers</div>
              <div className="arrow-down">↓</div>
              <div className="layer">Pooling Layers</div>
              <div className="arrow-down">↓</div>
              <div className="layer">Dropout (Regularization)</div>
              <div className="arrow-down">↓</div>
              <div className="layer">Fully Connected Layers</div>
              <div className="arrow-down">↓</div>
              <div className="layer">Output Layer (2 classes)</div>
            </div>

            <div className="model-details">
              <h3>Key Features:</h3>
              <ul>
                <li><strong>Input Size:</strong> 224×224 RGB images</li>
                <li><strong>Architecture:</strong> Convolutional Neural Network (CNN)</li>
                <li><strong>Training Data:</strong> 842 training images</li>
                <li><strong>Validation:</strong> 180 validation images</li>
                <li><strong>Test Set:</strong> 181 test images</li>
                <li><strong>Classes:</strong> Aedes aegypti, Aedes albopictus</li>
                <li><strong>Optimizer:</strong> Adam</li>
                <li><strong>Loss Function:</strong> Binary Crossentropy</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Accuracy Section */}
        <section className="accuracy-section">
          <h2>🎯 Performance Metrics</h2>
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-value">95%+</div>
              <div className="metric-label">Target Accuracy</div>
              <p>After proper training with clean data</p>
            </div>
            <div className="metric-card">
              <div className="metric-value">&lt;2s</div>
              <div className="metric-label">Prediction Speed</div>
              <p>Real-time inference on cloud infrastructure</p>
            </div>
            <div className="metric-card">
              <div className="metric-value">1203</div>
              <div className="metric-label">Dataset Size</div>
              <p>High-quality mosquito images</p>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="cta-section">
          <h2>Ready to Try It?</h2>
          <p>Upload your mosquito image and see our AI in action!</p>
          <Link to="/predict" className="btn btn-large">
            Start Classification →
          </Link>
        </section>
      </div>
    </div>
  );
}

export default HowItWorks;