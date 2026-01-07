import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

function Home() {
  return (
    <div className="home">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1 className="hero-title">
            🦟 Identify Dengue Mosquitoes with AI
          </h1>
          <p className="hero-subtitle">
            Upload a mosquito image and get instant AI-powered species identification
          </p>
          <div className="hero-buttons">
            <Link to="/predict" className="btn btn-primary">
              Start Prediction
            </Link>
            <Link to="/how-it-works" className="btn btn-secondary">
              Learn More
            </Link>
          </div>
        </div>
        <div className="hero-image">
          <div className="hero-illustration">
            🔬🦟🤖
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <h2 className="section-title">Why Choose MosquitoAI?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Fast & Accurate</h3>
            <p>Get instant predictions powered by deep learning models trained on thousands of mosquito images</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🎯</div>
            <h3>High Precision</h3>
            <p>Distinguish between Aedes aegypti and Aedes albopictus with advanced computer vision</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">📱</div>
            <h3>Easy to Use</h3>
            <p>Simple drag-and-drop interface. Just upload an image and get results in seconds</p>
          </div>

          <div className="feature-card">
            <div className="feature-icon">🔬</div>
            <h3>Scientific</h3>
            <p>Built on peer-reviewed research and validated classification algorithms</p>
          </div>
        </div>
      </section>

      {/* Species Info Section */}
      <section className="species-info">
        <h2 className="section-title">Mosquito Species We Identify</h2>
        <div className="species-grid">
          <div className="species-card">
            <h3>🦟 Aedes aegypti</h3>
            <ul>
              <li>Yellow fever mosquito</li>
              <li>Lyre-shaped white markings on thorax</li>
              <li>Primary dengue vector</li>
              <li>Daytime biter</li>
            </ul>
          </div>

          <div className="species-card">
            <h3>🦟 Aedes albopictus</h3>
            <ul>
              <li>Asian tiger mosquito</li>
              <li>Single white stripe down head/thorax</li>
              <li>Secondary dengue vector</li>
              <li>Aggressive biter</li>
            </ul>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta">
        <div className="cta-content">
          <h2>Ready to Identify?</h2>
          <p>Upload your mosquito image now and get instant AI analysis</p>
          <Link to="/predict" className="btn btn-large">
            Try It Now →
          </Link>
        </div>
      </section>
    </div>
  );
}

export default Home;