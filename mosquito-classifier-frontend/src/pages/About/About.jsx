import React from 'react';
import './About.css';

function About() {
  return (
    <div className="about-page">
      <div className="about-container">
        {/* Header Section */}
        <section className="about-header">
          <h1>About MosquitoAI</h1>
          <p className="subtitle">
            Leveraging artificial intelligence to combat dengue through accurate mosquito identification
          </p>
        </section>

        {/* Mission Section */}
        <section className="mission-section">
          <div className="mission-content">
            <div className="mission-text">
              <h2>🎯 Our Mission</h2>
              <p>
                MosquitoAI is dedicated to providing fast, accurate, and accessible tools for 
                identifying dengue-carrying mosquito species. Our AI-powered platform helps 
                researchers, healthcare workers, and the public take preventive action against 
                dengue outbreaks.
              </p>
              <p>
                By democratizing access to advanced computer vision technology, we aim to 
                contribute to global efforts in disease vector surveillance and control.
              </p>
            </div>
            <div className="mission-image">
              <div className="image-placeholder">🦟🔬🌍</div>
            </div>
          </div>
        </section>

        {/* Technology Section */}
        <section className="technology-section">
          <h2>🤖 Our Technology</h2>
          <div className="tech-grid">
            <div className="tech-card">
              <div className="tech-icon">🧠</div>
              <h3>Deep Learning</h3>
              <p>
                Powered by state-of-the-art convolutional neural networks (CNNs) trained on 
                thousands of mosquito images for high accuracy classification.
              </p>
            </div>

            <div className="tech-card">
              <div className="tech-icon">📊</div>
              <h3>Computer Vision</h3>
              <p>
                Advanced image processing techniques to extract and analyze distinctive 
                morphological features of different mosquito species.
              </p>
            </div>

            <div className="tech-card">
              <div className="tech-icon">⚡</div>
              <h3>Real-time Analysis</h3>
              <p>
                Lightning-fast predictions using optimized models deployed on cloud 
                infrastructure for instant results.
              </p>
            </div>

            <div className="tech-card">
              <div className="tech-icon">🔒</div>
              <h3>Privacy First</h3>
              <p>
                All image processing happens securely. We don't store your uploaded images 
                and respect your privacy.
              </p>
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="stats-section">
          <h2>📈 Impact & Statistics</h2>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-number">95%+</div>
              <div className="stat-label">Model Accuracy</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">1200+</div>
              <div className="stat-label">Training Images</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">&lt;2s</div>
              <div className="stat-label">Prediction Time</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">2</div>
              <div className="stat-label">Species Identified</div>
            </div>
          </div>
        </section>

        {/* Team Section */}
        <section className="team-section">
          <h2>👥 Our Team</h2>
          <p className="team-intro">
            Built by a dedicated team of data scientists, entomologists, and public health experts 
            committed to fighting vector-borne diseases.
          </p>
          <div className="team-grid">
            <div className="team-member">
              <div className="member-avatar">👨‍💻</div>
              <h3>Research Team</h3>
              <p>AI & Machine Learning Specialists</p>
            </div>
            <div className="team-member">
              <div className="member-avatar">👩‍🔬</div>
              <h3>Entomologists</h3>
              <p>Mosquito Biology Experts</p>
            </div>
            <div className="team-member">
              <div className="member-avatar">👨‍⚕️</div>
              <h3>Public Health</h3>
              <p>Disease Control Advisors</p>
            </div>
          </div>
        </section>

        {/* Partnership Section */}
        <section className="partnership-section">
          <h2>🤝 Collaborations</h2>
          <p>
            We work closely with health organizations, research institutions, and government 
            agencies to improve dengue surveillance and control efforts worldwide.
          </p>
          <div className="partner-logos">
            <div className="partner-logo">🏥 Health Orgs</div>
            <div className="partner-logo">🎓 Universities</div>
            <div className="partner-logo">🏛️ Gov Agencies</div>
            <div className="partner-logo">🔬 Research Labs</div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default About;