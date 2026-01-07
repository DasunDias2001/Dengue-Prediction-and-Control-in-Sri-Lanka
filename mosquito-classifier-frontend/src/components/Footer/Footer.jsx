import React from 'react';
import './Footer.css';

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-section">
          <h3>🦟 MosquitoAI</h3>
          <p>AI-powered dengue mosquito species identification</p>
        </div>

        <div className="footer-section">
          <h4>Quick Links</h4>
          <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/predict">Predict</a></li>
            <li><a href="/about">About</a></li>
            <li><a href="/contact">Contact</a></li>
          </ul>
        </div>

        <div className="footer-section">
          <h4>Resources</h4>
          <ul>
            <li><a href="/how-it-works">How It Works</a></li>
            <li><a href="#">API Documentation</a></li>
            <li><a href="#">Research Paper</a></li>
            <li><a href="#">Privacy Policy</a></li>
          </ul>
        </div>

        <div className="footer-section">
          <h4>Contact</h4>
          <p>📧 info@mosquitoai.com</p>
          <p>📱 +94 123 456 789</p>
          <p>📍 Colombo, Sri Lanka</p>
        </div>
      </div>

      <div className="footer-bottom">
        <p>&copy; 2026 MosquitoAI. All rights reserved.</p>
      </div>
    </footer>
  );
}

export default Footer;