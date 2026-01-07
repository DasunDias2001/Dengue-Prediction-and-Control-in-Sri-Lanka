import React, { useState } from 'react';
import './Contact.css';

function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const [submitted, setSubmitted] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Here you would typically send the data to a backend
    console.log('Form submitted:', formData);
    setSubmitted(true);
    
    // Reset form after 3 seconds
    setTimeout(() => {
      setSubmitted(false);
      setFormData({
        name: '',
        email: '',
        subject: '',
        message: ''
      });
    }, 3000);
  };

  return (
    <div className="contact-page">
      <div className="contact-container">
        {/* Header */}
        <section className="contact-header">
          <h1>Get In Touch</h1>
          <p>Have questions? We'd love to hear from you!</p>
        </section>

        <div className="contact-content">
          {/* Contact Form */}
          <div className="contact-form-section">
            <h2>📧 Send Us a Message</h2>
            
            {submitted ? (
              <div className="success-message">
                <div className="success-icon">✅</div>
                <h3>Thank You!</h3>
                <p>Your message has been sent successfully. We'll get back to you soon!</p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="contact-form">
                <div className="form-group">
                  <label htmlFor="name">Name *</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    required
                    placeholder="Your full name"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="email">Email *</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    required
                    placeholder="your.email@example.com"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="subject">Subject *</label>
                  <input
                    type="text"
                    id="subject"
                    name="subject"
                    value={formData.subject}
                    onChange={handleChange}
                    required
                    placeholder="What is this about?"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="message">Message *</label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleChange}
                    required
                    rows="6"
                    placeholder="Tell us more..."
                  />
                </div>

                <button type="submit" className="btn-submit">
                  Send Message →
                </button>
              </form>
            )}
          </div>

          {/* Contact Info */}
          <div className="contact-info-section">
            <h2>📍 Contact Information</h2>
            
            <div className="info-card">
              <div className="info-icon">📧</div>
              <div className="info-content">
                <h3>Email</h3>
                <p>info@mosquitoai.com</p>
                <p>support@mosquitoai.com</p>
              </div>
            </div>

            <div className="info-card">
              <div className="info-icon">📱</div>
              <div className="info-content">
                <h3>Phone</h3>
                <p>+94 123 456 789</p>
                <p>Mon-Fri, 9:00 AM - 5:00 PM</p>
              </div>
            </div>

            <div className="info-card">
              <div className="info-icon">📍</div>
              <div className="info-content">
                <h3>Address</h3>
                <p>123 AI Street</p>
                <p>Colombo, Sri Lanka</p>
              </div>
            </div>

            <div className="info-card">
              <div className="info-icon">🌐</div>
              <div className="info-content">
                <h3>Social Media</h3>
                <div className="social-links">
                  <a href="#" className="social-link">Twitter</a>
                  <a href="#" className="social-link">LinkedIn</a>
                  <a href="#" className="social-link">GitHub</a>
                </div>
              </div>
            </div>

            <div className="faq-box">
              <h3>💡 Quick Help</h3>
              <p>
                Looking for immediate answers? Check out our 
                <a href="/how-it-works"> How It Works</a> page or 
                visit our <a href="/about"> About</a> section.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Contact;