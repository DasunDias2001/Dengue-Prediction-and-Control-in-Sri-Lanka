import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const api = {
  // Health check
  healthCheck: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/`);
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Predict adult mosquito species (Aedes aegypti vs Aedes albopictus)
  predictMosquito: async (imageFile) => {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  },

  // Predict larvae species (Aedes aegypti vs Culex quinquefasciatus)
  predictLarvae: async (imageFile) => {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const response = await axios.post(`${API_BASE_URL}/larvae/identify`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Larvae prediction failed:', error);
      throw error;
    }
  },

  // Get adult mosquito model info
  getModelInfo: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model-info`);
      return response.data;
    } catch (error) {
      console.error('Failed to get model info:', error);
      throw error;
    }
  },

  // Get larvae model info
  getLarvaeModelInfo: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/larvae/model-info`);
      return response.data;
    } catch (error) {
      console.error('Failed to get larvae model info:', error);
      throw error;
    }
  },

  // Get larvae classes info
  getLarvaeClasses: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/larvae/classes`);
      return response.data;
    } catch (error) {
      console.error('Failed to get larvae classes:', error);
      throw error;
    }
  },
};

export default api;