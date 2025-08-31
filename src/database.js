const mongoose = require('mongoose');
require('dotenv').config();

class AgentMongoDB {
  constructor() {
    this.connected = false;
  }

  async connect() {
    try {
      await mongoose.connect(process.env.MONGODB_URI);
      this.connected = true;
      console.log('✅ MongoDB Atlas connected successfully!');
      console.log('� Database:', mongoose.connection.name);
    } catch (error) {
      console.error('❌ MongoDB connection failed:', error.message);
      throw error;
    }
  }

  async disconnect() {
    if (this.connected) {
      await mongoose.disconnect();
      this.connected = false;
      console.log('� MongoDB disconnected');
    }
  }

  getStats() {
    return {
      connected: this.connected,
      database: mongoose.connection.name,
      host: mongoose.connection.host
    };
  }
}

module.exports = AgentMongoDB;
