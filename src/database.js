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
      console.log('‚úÖ MongoDB Atlas connected successfully!');
      console.log('Ì≥ä Database:', mongoose.connection.name);
    } catch (error) {
      console.error('‚ùå MongoDB connection failed:', error.message);
      throw error;
    }
  }

  async disconnect() {
    if (this.connected) {
      await mongoose.disconnect();
      this.connected = false;
      console.log('Ì¥å MongoDB disconnected');
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
