const mongoose = require('mongoose');

const ConversationSchema = new mongoose.Schema({
  role: { type: String, enum: ['user', 'agent'], required: true },
  content: { type: String, required: true },
  timestamp: { type: Date, default: Date.now },
  metadata: {
    confidence: Number,
    processingTime: Number,
    tokens: Number
  }
});

const AgentSessionSchema = new mongoose.Schema({
  sessionId: { type: String, unique: true, required: true },
  userId: { type: String, required: true, index: true },
  conversations: [ConversationSchema],
  memory: {
    preferences: mongoose.Schema.Types.Mixed,
    context: mongoose.Schema.Types.Mixed,
    learnings: [String]
  },
  status: { type: String, enum: ['active', 'completed'], default: 'active' },
  createdAt: { type: Date, default: Date.now },
  lastActivity: { type: Date, default: Date.now }
});

const CacheSchema = new mongoose.Schema({
  key: { type: String, unique: true, required: true },
  value: mongoose.Schema.Types.Mixed,
  expireAt: { 
    type: Date, 
    index: { expireAfterSeconds: 0 }
  }
});

const AgentSession = mongoose.model('AgentSession', AgentSessionSchema);
const Cache = mongoose.model('Cache', CacheSchema);

module.exports = { AgentSession, Cache };
