const AgentMongoDB = require('./src/database');
const { AgentSession, Cache } = require('./models/Agent');

async function testMongoDB() {
  const db = new AgentMongoDB();
  
  try {
    console.log('Ì¥Ñ Connecting to MongoDB Atlas...');
    await db.connect();
    
    console.log('Ì≥ä Database Stats:', db.getStats());
    
    console.log('Ì¥ñ Creating test agent session...');
    const testSession = new AgentSession({
      sessionId: 'test_' + Date.now(),
      userId: 'serigne',
      conversations: [
        {
          role: 'user',
          content: 'Hello agent!'
        },
        {
          role: 'agent',
          content: 'Hello! I am your MongoDB-powered agent. Ready to help!',
          metadata: { confidence: 0.9 }
        }
      ],
      memory: {
        preferences: { language: 'fr', style: 'friendly' },
        context: { project: 'agent-dev' }
      }
    });
    
    await testSession.save();
    console.log('‚úÖ Agent session created:', testSession.sessionId);
    
    console.log('Ì≤æ Testing cache functionality...');
    const cache = new Cache({
      key: 'test_cache',
      value: { message: 'This is cached data', timestamp: new Date() },
      expireAt: new Date(Date.now() + 3600000)
    });
    
    await cache.save();
    console.log('‚úÖ Cache entry created');
    
    const sessions = await AgentSession.find({ userId: 'serigne' });
    console.log(`Ì≥ã Found ${sessions.length} session(s) for user serigne`);
    
    const cacheData = await Cache.findOne({ key: 'test_cache' });
    console.log('Ì≤æ Cache data:', cacheData?.value);
    
    console.log('\nÌæâ MongoDB setup successful! Your agent database is ready.');
    console.log('Ì¥ó You can view your data at: https://cloud.mongodb.com');
    
  } catch (error) {
    console.error('‚ùå Test failed:', error.message);
  } finally {
    await db.disconnect();
    process.exit(0);
  }
}

testMongoDB();
