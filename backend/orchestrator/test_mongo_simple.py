import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

async def test():
    url = os.getenv("MONGODB_URL")
    print(f"Testing connection...")
    
    client = AsyncIOMotorClient(url)
    try:
        result = await client.admin.command('ping')
        print("✅ Connexion réussie!")
        print(result)
    except Exception as e:
        print(f"❌ Erreur: {e}")
    finally:
        client.close()

asyncio.run(test())