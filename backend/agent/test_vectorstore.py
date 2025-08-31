# test_vectorstore.py
import sys
from pathlib import Path

# Ajouter le chemin parent au path Python
sys.path.insert(0, str(Path(__file__).parent.parent))  # Remonte à aoagentc/

# Maintenant on peut importer depuis utils
from aoagentc.utils.vectorstore import get_vectorstore

# Test
try:
    vs = get_vectorstore(collection_name="ao_documents")
    stats = vs.get_collection_stats()
    print(f"✅ Vectorstore connecté!")
    print(f"📊 Documents dans le vectorstore: {stats['document_count']}")
    
    if stats['document_count'] == 0:
        print("⚠️ Le vectorstore est vide!")
        print("Vous devez d'abord vectoriser vos documents avec:")
        print("python utils/vectorize_documents.py --source [VOTRE_DOSSIER] --collection ao_documents")
    else:
        print(f"✅ {stats['document_count']} documents disponibles")
        
        # Test de recherche
        results = vs.similarity_search("tender client reference", k=3)
        print(f"\n🔍 Test de recherche: {len(results)} résultats trouvés")
        
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()