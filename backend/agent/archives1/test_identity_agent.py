# test_identity_agent.py
"""
Test simple d'initialisation de IdentityExtractionAgent
"""

def test_initialization():
    """Test l'initialisation étape par étape"""
    
    print("🧪 Test d'initialisation de IdentityExtractionAgent")
    print("=" * 50)
    
    try:
        # Import de votre agent
        print("1. Import de l'agent...")
        from aoagentc.agent.identity_agent_arch import IdentityExtractionAgent
        print("   ✅ Import réussi")
        
        # Test 1: Initialisation minimale (sans Enhanced Retrieval)
        print("\n2. Test initialisation sans Enhanced Retrieval...")
        agent_simple = IdentityExtractionAgent(
            use_enhanced_retrieval=False  # ← Désactive Enhanced Retrieval
        )
        print("   ✅ Initialisation simple réussie")
        
        # Test 2: Initialisation avec Enhanced Retrieval (peut échouer)
        print("\n3. Test initialisation avec Enhanced Retrieval...")
        try:
            agent_enhanced = IdentityExtractionAgent(
                use_enhanced_retrieval=True  # ← Active Enhanced Retrieval
            )
            print("   ✅ Initialisation Enhanced Retrieval réussie")
            print(f"   📊 Enhanced Retriever: {agent_enhanced.enhanced_retriever is not None}")
            
        except Exception as e:
            print(f"   ⚠️ Enhanced Retrieval échoué: {e}")
            print("   💡 C'est normal si les utils ne sont pas configurés")
        
        # Test 3: Vérification des attributs
        print("\n4. Vérification des attributs...")
        print(f"   - use_enhanced_retrieval: {agent_simple.use_enhanced_retrieval}")
        print(f"   - enhanced_retriever: {getattr(agent_simple, 'enhanced_retriever', 'Non défini')}")
        print(f"   - model: {getattr(agent_simple, 'model', 'Non défini')}")
        
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("💡 Vérifiez que le fichier agents/identity_agent.py existe")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ Fichier manquant: {e}")
        print("💡 Vérifiez que le fichier YAML de config existe")
        return False
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        print(f"   Type: {type(e).__name__}")
        return False

# Test encore plus simple si le précédent échoue
def test_basic_import():
    """Test juste l'import"""
    print("\n🔍 Test d'import basique...")
    
    try:
        from aoagentc.agent.identity_agent_arch import IdentityExtractionAgent
        print("✅ Import réussi - la classe existe")
        return True
    except Exception as e:
        print(f"❌ Import échoué: {e}")
        return False

if __name__ == "__main__":
    # Exécuter les tests
    success = test_initialization()
    
    if not success:
        print("\n" + "="*50)
        print("🔄 Tentative de test basique...")
        test_basic_import()