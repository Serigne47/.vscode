# utils/vectorize_documents.py
"""
Script pour vectoriser les documents extraits par Enhanced Chunker
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from aoagentc.utils.arch0.enhanced_chunker1 import EnhancedChunker
from aoagentc.utils.arch0.vectorstore1 import AOVectorStore, get_vectorstore, index_extracted_elements

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def vectorize_from_json(
    json_file: Path,
    collection_name: str = "ao_documents",
    clear_existing: bool = False
) -> AOVectorStore:
    """
    Vectorise à partir d'un fichier JSON d'éléments extraits
    
    Args:
        json_file: Fichier JSON des éléments extraits
        collection_name: Nom de la collection ChromaDB
        clear_existing: Supprimer la collection existante
        
    Returns:
        Instance du vectorstore
    """
    logger.info(f"📄 Chargement de {json_file}")
    
    # Charger les éléments extraits
    with open(json_file, 'r', encoding='utf-8') as f:
        extracted_elements = json.load(f)
    
    logger.info(f"✅ {len(extracted_elements)} éléments chargés")
    
    # Initialiser le vectorstore
    vectorstore = get_vectorstore(collection_name=collection_name)
    
    # Supprimer la collection existante si demandé
    if clear_existing:
        logger.info("🗑️ Suppression de la collection existante...")
        vectorstore.delete_collection()
        vectorstore = get_vectorstore(collection_name=collection_name)
    
    # Indexer les éléments
    vectorstore = index_extracted_elements(extracted_elements, vectorstore)
    
    return vectorstore

def vectorize_from_folder(
    folder_path: Path,
    collection_name: str = "ao_documents",
    clear_existing: bool = False,
    save_extracted: bool = True
) -> AOVectorStore:
    """
    Extraction et vectorisation complète depuis un dossier
    
    Args:
        folder_path: Dossier contenant les documents
        collection_name: Nom de la collection ChromaDB
        clear_existing: Supprimer la collection existante
        save_extracted: Sauvegarder les éléments extraits en JSON
        
    Returns:
        Instance du vectorstore
    """
    logger.info(f"🚀 Vectorisation complète depuis {folder_path}")
    
    # Étape 1: Extraction avec Enhanced Chunker
    logger.info("📄 Extraction des documents...")
    chunker = EnhancedChunker()
    extracted_elements = chunker.process_folder(folder_path)
    
    # Sauvegarder les éléments extraits si demandé
    if save_extracted:
        output_file = Path("extracted_elements.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_elements, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"💾 Éléments sauvegardés dans {output_file}")
    
    # Étape 2: Vectorisation
    logger.info("🔄 Vectorisation...")
    vectorstore = get_vectorstore(collection_name=collection_name)
    
    if clear_existing:
        logger.info("🗑️ Suppression de la collection existante...")
        vectorstore.delete_collection()
        vectorstore = get_vectorstore(collection_name=collection_name)
    
    # Indexer les éléments
    vectorstore = index_extracted_elements(extracted_elements, vectorstore)
    
    return vectorstore

def test_vectorstore(vectorstore: AOVectorStore):
    """
    Teste les fonctionnalités du vectorstore
    """
    logger.info("🧪 Test du vectorstore...")
    
    # Stats générales
    stats = vectorstore.get_collection_stats()
    logger.info(f"📊 Statistiques: {stats}")
    
    # Tests de recherche par domaine
    test_queries = {
        "financial": "price tariff cost budget payment",
        "legal": "responsibility clause insurance penalty",
        "operational": "transport maritime air volume TEU",
        "timeline": "deadline deadline delay date planning"
    }
    
    for domain, query in test_queries.items():
        logger.info(f"\n🔍 Test recherche {domain}: '{query}'")
        
        # Recherche standard
        results = vectorstore.similarity_search(query, k=3)
        logger.info(f"   Recherche standard: {len(results)} résultats")
        
        # Recherche par domaine
        domain_results = vectorstore.search_by_domain(query, domain, k=3)
        logger.info(f"   Recherche {domain}: {len(domain_results)} résultats")
        
        # Afficher le premier résultat
        if domain_results:
            first_result = domain_results[0]
            logger.info(f"   Premier résultat: {first_result.metadata.get('type', 'N/A')} "
                       f"(priorité {domain}: {first_result.metadata.get(f'priority_{domain}', 0):.2f})")
            logger.info(f"   Texte: {first_result.page_content[:100]}...")

def main():
    """
    Script principal de vectorisation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Vectorisation des documents AO")
    parser.add_argument("--source", type=str, required=True,
                       help="Chemin vers le dossier de documents ou fichier JSON")
    parser.add_argument("--collection", type=str, default="ao_documents",
                       help="Nom de la collection ChromaDB")
    parser.add_argument("--clear", action="store_true",
                       help="Supprimer la collection existante")
    parser.add_argument("--test", action="store_true",
                       help="Tester le vectorstore après création")
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    
    try:
        if source_path.suffix == '.json':
            # Vectorisation depuis JSON
            vectorstore = vectorize_from_json(
                json_file=source_path,
                collection_name=args.collection,
                clear_existing=args.clear
            )
        elif source_path.is_dir():
            # Vectorisation depuis dossier
            vectorstore = vectorize_from_folder(
                folder_path=source_path,
                collection_name=args.collection,
                clear_existing=args.clear
            )
        else:
            raise ValueError(f"Source invalide: {source_path}")
        
        # Test si demandé
        if args.test:
            test_vectorstore(vectorstore)
        
        logger.info("🎉 Vectorisation terminée avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise

if __name__ == "__main__":
    main()