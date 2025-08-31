# utils/vectorize_documents.py
"""
Script de vectorisation intelligent avec analyse LLM OPTIONNELLE
Utilise EnhancedChunker + IntelligentVectorStore
Mode par défaut : EMBEDDINGS ONLY pour économiser les tokens
"""
import logging
import json
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.enhanced_chunker import EnhancedChunker
from utils.vectorstore import IntelligentVectorStore, create_intelligent_store

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectorization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntelligentVectorizer:
    """
    Orchestrateur de vectorisation intelligent
    Combine chunking intelligent + analyse LLM OPTIONNELLE + indexation avancée
    """
    
    def __init__(
        self,
        persist_directory: Path = Path("data/intelligent_store"),
        chunk_size: int = 2000,
        analyze_chunks: bool = False,  # CHANGÉ : False par défaut pour économiser
        use_cache: bool = True,
        embedding_only_mode: bool = True  # NOUVEAU : Mode embeddings only par défaut
    ):
        """
        Initialise le vectorizer intelligent
        
        Args:
            persist_directory: Répertoire de stockage
            chunk_size: Taille des chunks (2000 par défaut)
            analyze_chunks: Analyser avec LLM lors de l'indexation (DÉSACTIVÉ par défaut)
            use_cache: Utiliser le cache pour éviter re-traitement
            embedding_only_mode: Utiliser uniquement les embeddings (sans analyse LLM)
        """
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.embedding_only_mode = embedding_only_mode
        
        # Si embedding only mode, forcer analyze_chunks à False
        if embedding_only_mode:
            self.analyze_chunks = False
            logger.info("💡 Mode EMBEDDINGS ONLY activé - Analyse LLM désactivée")
        else:
            self.analyze_chunks = analyze_chunks
            
        self.use_cache = use_cache
        
        # Tracking
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'chunks_analyzed': 0,
            'concepts_extracted': 0,
            'time_elapsed': 0,
            'tokens_saved': 0  # NOUVEAU : Tracker les tokens économisés
        }
        
        logger.info("🚀 IntelligentVectorizer initialisé")
        logger.info(f"   - Chunk size: {chunk_size}")
        logger.info(f"   - Analyse LLM: {'DÉSACTIVÉE (économie)' if not self.analyze_chunks else 'Activée'}")
        logger.info(f"   - Mode: {'EMBEDDINGS ONLY' if embedding_only_mode else 'COMPLET'}")
        logger.info(f"   - Cache: {use_cache}")
    
    def vectorize_from_folder(
        self,
        folder_path: Path,
        clear_existing: bool = False,
        save_report: bool = True
    ) -> IntelligentVectorStore:
        """
        Pipeline complet : Extraction → Analyse → Indexation
        
        Args:
            folder_path: Dossier contenant les documents
            clear_existing: Réinitialiser l'index existant
            save_report: Sauvegarder un rapport détaillé
            
        Returns:
            IntelligentVectorStore configuré et peuplé
        """
        start_time = time.time()
        logger.info(f"📁 Démarrage vectorisation depuis: {folder_path}")
        
        # ============================================================
        # PHASE 1: EXTRACTION INTELLIGENTE
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: EXTRACTION INTELLIGENTE")
        logger.info("="*60)
        
        chunker = EnhancedChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=200,
            intelligent_mode=True  # Mode intelligent activé
        )
        
        logger.info("📄 Extraction des documents en cours...")
        chunks = chunker.process_folder(folder_path, output_mode="intelligent")
        
        self.stats['documents_processed'] = len(set(
            c['metadata'].get('source', '') for c in chunks
        ))
        self.stats['chunks_created'] = len(chunks)
        
        logger.info(f"✅ Extraction terminée:")
        logger.info(f"   - Documents traités: {self.stats['documents_processed']}")
        logger.info(f"   - Chunks créés: {self.stats['chunks_created']}")
        logger.info(f"   - Taille moyenne: {sum(c['metadata'].get('char_count', 0) for c in chunks) / len(chunks):.0f} chars")
        
        # Calculer les tokens économisés
        if self.embedding_only_mode:
            tokens_per_chunk = 1500  # Estimation moyenne
            self.stats['tokens_saved'] = len(chunks) * tokens_per_chunk
            logger.info(f"   💰 Tokens économisés: ~{self.stats['tokens_saved']:,} ({self.stats['tokens_saved'] * 0.00015:.2f}$)")
        
        # Sauvegarder les chunks extraits
        self._save_chunks(chunks)
        
        # ============================================================
        # PHASE 2: ENRICHISSEMENT ET PRÉPARATION
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: ENRICHISSEMENT DES MÉTADONNÉES")
        logger.info("="*60)
        
        enriched_chunks = self._enrich_chunks(chunks)
        
        # ============================================================
        # PHASE 3: INDEXATION INTELLIGENTE
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: INDEXATION " + ("EMBEDDINGS ONLY" if self.embedding_only_mode else "INTELLIGENTE"))
        logger.info("="*60)
        
        # Créer ou réinitialiser le store
        if clear_existing:
            logger.info("🗑️ Suppression de l'index existant...")
            store_path = self.persist_directory
            if store_path.exists():
                import shutil
                shutil.rmtree(store_path)
        
        # Créer le vectorstore intelligent
        # IMPORTANT : Forcer analyze_on_index=False en mode embedding only
        vectorstore = create_intelligent_store(
            enriched_chunks,
            persist_directory=self.persist_directory,
            analyze_on_index=False if self.embedding_only_mode else self.analyze_chunks  # CHANGÉ
        )
        
        # ============================================================
        # PHASE 4: VALIDATION ET RAPPORT
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: VALIDATION ET RAPPORT")
        logger.info("="*60)
        
        self.stats['time_elapsed'] = time.time() - start_time
        
        # Tester quelques requêtes (SANS LLM en mode embedding only)
        validation_results = self._validate_vectorstore(vectorstore)
        
        # Générer le rapport
        if save_report:
            report = self._generate_report(
                chunks, 
                vectorstore, 
                validation_results
            )
            self._save_report(report)
        
        logger.info(f"\n✅ VECTORISATION COMPLÈTE EN {self.stats['time_elapsed']:.1f}s")
        logger.info(f"   - Index prêt: {vectorstore.get_stats()['total_documents']} documents")
        
        if self.embedding_only_mode:
            logger.info(f"   💰 MODE ÉCONOMIQUE: Seuls les embeddings ont été créés")
            logger.info(f"   💡 Pour activer l'analyse LLM: utilisez --enable-llm-analysis")
        else:
            logger.info(f"   - Concepts indexés: {vectorstore.get_stats()['total_concepts']}")
        
        return vectorstore
    
    def vectorize_from_json(
        self,
        json_file: Path,
        clear_existing: bool = False
    ) -> IntelligentVectorStore:
        """
        Vectorise depuis un fichier JSON de chunks déjà extraits
        
        Args:
            json_file: Fichier JSON contenant les chunks
            clear_existing: Réinitialiser l'index
            
        Returns:
            IntelligentVectorStore configuré
        """
        logger.info(f"📄 Chargement depuis JSON: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        logger.info(f"✅ {len(chunks)} chunks chargés")
        
        # Calculer les tokens économisés
        if self.embedding_only_mode:
            tokens_per_chunk = 1500
            self.stats['tokens_saved'] = len(chunks) * tokens_per_chunk
            logger.info(f"💰 Mode économique: ~{self.stats['tokens_saved']:,} tokens économisés")
        
        # Enrichir si nécessaire
        enriched_chunks = self._enrich_chunks(chunks)
        
        # Créer le store
        if clear_existing:
            store_path = self.persist_directory
            if store_path.exists():
                import shutil
                shutil.rmtree(store_path)
        
        # IMPORTANT : Forcer analyze_on_index=False en mode embedding only
        vectorstore = create_intelligent_store(
            enriched_chunks,
            persist_directory=self.persist_directory,
            analyze_on_index=False if self.embedding_only_mode else self.analyze_chunks  # CHANGÉ
        )
        
        return vectorstore
    
    def _enrich_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Enrichit les chunks avec métadonnées supplémentaires
        """
        logger.info("🔧 Enrichissement des métadonnées...")
        
        enriched = []
        doc_structure = {}
        
        for i, chunk in enumerate(chunks):
            # Tracker la structure du document
            doc_name = chunk['metadata'].get('source', 'unknown')
            if doc_name not in doc_structure:
                doc_structure[doc_name] = {
                    'sections': set(),
                    'pages': set(),
                    'chunk_count': 0
                }
            
            # Enrichir les métadonnées
            chunk['metadata']['chunk_index'] = i
            chunk['metadata']['total_chunks'] = len(chunks)
            
            # Ajouter position relative
            chunk['metadata']['position_relative'] = i / len(chunks)
            if i < len(chunks) * 0.2:
                chunk['metadata']['document_zone'] = 'introduction'
            elif i < len(chunks) * 0.8:
                chunk['metadata']['document_zone'] = 'body'
            else:
                chunk['metadata']['document_zone'] = 'conclusion'
            
            # Tracker structure
            if section := chunk['metadata'].get('section'):
                doc_structure[doc_name]['sections'].add(section)
            if page := chunk['metadata'].get('page'):
                doc_structure[doc_name]['pages'].add(page)
            doc_structure[doc_name]['chunk_count'] += 1
            
            enriched.append(chunk)
        
        # Afficher structure découverte
        logger.info("📊 Structure des documents:")
        for doc_name, structure in doc_structure.items():
            logger.info(f"   {doc_name}:")
            logger.info(f"     - Sections: {len(structure['sections'])}")
            logger.info(f"     - Pages: {len(structure['pages'])}")
            logger.info(f"     - Chunks: {structure['chunk_count']}")
        
        return enriched
    
    def _validate_vectorstore(
        self, 
        vectorstore: IntelligentVectorStore
    ) -> Dict[str, Any]:
        """
        Valide le vectorstore avec des requêtes de test
        """
        logger.info("🧪 Validation du vectorstore...")
        
        test_queries = [
            "What is the submission deadline?",
            "What are the payment terms and conditions?",
            "Technical requirements for the system",
            "Insurance and liability clauses",
            "Transport volumes and logistics"
        ]
        
        results = {}
        for query in test_queries:
            try:
                search_results = vectorstore.intelligent_search(
                    query, 
                    k=3,
                    refine_with_llm=False  # TOUJOURS False pour validation rapide
                )
                results[query] = {
                    'found': len(search_results),
                    'top_score': search_results[0]['score'] if search_results else 0
                }
                logger.info(f"   ✓ '{query[:30]}...': {len(search_results)} résultats")
            except Exception as e:
                results[query] = {'found': 0, 'error': str(e)}
                logger.warning(f"   ✗ '{query[:30]}...': Erreur - {e}")
        
        return results
    
    def _save_chunks(self, chunks: List[Dict]):
        """
        Sauvegarde les chunks extraits
        """
        output_file = self.persist_directory / "extracted_chunks.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Chunks sauvegardés: {output_file}")
    
    def _generate_report(
        self,
        chunks: List[Dict],
        vectorstore: IntelligentVectorStore,
        validation_results: Dict
    ) -> Dict[str, Any]:
        """
        Génère un rapport détaillé de vectorisation
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'EMBEDDINGS_ONLY' if self.embedding_only_mode else 'FULL_ANALYSIS',
            'configuration': {
                'chunk_size': self.chunk_size,
                'analyze_chunks': self.analyze_chunks,
                'embedding_only_mode': self.embedding_only_mode,
                'persist_directory': str(self.persist_directory)
            },
            'statistics': self.stats,
            'chunks_analysis': {
                'total': len(chunks),
                'by_type': {},
                'by_document': {},
                'average_size': sum(c['metadata'].get('char_count', 0) for c in chunks) / len(chunks)
            },
            'vectorstore_stats': vectorstore.get_stats(),
            'validation': validation_results,
            'cost_estimate': {
                'embeddings_cost': len(chunks) * 0.00002,  # Estimation
                'llm_analysis_cost': 0 if self.embedding_only_mode else len(chunks) * 0.03,
                'total_cost': len(chunks) * (0.00002 if self.embedding_only_mode else 0.03002)
            }
        }
        
        # Analyser distribution des chunks
        for chunk in chunks:
            chunk_type = chunk.get('type', 'unknown')
            doc_name = chunk['metadata'].get('source', 'unknown')
            
            report['chunks_analysis']['by_type'][chunk_type] = \
                report['chunks_analysis']['by_type'].get(chunk_type, 0) + 1
            
            report['chunks_analysis']['by_document'][doc_name] = \
                report['chunks_analysis']['by_document'].get(doc_name, 0) + 1
        
        return report
    
    def _save_report(self, report: Dict):
        """
        Sauvegarde le rapport de vectorisation
        """
        report_file = self.persist_directory / f"vectorization_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Rapport sauvegardé: {report_file}")


def quick_test(vectorstore: IntelligentVectorStore):
    """
    Test rapide du vectorstore (SANS analyse LLM coûteuse)
    """
    print("\n" + "="*60)
    print("TEST RAPIDE DU VECTORSTORE (MODE ÉCONOMIQUE)")
    print("="*60)
    
    # Requêtes de test
    test_cases = [
        {
            'query': "deadline for submission",
            'expected_concepts': ['deadline', 'timeline', 'submission']
        },
        {
            'query': "payment terms and pricing",
            'expected_concepts': ['payment', 'pricing', 'financial']
        },
        {
            'query': "technical specifications",
            'expected_concepts': ['technical', 'requirements', 'specifications']
        }
    ]
    
    for test in test_cases:
        print(f"\n🔍 Test: {test['query']}")
        print("-" * 40)
        
        results = vectorstore.intelligent_search(
            test['query'],
            k=2,
            refine_with_llm=False  # CHANGÉ : False pour économiser
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Résultat {i}:")
            print(f"   Score: {result.get('score', 0):.2f}")
            if 'source' in result:
                print(f"   Source: {result['source'].get('breadcrumb', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 0)}%")
            print(f"   Extrait: {result.get('text', '')[:150]}...")


def main():
    """
    Point d'entrée principal
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Vectorisation Intelligente (Mode Économique par défaut)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        required=True,
        help="Dossier de documents ou fichier JSON de chunks"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/intelligent_store",
        help="Répertoire de sortie pour l'index"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=2000,
        help="Taille des chunks (défaut: 2000)"
    )
    parser.add_argument(
        "--enable-llm-analysis",  # CHANGÉ : Nouveau flag pour ACTIVER l'analyse
        action="store_true",
        help="Activer l'analyse LLM (coûteux - désactivé par défaut)"
    )
    parser.add_argument(
        "--no-analysis",  # Gardé pour compatibilité
        action="store_true",
        help="[DEPRECATED] Utilisez --enable-llm-analysis pour activer l'analyse"
    )
    parser.add_argument(
        "--clear", 
        action="store_true",
        help="Supprimer l'index existant"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Lancer les tests après vectorisation"
    )
    
    args = parser.parse_args()
    
    # Vérifier OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY non trouvée dans les variables d'environnement")
        print("   Définissez-la avec: export OPENAI_API_KEY='votre-clé'")
        return
    
    source_path = Path(args.source)
    
    if not source_path.exists():
        print(f"❌ Chemin introuvable: {source_path}")
        return
    
    # Déterminer le mode
    embedding_only = not args.enable_llm_analysis  # Par défaut: embedding only
    
    if embedding_only:
        print("\n💰 MODE ÉCONOMIQUE ACTIVÉ")
        print("   - Utilisation des embeddings uniquement")
        print("   - Pas d'analyse LLM (économie ~99% des tokens)")
        print("   - Pour activer l'analyse LLM: ajoutez --enable-llm-analysis")
    else:
        print("\n⚠️ MODE ANALYSE COMPLÈTE ACTIVÉ")
        print("   - Analyse LLM pour chaque chunk")
        print("   - Coût estimé: ~$0.03 par chunk")
        response = input("   Continuer? (y/n): ")
        if response.lower() != 'y':
            print("Annulé.")
            return
    
    try:
        # Créer le vectorizer
        vectorizer = IntelligentVectorizer(
            persist_directory=Path(args.output),
            chunk_size=args.chunk_size,
            analyze_chunks=not embedding_only,  # Analyse seulement si explicitement demandé
            embedding_only_mode=embedding_only  # Mode embedding only par défaut
        )
        
        # Lancer la vectorisation
        if source_path.suffix == '.json':
            vectorstore = vectorizer.vectorize_from_json(
                json_file=source_path,
                clear_existing=args.clear
            )
        elif source_path.is_dir():
            vectorstore = vectorizer.vectorize_from_folder(
                folder_path=source_path,
                clear_existing=args.clear
            )
        else:
            raise ValueError(f"Source invalide (doit être dossier ou .json): {source_path}")
        
        # Tests si demandés
        if args.test:
            quick_test(vectorstore)
        
        print("\n✅ VECTORISATION TERMINÉE AVEC SUCCÈS!")
        print(f"   Index sauvegardé dans: {args.output}")
        
        if embedding_only:
            print(f"\n💰 ÉCONOMIES RÉALISÉES:")
            print(f"   - Tokens économisés: ~{vectorizer.stats.get('tokens_saved', 0):,}")
            print(f"   - Coût évité: ~${vectorizer.stats.get('tokens_saved', 0) * 0.00002:.2f}")
            print(f"   - Temps gagné: ~{vectorizer.stats.get('tokens_saved', 0) // 100} secondes")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())