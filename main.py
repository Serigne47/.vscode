# main.py
"""
Point d'entrée principal pour l'analyse d'appels d'offres
"""
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).resolve().parent))

from aoagentc.utils.arch0.enhanced_chunker1 import EnhancedChunker
from aoagentc.utils.arch0.enhanced_retrieval1 import EnhancedAORetriever
from aoagentc.utils.arch0.vectorstore1 import get_vectorstore
from utils.report_generator import AOReportGenerator
from agent.orchestrator import AOExtractionOrchestrator
from langchain_core.documents import Document

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ao_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - MODIFIEZ ICI VOS CHEMINS ET PARAMÈTRES
# ============================================================================

class Config:
    """Configuration centralisée de l'application"""
    
    # Chemin vers vos documents d'appel d'offres
    DOCUMENTS_PATH = Path(r"C:\Users\serigne.faye\OneDrive - MSC\Bureau\Dossier de travail\MISSIONS\B PILOTAGE PROJETS INNO\2.6 PoC Power Platform\9 Agent Analyse AO\Z DOCUMENTATION\Reckitt Tender documentation 1")
    
    # Chemin pour la base de données vectorielle
    VECTORSTORE_PATH = Path("data/chroma_db")
    
    # Chemin pour les rapports générés
    REPORTS_PATH = Path("reports")
    
    # Configuration OpenAI
    OPENAI_MODEL = "gpt-4o-mini"  # ou "gpt-3.5-turbo"
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Configuration chunking
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    
    # Configuration analyse
    MAX_PARALLEL_AGENTS = 6  # Nombre d'agents à exécuter en parallèle
    ENABLE_CACHE = True  # Active le cache pour éviter de retraiter les mêmes documents
    
    # Export des résultats
    EXPORT_FORMATS = ["json", "excel", "word"]  # Formats d'export souhaités
    
    @classmethod
    def validate(cls):
        """Valide la configuration"""
        if not cls.DOCUMENTS_PATH.exists():
            raise ValueError(f"Le chemin des documents n'existe pas: {cls.DOCUMENTS_PATH}")
        
        # Créer les dossiers s'ils n'existent pas
        cls.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        cls.REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Vérifier la clé API OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("⚠️ OPENAI_API_KEY non trouvée dans les variables d'environnement")
            logger.info("Définissez-la avec: export OPENAI_API_KEY='your-key-here'")
        
        return True

# ============================================================================
# FONCTIONS PRINCIPALES
# ============================================================================

async def analyze_tender_documents(
    documents_path: Path = None,
    force_reindex: bool = False
) -> dict:
    """
    Analyse complète des documents d'appel d'offres
    
    Args:
        documents_path: Chemin vers les documents (utilise Config par défaut)
        force_reindex: Force la réindexation même si les documents sont déjà dans la base
    
    Returns:
        Rapport d'analyse complet
    """
    try:
        # Utiliser le chemin de Config si non spécifié
        if documents_path is None:
            documents_path = Config.DOCUMENTS_PATH
        
        logger.info("="*80)
        logger.info("🚀 DÉMARRAGE DE L'ANALYSE D'APPEL D'OFFRES")
        logger.info(f"📁 Dossier source: {documents_path}")
        logger.info("="*80)
        
        # Phase 1: Chargement et chunking des documents
        logger.info("\n📄 PHASE 1: Chargement des documents...")
        chunker = EnhancedChunker()
        documents = []
        
        # Parcourir tous les fichiers du dossier
        supported_extensions = {'.pdf', '.docx', '.doc', '.xlsx', '.xls'}
        files_processed = 0
        
        for file_path in documents_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                logger.info(f"  → Traitement de: {file_path.name}")
                try:
                    # Extraction avec structure
                    structured_elements = chunker.extract_with_structure(file_path)
                    
                    # Conversion en Documents LangChain
                    for element in structured_elements:
                        doc = Document(
                            page_content=element.get('text', ''),
                            metadata={
                                'source': file_path.name,
                                'type': element.get('type'),
                                'section': element.get('section'),
                                'page': element.get('page'),
                                'file_path': str(file_path)
                            }
                        )
                        documents.append(doc)
                    
                    files_processed += 1
                    logger.info(f"    ✓ {len(structured_elements)} éléments extraits")
                    
                except Exception as e:
                    logger.error(f"    ✗ Erreur: {e}")
        
        if files_processed == 0:
            raise ValueError(f"Aucun document trouvé dans {documents_path}")
        
        logger.info(f"\n✅ {files_processed} fichiers traités, {len(documents)} chunks créés")
        
        # Phase 2: Indexation vectorielle (si nécessaire)
        logger.info("\n🗄️ PHASE 2: Indexation vectorielle...")
        vectorstore = get_vectorstore(Config.VECTORSTORE_PATH)
        
        if force_reindex or not Config.VECTORSTORE_PATH.exists():
            logger.info("  → Indexation des documents...")
            vectorstore.add_documents(documents)
            logger.info("  ✓ Indexation terminée")
        else:
            logger.info("  → Utilisation de l'index existant")
        
        # Phase 3: Analyse multi-agent
        logger.info("\n🤖 PHASE 3: Analyse multi-agent...")
        orchestrator = AOExtractionOrchestrator(
            vectorstore=vectorstore,
            config={
                'model': Config.OPENAI_MODEL,
                'max_parallel': Config.MAX_PARALLEL_AGENTS
            }
        )
        
        # Lancer l'analyse
        analysis_result = await orchestrator.analyze_ao(documents)
        
        if analysis_result['status'] == 'error':
            logger.error(f"❌ Erreur lors de l'analyse: {analysis_result['error']}")
            return analysis_result
        
        # Phase 4: Génération des rapports
        logger.info("\n📊 PHASE 4: Génération des rapports...")
        report_generator = AOReportGenerator()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export JSON
        if "json" in Config.EXPORT_FORMATS:
            json_path = Config.REPORTS_PATH / f"ao_analysis_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result['report'], f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"  ✓ Rapport JSON: {json_path}")
        
        # Export Excel (si implementé)
        if "excel" in Config.EXPORT_FORMATS:
            try:
                excel_path = Config.REPORTS_PATH / f"ao_analysis_{timestamp}.xlsx"
                report_generator.export_to_excel(analysis_result['report'], excel_path)
                logger.info(f"  ✓ Rapport Excel: {excel_path}")
            except Exception as e:
                logger.warning(f"  ⚠️ Export Excel non disponible: {e}")
        
        # Affichage du résumé
        logger.info("\n" + "="*80)
        logger.info("📋 RÉSUMÉ DE L'ANALYSE")
        logger.info("="*80)
        
        summary = analysis_result['report']['executive_summary']
        logger.info(f"Client: {summary['client']}")
        logger.info(f"Référence AO: {summary['reference']}")
        logger.info(f"Date limite: {summary['date_limite']}")
        logger.info(f"Volumes estimés: {summary['volume_estime']}")
        logger.info(f"Durée contrat: {summary.get('duree_contrat', 'N/A')} mois")
        logger.info(f"Complexité: {summary['complexite']}")
        logger.info(f"Risque global: {summary['risque_global']}")
        logger.info(f"Complétude données: {summary['completude_donnees']}")
        
        go_no_go = summary['go_no_go_recommendation']
        logger.info(f"\n🎯 RECOMMANDATION: {go_no_go['decision']} (Confiance: {go_no_go['confidence']})")
        
        # Actions urgentes
        if analysis_result['report'].get('actions_urgentes'):
            logger.info("\n⚡ ACTIONS URGENTES:")
            for action in analysis_result['report']['actions_urgentes'][:3]:
                logger.info(f"  • {action['action']} ({action['deadline']})")
        
        # Données manquantes critiques
        if analysis_result['report'].get('donnees_manquantes'):
            logger.info("\n⚠️ DONNÉES MANQUANTES CRITIQUES:")
            for missing in analysis_result['report']['donnees_manquantes'][:5]:
                logger.info(f"  • {missing}")
        
        # Statistiques d'exécution
        logger.info("\n📊 STATISTIQUES D'EXÉCUTION:")
        metadata = analysis_result.get('metadata', {})
        if metadata.get('processing_time'):
            logger.info(f"  • Temps de traitement: {metadata['processing_time']:.2f} secondes")
        logger.info(f"  • Documents traités: {metadata.get('documents_processed', 0)}")
        logger.info(f"  • Chunks analysés: {metadata.get('chunks_analyzed', 0)}")
        logger.info(f"  • Agents exécutés: {metadata.get('agents_run', 0)}")
        logger.info(f"  • Confiance globale: {analysis_result.get('confidence_global', 0):.1%}")
        
        logger.info("\n✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
        logger.info("="*80)
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'report': {}
        }

def analyze_specific_folder(folder_path: str):
    """
    Fonction helper pour analyser un dossier spécifique
    
    Args:
        folder_path: Chemin vers le dossier à analyser
    """
    path = Path(folder_path)
    if not path.exists():
        logger.error(f"Le dossier n'existe pas: {path}")
        return
    
    # Lancer l'analyse asynchrone
    result = asyncio.run(analyze_tender_documents(path))
    return result

# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    """
    Point d'entrée principal du programme
    Vous pouvez modifier le chemin directement dans Config.DOCUMENTS_PATH
    ou passer un argument en ligne de commande
    """
    
    # Valider la configuration
    try:
        Config.validate()
    except Exception as e:
        logger.error(f"Erreur de configuration: {e}")
        sys.exit(1)
    
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) > 1:
        # Si un chemin est passé en argument, l'utiliser
        custom_path = Path(sys.argv[1])
        if custom_path.exists():
            logger.info(f"Utilisation du chemin personnalisé: {custom_path}")
            result = asyncio.run(analyze_tender_documents(custom_path))
        else:
            logger.error(f"Le chemin spécifié n'existe pas: {custom_path}")
            sys.exit(1)
    else:
        # Utiliser le chemin par défaut de Config
        logger.info(f"Utilisation du chemin par défaut: {Config.DOCUMENTS_PATH}")
        result = asyncio.run(analyze_tender_documents())
    
    # Afficher le chemin du rapport généré
    if result['status'] == 'success':
        logger.info(f"\n📁 Les rapports ont été sauvegardés dans: {Config.REPORTS_PATH}")
        logger.info("Consultez les fichiers générés pour l'analyse détaillée.")
    else:
        logger.error("L'analyse a échoué. Consultez les logs pour plus de détails.")
        sys.exit(1)