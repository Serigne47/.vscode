# utils/vectorstore.py
"""
Gestion de la base vectorielle ChromaDB pour le stockage et la recherche
des documents d'appels d'offres avec agents spécialisés
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import json
import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class AOVectorStore:
    """
    Gestionnaire de base vectorielle optimisé pour les appels d'offres
    avec support pour agents spécialisés par domaine
    """
    
    def __init__(
        self,
        persist_directory: Path = Path("data/chroma_db"),
        collection_name: str = "ao_documents",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialise le vectorstore
        
        Args:
            persist_directory: Chemin de persistance de la base
            collection_name: Nom de la collection
            embedding_model: Modèle d'embedding OpenAI
        """
        # Assignation des paramètres
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Créer le répertoire si nécessaire
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Vérifier la clé OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY non trouvée dans les variables d'environnement")
        
        # Initialiser les embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            chunk_size=500  # Pour éviter les timeouts sur gros volumes
        )
        
        # Initialiser le vectorstore
        self.vectorstore = self._initialize_vectorstore()
        
        # Cache pour éviter les duplications
        self.document_hashes = self._load_document_hashes()
        
        logger.info(f"✅ VectorStore initialisé: {self.persist_directory}/{collection_name}")
    
    def _initialize_vectorstore(self) -> Chroma:
        """
        Initialise ou charge le vectorstore existant
        """
        try:
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            # Vérifier si la collection existe
            try:
                count = vectorstore._collection.count()
                logger.info(f"Collection '{self.collection_name}' chargée: {count} documents")
            except Exception:
                logger.info(f"Création de la nouvelle collection '{self.collection_name}'")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du vectorstore: {e}")
            raise
    
    def _load_document_hashes(self) -> set:
        """
        Charge les hashes des documents déjà indexés
        """
        hash_file = self.persist_directory / f"{self.collection_name}_hashes.json"
        if hash_file.exists():
            try:
                with open(hash_file, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Impossible de charger les hashes: {e}")
        return set()
    
    def _save_document_hashes(self):
        """
        Sauvegarde les hashes des documents indexés
        """
        hash_file = self.persist_directory / f"{self.collection_name}_hashes.json"
        try:
            with open(hash_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.document_hashes), f, indent=2)
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder les hashes: {e}")
    
    def _compute_document_hash(self, document: Document) -> str:
        """
        Calcule un hash unique pour un document
        """
        content = f"{document.page_content}{json.dumps(document.metadata, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _compute_element_hash(self, element: Dict) -> str:
        """
        Calcule un hash unique pour un élément extrait
        """
        content = f"{element.get('text', '')}{json.dumps(element.get('metadata', {}), sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_financial_priority(self, element: Dict) -> float:
        """
        Calcule la priorité pour l'agent financier
        """
        score = 0.0
        text = element.get('text', '').lower()
        element_type = element.get('type', '')
        
        # Bonus selon le type
        if element_type == 'financial':
            score += 1.0
        elif element.get('metadata', {}).get('contains_amounts'):
            score += 0.8
        
        # Bonus selon les mots-clés
        financial_keywords = [
            'price', 'prix', 'tariff', 'tarif', 'cost', 'coût', 'budget', 
            'eur', 'usd', 'payment', 'paiement', 'invoicing', 'facturation',
            'warranty', 'garantie', 'advance', 'avance', 'currency', 'monnaie',
            'price revision', 'révision prix', 'fuel surcharge', 'baf'
        ]
        
        keyword_matches = sum(1 for keyword in financial_keywords if keyword in text)
        score += keyword_matches * 0.1
        
        return min(score, 1.0)
    
    def _calculate_legal_priority(self, element: Dict) -> float:
        """
        Calcule la priorité pour l'agent légal
        """
        score = 0.0
        text = element.get('text', '').lower()
        element_type = element.get('type', '')
        
        if element_type == 'legal':
            score += 1.0
        
        legal_keywords = [
            'responsibility', 'responsabilité', 'liability', 'insurance', 'assurance',
            'clause', 'article', 'penalty', 'pénalité', 'force majeure',
            'jurisdiction', 'juridiction', 'applicable law', 'loi applicable',
            'confidentiality', 'confidentialité', 'gdpr', 'compliance', 'conformité'
        ]
        
        keyword_matches = sum(1 for keyword in legal_keywords if keyword in text)
        score += keyword_matches * 0.15
        
        return min(score, 1.0)
    
    def _calculate_operational_priority(self, element: Dict) -> float:
        """
        Calcule la priorité pour l'agent opérationnel
        """
        score = 0.0
        text = element.get('text', '').lower()
        element_type = element.get('type', '')
        
        if element_type in ['volume', 'table']:
            score += 0.8
        
        operational_keywords = [
            'transport', 'maritime', 'air', 'aérien', 'road', 'route', 'rail',
            'multimodal', 'customs clearance', 'dédouanement', 'warehousing',
            'entreposage', 'delivery', 'livraison', 'sla', 'kpi', 'temperature',
            'température', 'gps', 'certification', 'teu', 'tonnage', 'container'
        ]
        
        keyword_matches = sum(1 for keyword in operational_keywords if keyword in text)
        score += keyword_matches * 0.1
        
        return min(score, 1.0)
    
    def _calculate_timeline_priority(self, element: Dict) -> float:
        """
        Calcule la priorité pour les éléments temporels
        """
        score = 0.0
        text = element.get('text', '').lower()
        element_type = element.get('type', '')
        
        if element_type == 'timeline':
            score += 1.0
        elif element.get('metadata', {}).get('contains_dates'):
            score += 0.8
        
        timeline_keywords = [
            'deadline', 'échéance', 'delay', 'délai', 'date', 'planning',
            'turn', 'tour', 'response', 'réponse', 'submission', 'soumission',
            'timeline', 'schedule', 'planning'
        ]
        
        keyword_matches = sum(1 for keyword in timeline_keywords if keyword in text)
        score += keyword_matches * 0.2
        
        return min(score, 1.0)
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nettoie les métadonnées pour ChromaDB (supprime listes, None, etc.)
        """
        clean_meta = {}
        
        for key, value in metadata.items():
            # ChromaDB accepte seulement: str, int, float, bool
            if value is None:
                continue
            elif isinstance(value, (list, dict)):
                # Convertir listes et dicts en string
                if value:  # Si non vide
                    clean_meta[key] = str(value)[:500]  # Limiter la taille
            elif isinstance(value, (str, int, float, bool)):
                clean_meta[key] = value
            else:
                # Convertir autres types en string
                clean_meta[key] = str(value)
        
        return clean_meta

    def add_extracted_elements(
        self,
        extracted_elements: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> int:
        """
        Convertit et ajoute les éléments extraits par Enhanced Chunker
        
        Args:
            extracted_elements: Résultat de EnhancedChunker.process_folder()
            batch_size: Taille des batches
            
        Returns:
            Nombre de documents ajoutés
        """
        documents = []
        
        for element in extracted_elements:
            # Préparer les métadonnées
            raw_metadata = {
                # Métadonnées de base
                "source": element.get('metadata', {}).get('source', 'unknown'),
                "type": element.get('type', 'unknown'),
                
                # Métadonnées de structure
                "section": element.get('metadata', {}).get('section', ''),
                "subsection": element.get('metadata', {}).get('subsection', ''),
                "page": element.get('metadata', {}).get('page'),
                "section_num": element.get('metadata', {}).get('section_num'),
                "sheet": element.get('metadata', {}).get('sheet', ''),
                
                # Métadonnées de contenu détectées
                "contains_volumes": element.get('metadata', {}).get('contains_volumes', False),
                "contains_dates": element.get('metadata', {}).get('contains_dates', False),
                "contains_amounts": element.get('metadata', {}).get('contains_amounts', False),
                "is_table": element.get('metadata', {}).get('is_table', False),
                
                # Métadonnées pour Excel
                "rows": element.get('metadata', {}).get('rows', 0),
                "has_columns": bool(element.get('metadata', {}).get('columns')),
                
                # Ajout des données de tableau structuré si disponible
                "has_parsed_table": bool(element.get('parsed_table')),
                
                # Calcul de priorité pour les agents spécialisés
                "priority_financial": self._calculate_financial_priority(element),
                "priority_legal": self._calculate_legal_priority(element),
                "priority_operational": self._calculate_operational_priority(element),
                "priority_timeline": self._calculate_timeline_priority(element),
                
                # Hash unique pour dédoublonnage
                "element_hash": self._compute_element_hash(element)
            }
            
            # Ajouter données de tableau en métadonnées si disponible
            if element.get('parsed_table') and isinstance(element['parsed_table'], dict):
                raw_metadata["table_data"] = json.dumps(element['parsed_table'])[:500]  # Limiter taille
            
            # Nettoyer les métadonnées
            clean_metadata = self._clean_metadata(raw_metadata)
            
            # Créer un Document LangChain depuis l'élément extrait
            doc = Document(
                page_content=element.get('text', ''),
                metadata=clean_metadata
            )
            
            documents.append(doc)
        
        # Ajouter au vectorstore
        return self.add_documents(documents, batch_size)
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 50,
        skip_duplicates: bool = True
    ) -> int:
        """
        Ajoute des documents au vectorstore avec gestion des duplications
        
        Args:
            documents: Liste des documents à ajouter
            batch_size: Taille des batches pour l'insertion
            skip_duplicates: Éviter les doublons
            
        Returns:
            Nombre de documents ajoutés
        """
        if not documents:
            logger.warning("Aucun document à ajouter")
            return 0
        
        documents_to_add = []
        skipped = 0
        
        # Filtrer les duplications si demandé
        if skip_duplicates:
            for doc in documents:
                doc_hash = self._compute_document_hash(doc)
                if doc_hash not in self.document_hashes:
                    documents_to_add.append(doc)
                    self.document_hashes.add(doc_hash)
                else:
                    skipped += 1
            
            if skipped > 0:
                logger.info(f"⏭️ {skipped} documents ignorés (déjà indexés)")
        else:
            documents_to_add = documents
        
        if not documents_to_add:
            logger.info("Tous les documents sont déjà indexés")
            return 0
        
        # Ajouter par batches pour éviter les timeouts
        total_added = 0
        for i in range(0, len(documents_to_add), batch_size):
            batch = documents_to_add[i:i + batch_size]
            try:
                self.vectorstore.add_documents(batch)
                total_added += len(batch)
                logger.info(f"  → Batch {i//batch_size + 1}: {len(batch)} documents ajoutés")
            except Exception as e:
                logger.error(f"Erreur lors de l'ajout du batch {i//batch_size + 1}: {e}")
                continue
        
        # Persister les changements
        try:
            self.vectorstore.persist()
        except AttributeError:
            # ChromaDB nouvelle version n'a plus persist()
            logger.debug("Persist() non disponible - sauvegarde automatique")
        
        self._save_document_hashes()
        
        logger.info(f"✅ {total_added} documents ajoutés au vectorstore")
        return total_added
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None,
        score_threshold: float = 0.0  # Réduire le seuil par défaut
    ) -> List[Document]:
        """
        Recherche par similarité avec filtrage optionnel
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats
            filter: Filtres sur les métadonnées
            score_threshold: Seuil de similarité minimum
            
        Returns:
            Documents les plus similaires
        """
        try:
            # Recherche avec scores
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k * 2,  # Over-fetch pour filtrage
                filter=filter
            )
            
            # Filtrer par score threshold (ChromaDB utilise distance, plus faible = meilleur)
            filtered_results = [
                doc for doc, score in results_with_scores
                if score <= (1.0 - score_threshold)  # Inverser pour distance
            ][:k]
            
            logger.info(f"🔍 Recherche '{query[:50]}...': {len(filtered_results)} résultats")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            return []
    
    def search_by_domain(
        self,
        query: str,
        domain: str,
        k: int = 5,
        min_priority: float = 0.3
    ) -> List[Document]:
        """
        Recherche spécialisée par domaine (financial, legal, operational, timeline)
        
        Args:
            query: Requête de recherche
            domain: Domaine cible ('financial', 'legal', 'operational', 'timeline')
            k: Nombre de résultats
            min_priority: Score minimum de priorité pour le domaine
            
        Returns:
            Documents pertinents pour le domaine
        """
        try:
            # Recherche standard d'abord (filtre ChromaDB complexe pas toujours fiable)
            results = self.similarity_search(query=query, k=k * 3)
            
            # Filtrer par priorité du domaine
            domain_results = []
            for doc in results:
                priority = doc.metadata.get(f"priority_{domain}", 0)
                if priority >= min_priority:
                    domain_results.append((doc, priority))
            
            # Trier par priorité décroissante et retourner top k
            domain_results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in domain_results[:k]]
            
        except Exception as e:
            logger.error(f"Erreur recherche par domaine {domain}: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict] = None,
        search_type: str = "mmr"
    ) -> List[Document]:
        """
        Recherche hybride avec MMR (Maximum Marginal Relevance)
        
        Args:
            query: Requête de recherche
            k: Nombre de résultats
            filter: Filtres sur les métadonnées
            search_type: Type de recherche ("similarity" ou "mmr")
            
        Returns:
            Documents pertinents avec diversité
        """
        try:
            if search_type == "mmr":
                # MMR pour diversité des résultats
                return self.vectorstore.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=k * 3,  # Fetch plus pour meilleure diversité
                    filter=filter
                )
            else:
                return self.similarity_search(query, k, filter)
        except Exception as e:
            logger.error(f"Erreur recherche hybride: {e}")
            return self.similarity_search(query, k, filter)
    
    def get_documents_by_type(self, doc_types: List[str], limit: int = 100) -> List[Document]:
        """
        Récupère les documents par type
        
        Args:
            doc_types: Liste des types ('table', 'financial', 'legal', etc.)
            limit: Limite de résultats
            
        Returns:
            Documents du type demandé
        """
        try:
            # Recherche générale puis filtrage local (plus fiable)
            all_docs = self.similarity_search("", k=limit * 2)
            
            filtered_docs = [
                doc for doc in all_docs
                if doc.metadata.get('type', '') in doc_types
            ]
            
            return filtered_docs[:limit]
            
        except Exception as e:
            logger.warning(f"Erreur get_documents_by_type: {e}")
            return []
    
    def delete_collection(self):
        """
        Supprime la collection actuelle
        """
        try:
            self.vectorstore._client.delete_collection(self.collection_name)
            self.document_hashes.clear()
            self._save_document_hashes()
            logger.info(f"❌ Collection '{self.collection_name}' supprimée")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de la collection
        """
        try:
            count = self.vectorstore._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "unique_documents": len(self.document_hashes),
                "persist_directory": str(self.persist_directory),
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "unique_documents": len(self.document_hashes),
                "persist_directory": str(self.persist_directory),
                "embedding_model": self.embedding_model,
                "error": str(e)
            }
    
    def as_retriever(self, **kwargs):
        """
        Retourne un retriever configuré
        """
        default_kwargs = {
            "search_type": "similarity",
            "search_kwargs": {"k": 5}
        }
        default_kwargs.update(kwargs)
        return self.vectorstore.as_retriever(**default_kwargs)


# Fonctions utilitaires hors classe

def get_vectorstore(
    persist_directory: Path = Path("data/chroma_db"),
    collection_name: str = "ao_documents",
    embedding_model: str = "text-embedding-3-small"
) -> AOVectorStore:
    """
    Factory function pour obtenir une instance de vectorstore
    
    Args:
        persist_directory: Chemin de persistance
        collection_name: Nom de la collection
        embedding_model: Modèle d'embedding
        
    Returns:
        Instance configurée de AOVectorStore
    """
    return AOVectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model
    )


def index_extracted_elements(
    extracted_elements: List[Dict[str, Any]],
    vectorstore: AOVectorStore = None,
    collection_name: str = "ao_documents"
) -> AOVectorStore:
    """
    Index les éléments extraits par Enhanced Chunker
    
    Args:
        extracted_elements: Résultats de Enhanced Chunker
        vectorstore: Instance existante (optionnel)
        collection_name: Nom de la collection
        
    Returns:
        Instance du vectorstore avec les documents indexés
    """
    if vectorstore is None:
        vectorstore = get_vectorstore(collection_name=collection_name)
    
    logger.info(f"🔄 Indexation de {len(extracted_elements)} éléments...")
    
    added_count = vectorstore.add_extracted_elements(extracted_elements)
    
    stats = vectorstore.get_collection_stats()
    logger.info(f"✅ Indexation terminée:")
    logger.info(f"   - {added_count} nouveaux documents")
    logger.info(f"   - {stats.get('document_count', 0)} documents au total")
    
    return vectorstore