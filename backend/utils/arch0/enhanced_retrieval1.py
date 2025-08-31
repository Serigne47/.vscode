# utils/enhanced_retrieval.py
"""
Système de retrieval amélioré avec re-ranking et recherche hybride
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("⚠️ sentence-transformers non installé. Re-ranking désactivé.")

logger = logging.getLogger(__name__)

class EnhancedAORetriever:
    """
    Système de retrieval hybride avec re-ranking pour précision maximale
    """
    
    def __init__(
        self,
        vectorstore,
        use_reranking: bool = True,
        use_compression: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialise le retriever amélioré
        
        Args:
            vectorstore: Instance du vectorstore
            use_reranking: Activer le re-ranking avec cross-encoder
            use_compression: Activer la compression contextuelle
            rerank_model: Modèle de re-ranking à utiliser
        """
        self.vectorstore = vectorstore
        self.use_reranking = use_reranking and CROSS_ENCODER_AVAILABLE
        self.use_compression = use_compression
        
        # Initialiser le cross-encoder pour re-ranking
        if self.use_reranking:
            try:
                self.cross_encoder = CrossEncoder(rerank_model)
                logger.info(f"✅ Re-ranking activé avec {rerank_model}")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser le cross-encoder: {e}")
                self.use_reranking = False
        
        # LLM pour compression si nécessaire
        if self.use_compression:
            self.compressor_llm = ChatOpenAI(
                model="gpt-5-mini",
                temperature=0
            )
        
        # Cache pour les requêtes
        self.query_cache = {}
        
        logger.info("✅ EnhancedAORetriever initialisé")
    
    def retrieve_with_context(
        self,
        query: str,
        category: str = None,
        k: int = 5,
        fetch_k: int = 20
    ) -> List[Document]:
        """
        Retrieval avec enrichissement contextuel
        
        Args:
            query: Requête de recherche
            category: Catégorie pour expansion de requête
            k: Nombre de documents finaux
            fetch_k: Nombre de documents à récupérer avant re-ranking
            
        Returns:
            Documents les plus pertinents avec contexte enrichi
        """
        # Vérifier le cache
        cache_key = f"{query}_{category}_{k}"
        if cache_key in self.query_cache:
            logger.info(f"💾 Résultat depuis le cache pour: {query[:50]}...")
            return self.query_cache[cache_key]
        
        # Expansion de requête basée sur la catégorie
        expanded_query = self._expand_query(query, category)
        logger.info(f"🔍 Recherche: {expanded_query[:100]}...")
        
        # Recherche hybride (vectorielle + keyword si disponible)
        candidates = self._hybrid_search(expanded_query, fetch_k)
        
        if not candidates:
            logger.warning("Aucun document trouvé")
            return []
        
        # Re-ranking si activé
        if self.use_reranking and len(candidates) > k:
            reranked = self._rerank_documents(expanded_query, candidates, k)
        else:
            reranked = candidates[:k]
        
        # Enrichissement contextuel
        enriched = self._add_surrounding_context(reranked)
        
        # Compression si activée
        if self.use_compression:
            enriched = self._compress_documents(query, enriched)
        
        # Mettre en cache
        self.query_cache[cache_key] = enriched
        
        logger.info(f"✅ {len(enriched)} documents retournés")
        return enriched
    
    def _expand_query(self, query: str, category: Optional[str]) -> str:
        """
        Expansion de requête selon le domaine
        """
        if not category:
            return query
        
        expansions = {
            "identity": "client émetteur référence appel offre AO tender RFP marché contrat",
            "volume": "volume quantité TEU FEU tonnage tonnes palettes m3 mètres cubes containers prévision estimation",
            "financial": "prix tarif coût facturation paiement payment devise EUR USD garantie budget révision indexation",
            "legal": "clause responsabilité liability pénalité penalty résiliation termination assurance insurance juridique",
            "operational": "transport livraison delivery service entrepôt warehouse température GDP KPI SLA opération",
            "timeline": "date délai deadline échéance calendrier planning durée période temps jours mois"
        }
        
        expansion = expansions.get(category, "")
        if expansion:
            return f"{query} {expansion}"
        return query
    
    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """
        Recherche hybride combinant vectorielle et BM25
        """
        try:
            # Recherche vectorielle
            vector_results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                score_threshold=0.5
            )
            
            # Si on a assez de résultats vectoriels, on peut essayer BM25 en complément
            if len(vector_results) < k // 2:
                # Récupérer plus de documents pour BM25
                all_docs = self.vectorstore.similarity_search(
                    query="",  # Requête vide pour tout récupérer
                    k=k * 5
                )
                
                if all_docs:
                    # Créer un retriever BM25
                    bm25_retriever = BM25Retriever.from_documents(all_docs)
                    bm25_retriever.k = k // 2
                    
                    # Recherche BM25
                    bm25_results = bm25_retriever.get_relevant_documents(query)
                    
                    # Combiner les résultats (dédupliqués)
                    seen_content = {doc.page_content for doc in vector_results}
                    for doc in bm25_results:
                        if doc.page_content not in seen_content:
                            vector_results.append(doc)
                            seen_content.add(doc.page_content)
            
            return vector_results
            
        except Exception as e:
            logger.error(f"Erreur recherche hybride: {e}")
            # Fallback sur recherche simple
            return self.vectorstore.similarity_search(query, k=k)
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[Document],
        k: int
    ) -> List[Document]:
        """
        Re-ranking avec cross-encoder pour précision maximale
        """
        if not self.use_reranking or not documents:
            return documents[:k]
        
        try:
            # Préparer les paires (query, document)
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Calculer les scores avec le cross-encoder
            scores = self.cross_encoder.predict(pairs)
            
            # Trier par score décroissant
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Ajouter le score aux métadonnées
            reranked_docs = []
            for doc, score in doc_scores[:k]:
                doc.metadata['rerank_score'] = float(score)
                reranked_docs.append(doc)
            
            logger.info(f"🎯 Re-ranking: top score {doc_scores[0][1]:.3f}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Erreur re-ranking: {e}")
            return documents[:k]
    
    def _add_surrounding_context(self, documents: List[Document]) -> List[Document]:
        """
        Ajoute le contexte des chunks voisins
        """
        enriched_docs = []
        
        for doc in documents:
            # Copier le document
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy()
            )
            
            # Essayer de récupérer les chunks voisins si possible
            if 'source' in doc.metadata and 'section_num' in doc.metadata:
                try:
                    # Rechercher les chunks du même document/section
                    filter_dict = {
                        'source': doc.metadata['source']
                    }
                    
                    neighbors = self.vectorstore.similarity_search(
                        query=doc.page_content[:100],  # Début du chunk actuel
                        k=3,
                        filter=filter_dict
                    )
                    
                    # Ajouter le contexte aux métadonnées
                    if len(neighbors) > 1:
                        context_before = ""
                        context_after = ""
                        
                        for neighbor in neighbors:
                            if neighbor.page_content != doc.page_content:
                                # Déterminer si avant ou après
                                if neighbor.metadata.get('section_num', 0) < doc.metadata.get('section_num', 0):
                                    context_before = neighbor.page_content[-200:]  # Fin du chunk précédent
                                else:
                                    context_after = neighbor.page_content[:200]  # Début du chunk suivant
                        
                        enriched_doc.metadata['context_before'] = context_before
                        enriched_doc.metadata['context_after'] = context_after
                
                except Exception as e:
                    logger.debug(f"Impossible d'enrichir le contexte: {e}")
            
            enriched_docs.append(enriched_doc)
        
        return enriched_docs
    
    def _compress_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Compression contextuelle des documents
        """
        if not self.use_compression or not documents:
            return documents
        
        try:
            # Créer un compresseur
            compressor = LLMChainExtractor.from_llm(self.compressor_llm)
            
            # Créer un retriever avec compression
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=DummyRetriever(documents)  # Wrapper pour les docs
            )
            
            # Appliquer la compression
            compressed = compression_retriever.get_relevant_documents(query)
            
            logger.info(f"📦 Compression: {len(documents)} → {len(compressed)} documents")
            return compressed
            
        except Exception as e:
            logger.error(f"Erreur compression: {e}")
            return documents
    
    def multi_query_retrieve(
        self,
        queries: List[str],
        k_per_query: int = 3,
        deduplicate: bool = True
    ) -> List[Document]:
        """
        Retrieval avec plusieurs requêtes
        
        Args:
            queries: Liste de requêtes
            k_per_query: Documents par requête
            deduplicate: Dédupliquer les résultats
            
        Returns:
            Documents combinés
        """
        all_docs = []
        seen_content = set()
        
        for query in queries:
            docs = self.retrieve_with_context(query, k=k_per_query)
            
            for doc in docs:
                if deduplicate:
                    if doc.page_content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc.page_content)
                else:
                    all_docs.append(doc)
        
        logger.info(f"🔍 Multi-query: {len(queries)} requêtes → {len(all_docs)} documents")
        return all_docs
    
    def clear_cache(self):
        """
        Vide le cache de requêtes
        """
        self.query_cache.clear()
        logger.info("💾 Cache vidé")


class DummyRetriever:
    """
    Retriever factice pour wrapper des documents dans la compression
    """
    def __init__(self, documents: List[Document]):
        self.documents = documents
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.documents