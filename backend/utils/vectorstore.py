# utils/intelligent_vectorstore.py
"""
Vectorstore Intelligent exploitant les capacités des LLMs
Recherche sémantique + conceptuelle avec traçabilité complète
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict
import time

import faiss
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

class IntelligentVectorStore:
    """
    Vectorstore intelligent avec double indexation (sémantique + conceptuelle)
    et exploitation maximale des capacités LLM
    """
    
    def __init__(
        self,
        persist_directory: Path = Path("data/intelligent_store"),
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        analyze_on_index: bool = True
    ):
        """
        Initialise le vectorstore intelligent
        
        Args:
            persist_directory: Chemin de persistance
            embedding_model: Modèle d'embeddings
            llm_model: Modèle LLM pour analyse
            analyze_on_index: Analyser les chunks lors de l'indexation
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Modèles
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.analyze_on_index = analyze_on_index
        
        # Index FAISS (plus performant que ChromaDB)
        self.dimension = 1536  # Dimension des embeddings OpenAI
        self.index = None
        self.id_to_metadata = {}
        self.id_to_text = {}
        self.id_to_embedding = {}
        
        # Index conceptuel (tags LLM)
        self.concept_index = defaultdict(set)  # concept -> {doc_ids}
        self.doc_concepts = {}  # doc_id -> [concepts]
        
        # Cache intelligent
        self.query_cache = {}  # query -> results
        self.concept_cache = {}  # doc_id -> concepts
        self.context_cache = {}  # doc_id -> context
        
        # Traçabilité
        self.chunk_relations = {}  # doc_id -> {prev, next}
        self.document_structure = {}  # doc_name -> structure
        
        # Charger l'index existant si disponible
        self._load_index()
        
        logger.info(f"✅ IntelligentVectorStore initialisé")
        logger.info(f"   - Embeddings: {embedding_model}")
        logger.info(f"   - LLM: {llm_model}")
        logger.info(f"   - Analyse auto: {analyze_on_index}")
    
    # ============================================================================
    # ANALYSE LLM DES CHUNKS
    # ============================================================================
    
    def _analyze_chunk_with_llm(self, text: str, metadata: Dict) -> Dict[str, Any]:
        """
        Analyze a chunk with LLM to extract concepts and information
        """
        analysis_prompt = PromptTemplate(
            input_variables=["text", "section"],
            template="""Analyze this tender text and extract the following information.
    Be precise and factual.

    TEXT (Section: {section}):
    {text}

    Return in JSON with this exact structure:
    {{
        "concepts": ["list", "of", "key", "concepts"],
        "entities": {{
            "companies": [],
            "dates": [],
            "amounts": [],
            "locations": []
        }},
        "requirements": [],  // Identified requirements
        "key_points": [],    // Key points (max 3)
        "category": "",      // financial/legal/technical/operational/timeline/general
        "priority": 0.0,     // Importance 0-1
        "summary": ""        // 1-sentence summary
    }}"""
        )
        
        try:
            chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
            result = chain.run(text=text[:400], section=metadata.get('section', 'N/A'))
            
            # Parser le JSON
            analysis = json.loads(result)
            
            # Validation et nettoyage
            analysis['concepts'] = analysis.get('concepts', [])[:10]  # Max 10 concepts
            analysis['priority'] = max(0.0, min(1.0, float(analysis.get('priority', 0.5))))
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Erreur analyse LLM: {e}")
            # Analyse de fallback basique
            return {
                "concepts": self._extract_basic_concepts(text),
                "entities": {"companies": [], "dates": [], "amounts": [], "locations": []},
                "requirements": [],
                "key_points": [],
                "category": "general",
                "priority": 0.5,
                "summary": text[:100] + "..."
            }
    
    def _extract_basic_concepts(self, text: str) -> List[str]:
        """
        Extraction basique de concepts si LLM échoue
        """
        text_lower = text.lower()
        concepts = []
        
        # Mots-clés par domaine
        concept_keywords = {
            "deadline": ["deadline", "date limite", "échéance", "submission"],
            "pricing": ["price", "prix", "cost", "coût", "tarif"],
            "technical": ["technical", "technique", "specification", "requirement"],
            "legal": ["contract", "liability", "insurance", "terms"],
            "volume": ["volume", "quantity", "tonnage", "teu"],
            "transport": ["transport", "logistics", "shipping", "delivery"]
        }
        
        for concept, keywords in concept_keywords.items():
            if any(kw in text_lower for kw in keywords):
                concepts.append(concept)
        
        return concepts
    
    # ============================================================================
    # INDEXATION INTELLIGENTE
    # ============================================================================
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> int:
        """
        Ajoute des chunks avec analyse LLM et double indexation
        
        Args:
            chunks: Chunks du EnhancedChunker
            batch_size: Taille des batches
            
        Returns:
            Nombre de chunks ajoutés
        """
        if not chunks:
            return 0
        
        logger.info(f"🔄 Indexation intelligente de {len(chunks)} chunks...")
        
        # Initialiser l'index FAISS si nécessaire
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product pour similarité
        
        added_count = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for chunk in batch:
                try:
                    # 1. Enrichir les métadonnées avec position détaillée
                    chunk_id = self._generate_chunk_id(chunk)
                    enriched_metadata = self._enrich_metadata(chunk, chunk_id)
                    
                    # 2. Analyser avec LLM si activé
                    if self.analyze_on_index:
                        analysis = self._analyze_chunk_with_llm(
                            chunk['text'], 
                            enriched_metadata
                        )
                        enriched_metadata['llm_analysis'] = analysis
                        
                        # Indexer les concepts
                        self._index_concepts(chunk_id, analysis['concepts'])
                        self.doc_concepts[chunk_id] = analysis['concepts']
                    
                    # 3. Créer l'embedding
                    embedding = self.embeddings.embed_query(chunk['text'])
                    embedding_np = np.array([embedding], dtype=np.float32)
                    
                    # Normaliser pour Inner Product
                    faiss.normalize_L2(embedding_np)
                    
                    # 4. Ajouter à l'index FAISS
                    self.index.add(embedding_np)
                    
                    # 5. Stocker les métadonnées et relations
                    doc_index = self.index.ntotal - 1
                    self.id_to_metadata[doc_index] = enriched_metadata
                    self.id_to_text[doc_index] = chunk['text']
                    self.id_to_embedding[doc_index] = embedding
                    
                    # 6. Établir les relations avec chunks adjacents
                    self._establish_chunk_relations(doc_index, chunk, chunks)
                    
                    added_count += 1
                    
                except Exception as e:
                    logger.error(f"Erreur indexation chunk: {e}")
                    continue
            
            logger.info(f"  → Batch {i//batch_size + 1}: {len(batch)} chunks traités")
        
        # Sauvegarder l'index
        self._save_index()
        
        logger.info(f"✅ {added_count} chunks indexés avec succès")
        return added_count
    
    def _enrich_metadata(self, chunk: Dict, chunk_id: str) -> Dict[str, Any]:
        """
        Enrichit les métadonnées avec informations de traçabilité
        """
        metadata = chunk.get('metadata', {}).copy()
        
        # Ajouter traçabilité détaillée
        metadata.update({
            'chunk_id': chunk_id,
            'indexed_at': datetime.now().isoformat(),
            'chunk_type': chunk.get('type', 'unknown'),
            
            # Position hiérarchique
            'document': metadata.get('source', 'unknown'),
            'section': metadata.get('section', ''),
            'page': metadata.get('page', 0),
            'position': metadata.get('position', ''),
            
            # Breadcrumb pour navigation
            'breadcrumb': self._create_breadcrumb(metadata),
            
            # Taille et complétude
            'char_count': len(chunk.get('text', '')),
            'is_complete': metadata.get('is_complete_section', False)
        })
        
        return metadata
    
    def _create_breadcrumb(self, metadata: Dict) -> str:
        """
        Crée un fil d'Ariane pour la traçabilité
        """
        parts = []
        
        if doc := metadata.get('document'):
            parts.append(doc)
        if section := metadata.get('section'):
            parts.append(f"Section: {section}")
        if page := metadata.get('page'):
            parts.append(f"Page {page}")
        
        return " > ".join(parts) if parts else "Document"
    
    def _generate_chunk_id(self, chunk: Dict) -> str:
        """
        Génère un ID unique pour un chunk
        """
        content = f"{chunk.get('text', '')}{chunk.get('metadata', {})}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _index_concepts(self, chunk_id: str, concepts: List[str]):
        """
        Indexe les concepts pour recherche rapide
        """
        for concept in concepts:
            self.concept_index[concept.lower()].add(chunk_id)
    
    def _establish_chunk_relations(
        self, 
        doc_index: int, 
        current_chunk: Dict, 
        all_chunks: List[Dict]
    ):
        """
        Établit les relations prev/next entre chunks
        """
        current_source = current_chunk.get('metadata', {}).get('source')
        current_section = current_chunk.get('metadata', {}).get('section')
        
        relations = {
            'prev': None,
            'next': None,
            'same_section': [],
            'same_document': []
        }
        
        # Trouver les chunks liés
        for i, chunk in enumerate(all_chunks):
            if chunk == current_chunk:
                # Chunk précédent
                if i > 0:
                    prev_chunk = all_chunks[i-1]
                    if prev_chunk.get('metadata', {}).get('source') == current_source:
                        relations['prev'] = self._generate_chunk_id(prev_chunk)
                
                # Chunk suivant
                if i < len(all_chunks) - 1:
                    next_chunk = all_chunks[i+1]
                    if next_chunk.get('metadata', {}).get('source') == current_source:
                        relations['next'] = self._generate_chunk_id(next_chunk)
        
        self.chunk_relations[doc_index] = relations
    
    # ============================================================================
    # RECHERCHE INTELLIGENTE
    # ============================================================================
    
    def intelligent_search(
        self,
        query: str,
        k: int = 5,
        use_cache: bool = True,
        refine_with_llm: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recherche intelligente multi-phase avec LLM
        
        Args:
            query: Question de l'utilisateur
            k: Nombre de résultats
            use_cache: Utiliser le cache
            refine_with_llm: Raffiner avec LLM
            
        Returns:
            Résultats enrichis avec traçabilité
        """
        # 1. Vérifier le cache
        cache_key = f"{query}_{k}"
        if use_cache and cache_key in self.query_cache:
            logger.info(f"📋 Résultats depuis cache pour: {query[:50]}...")
            return self.query_cache[cache_key]
        
        logger.info(f"🔍 Recherche intelligente: {query[:50]}...")
        
        # 2. Phase 1: Comprendre la requête avec LLM
        query_analysis = self._analyze_query(query)
        logger.info(f"   Concepts identifiés: {query_analysis['concepts']}")
        
        # 3. Phase 2: Double recherche
        # 3a. Recherche sémantique
        semantic_results = self._semantic_search(
            query, 
            k=k * 2  # Over-fetch pour scoring
        )
        
        # 3b. Recherche conceptuelle
        conceptual_results = self._conceptual_search(
            query_analysis['concepts'],
            k=k * 2
        )
        
        # 4. Fusionner et dédupliquer
        all_results = self._merge_results(semantic_results, conceptual_results)
        
        # 5. Phase 3: Scoring intelligent par LLM
        if refine_with_llm and all_results:
            scored_results = self._score_with_llm(query, all_results, query_analysis)
        else:
            scored_results = all_results[:k]
        
        # 6. Enrichir avec contexte
        final_results = self._enrich_with_context(scored_results[:k])
        
        # 7. Mettre en cache
        if use_cache:
            self.query_cache[cache_key] = final_results
        
        return final_results
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyse la requête avec LLM pour comprendre l'intention
        """
        analysis_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Analyze this tender-related question and identify:
1. Key concepts being searched
2. Type of information needed (financial, legal, technical, timeline, operational)
3. Entities mentioned
4. Possible reformulations

Question: {query}

Return in JSON format:
{{
    "concepts": ["list", "of", "concepts"],
    "category": "financial/legal/technical/timeline/operational/general",
    "entities": [],
    "reformulations": ["reformulation 1", "reformulation 2"],
    "priority_sections": ["priority sections to search"]
}}"""
        )
        
        try:
            chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
            result = chain.run(query=query)
            return json.loads(result)
        except:
            # Fallback basique
            return {
                "concepts": query.lower().split()[:5],
                "category": "general",
                "entities": [],
                "reformulations": [query],
                "priority_sections": []
            }
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """
        Recherche par similarité sémantique avec FAISS
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Créer l'embedding de la requête
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Recherche FAISS
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        # Retourner (index, score) pairs
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS retourne -1 si pas assez de résultats
                results.append((int(idx), float(dist)))
        
        return results
    
    def _conceptual_search(
        self, 
        concepts: List[str], 
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Recherche par concepts (tags LLM)
        """
        if not concepts:
            return []
        
        # Compter les matches de concepts
        doc_scores = defaultdict(float)
        
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Chercher dans l'index conceptuel
            if concept_lower in self.concept_index:
                for chunk_id in self.concept_index[concept_lower]:
                    # Retrouver l'index depuis le chunk_id
                    for idx, metadata in self.id_to_metadata.items():
                        if metadata.get('chunk_id') == chunk_id:
                            doc_scores[idx] += 1.0 / len(concepts)
                            break
        
        # Trier par score et retourner top k
        sorted_results = sorted(
            doc_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        return sorted_results
    
    def _merge_results(
        self,
        semantic_results: List[Tuple[int, float]],
        conceptual_results: List[Tuple[int, float]]
    ) -> List[Dict[str, Any]]:
        """
        Fusionne les résultats sémantiques et conceptuels
        """
        # Combiner les scores
        combined_scores = {}
        
        # Pondération: 70% sémantique, 30% conceptuel
        for idx, score in semantic_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.7
        
        for idx, score in conceptual_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + score * 0.3
        
        # Trier par score combiné
        sorted_indices = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Créer les résultats
        results = []
        for idx, score in sorted_indices:
            if idx in self.id_to_text:
                results.append({
                    'index': idx,
                    'text': self.id_to_text[idx],
                    'metadata': self.id_to_metadata[idx],
                    'initial_score': score
                })
        
        return results
    
    def _score_with_llm(
        self,
        query: str,
        results: List[Dict],
        query_analysis: Dict
    ) -> List[Dict]:
        """
        Score les résultats avec le LLM pour pertinence réelle
        """
        if not results:
            return results
        
        scoring_prompt = PromptTemplate(
    input_variables=["query", "text", "section", "category"],
    template="""Evaluate the relevance of this text to the question.

Question: {query}
Category sought: {category}
Document section: {section}

Text to evaluate:
{text}

Give a score from 0 to 10 where:
- 0-3: Not relevant
- 4-6: Partially relevant  
- 7-9: Highly relevant
- 10: Perfect match

Reply ONLY with the numeric score."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=scoring_prompt)
        
        scored_results = []
        for result in results[:10]:  # Limiter pour performance
            try:
                score_str = chain.run(
                    query=query,
                    text=result['text'][:1500],  # Limiter taille
                    section=result['metadata'].get('section', 'N/A'),
                    category=query_analysis.get('category', 'general')
                )
                
                # Parser le score
                score = float(score_str.strip())
                result['llm_score'] = max(0, min(10, score))
                scored_results.append(result)
                
            except Exception as e:
                # Garder avec score initial si erreur
                result['llm_score'] = result['initial_score'] * 10
                scored_results.append(result)
        
        # Trier par score LLM
        scored_results.sort(key=lambda x: x['llm_score'], reverse=True)
        
        return scored_results
    
    def _enrich_with_context(self, results: List[Dict]) -> List[Dict]:
        """
        Enrichit les résultats avec contexte adjacent
        """
        enriched_results = []
        
        for result in results:
            idx = result['index']
            
            # Récupérer les chunks adjacents
            relations = self.chunk_relations.get(idx, {})
            
            context = {
                'main_chunk': result['text'],
                'prev_snippet': '',
                'next_snippet': '',
                'breadcrumb': result['metadata'].get('breadcrumb', ''),
                'section': result['metadata'].get('section', ''),
                'page': result['metadata'].get('page', 'N/A')
            }
            
            # Ajouter snippets des chunks adjacents
            if relations.get('prev'):
                for prev_idx, prev_meta in self.id_to_metadata.items():
                    if prev_meta.get('chunk_id') == relations['prev']:
                        context['prev_snippet'] = self.id_to_text[prev_idx][-200:]
                        break
            
            if relations.get('next'):
                for next_idx, next_meta in self.id_to_metadata.items():
                    if next_meta.get('chunk_id') == relations['next']:
                        context['next_snippet'] = self.id_to_text[next_idx][:200]
                        break
            
            # Créer le résultat enrichi avec traçabilité complète
            enriched_result = {
                'text': result['text'],
                'score': result.get('llm_score', result['initial_score']),
                'metadata': result['metadata'],
                'context': context,
                'source': {
                    'document': result['metadata'].get('document', 'Unknown'),
                    'section': result['metadata'].get('section', ''),
                    'page': result['metadata'].get('page', 'N/A'),
                    'chunk_id': result['metadata'].get('chunk_id', ''),
                    'breadcrumb': context['breadcrumb']
                },
                'confidence': min(100, result.get('llm_score', 5) * 10)
            }
            
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    # ============================================================================
    # REFINEMENT ET VALIDATION
    # ============================================================================
    
    def refine_answer(
        self,
        query: str,
        initial_results: List[Dict],
        max_iterations: int = 2
    ) -> Dict[str, Any]:
        """
        Raffine la réponse en demandant plus de contexte si nécessaire
        """
        refinement_prompt = PromptTemplate(
            input_variables=["query", "current_answer"],
            template="""Analyze this answer to the question and determine if it is complete.

        Question: {query}
        Current answer based on documents:
        {current_answer}

        Is this answer complete and satisfactory?
        If not, what information is missing?

        Return in JSON:
        {{
            "is_complete": true/false,
            "missing_info": ["list of missing information"],
            "suggested_searches": ["suggested additional queries"],
            "confidence": 0.0-1.0
        }}"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=refinement_prompt)
        
        current_answer = "\n".join([r['text'][:500] for r in initial_results])
        iteration = 0
        all_results = initial_results.copy()
        
        while iteration < max_iterations:
            try:
                analysis = json.loads(chain.run(
                    query=query,
                    current_answer=current_answer
                ))
                
                if analysis['is_complete'] or analysis['confidence'] > 0.8:
                    break
                
                # Chercher plus d'informations
                for suggested_search in analysis.get('suggested_searches', [])[:2]:
                    additional_results = self.intelligent_search(
                        suggested_search,
                        k=3,
                        use_cache=False,
                        refine_with_llm=False
                    )
                    all_results.extend(additional_results)
                
                current_answer = "\n".join([r['text'][:500] for r in all_results])
                iteration += 1
                
            except Exception as e:
                logger.warning(f"Erreur refinement: {e}")
                break
        
        return {
            'final_results': all_results,
            'is_complete': len(all_results) > len(initial_results),
            'iterations': iteration
        }
    
    # ============================================================================
    # PERSISTANCE
    # ============================================================================
    
    def _save_index(self):
        """
        Sauvegarde l'index et les métadonnées
        """
        try:
            # Sauvegarder l'index FAISS
            if self.index and self.index.ntotal > 0:
                faiss.write_index(
                    self.index,
                    str(self.persist_directory / "faiss.index")
                )
            
            # Sauvegarder les métadonnées
            metadata = {
                'id_to_metadata': self.id_to_metadata,
                'id_to_text': self.id_to_text,
                'concept_index': dict(self.concept_index),
                'doc_concepts': self.doc_concepts,
                'chunk_relations': self.chunk_relations,
                'document_structure': self.document_structure
            }
            
            with open(self.persist_directory / "metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            # Sauvegarder le cache
            cache_data = {
                'query_cache': self.query_cache,
                'concept_cache': self.concept_cache
            }
            
            with open(self.persist_directory / "cache.json", 'w') as f:
                json.dump(cache_data, f, default=str)
            
            logger.info(f"💾 Index sauvegardé: {self.index.ntotal if self.index else 0} documents")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
    
    def _load_index(self):
        """
        Charge l'index existant
        """
        try:
            # Charger l'index FAISS
            index_path = self.persist_directory / "faiss.index"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"📂 Index FAISS chargé: {self.index.ntotal} documents")
            
            # Charger les métadonnées
            metadata_path = self.persist_directory / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.id_to_metadata = metadata.get('id_to_metadata', {})
                    self.id_to_text = metadata.get('id_to_text', {})
                    self.concept_index = defaultdict(set, metadata.get('concept_index', {}))
                    self.doc_concepts = metadata.get('doc_concepts', {})
                    self.chunk_relations = metadata.get('chunk_relations', {})
                    self.document_structure = metadata.get('document_structure', {})
            
            # Charger le cache
            cache_path = self.persist_directory / "cache.json"
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                    self.query_cache = cache_data.get('query_cache', {})
                    self.concept_cache = cache_data.get('concept_cache', {})
            
        except Exception as e:
            logger.warning(f"Impossible de charger l'index existant: {e}")
    
    # ============================================================================
    # MÉTHODES UTILITAIRES
    # ============================================================================
    
    def clear_cache(self):
        """Vide le cache des requêtes"""
        self.query_cache.clear()
        self.concept_cache.clear()
        logger.info("🗑️ Cache vidé")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du vectorstore"""
        return {
            'total_documents': self.index.ntotal if self.index else 0,
            'total_concepts': len(self.concept_index),
            'cached_queries': len(self.query_cache),
            'persist_directory': str(self.persist_directory)
        }
    
    def delete_all(self):
        """Supprime tout l'index"""
        self.index = None
        self.id_to_metadata.clear()
        self.id_to_text.clear()
        self.id_to_embedding.clear()
        self.concept_index.clear()
        self.doc_concepts.clear()
        self.clear_cache()
        
        # Supprimer les fichiers
        for file in self.persist_directory.glob("*"):
            file.unlink()
        
        logger.info("❌ Index complètement supprimé")


# ============================================================================
# FONCTION D'INTERFACE PRINCIPALE
# ============================================================================

def create_intelligent_store(
    chunks: List[Dict[str, Any]],
    persist_directory: Path = Path("data/intelligent_store"),
    analyze_on_index: bool = True
) -> IntelligentVectorStore:
    """
    Crée et peuple un vectorstore intelligent
    
    Args:
        chunks: Résultats du EnhancedChunker
        persist_directory: Répertoire de persistance
        analyze_on_index: Analyser avec LLM lors de l'indexation
        
    Returns:
        IntelligentVectorStore configuré et peuplé
    """
    logger.info("🚀 Création du vectorstore intelligent...")
    
    # Créer le store
    store = IntelligentVectorStore(
        persist_directory=persist_directory,
        analyze_on_index=analyze_on_index
    )
    
    # Indexer les chunks
    if chunks:
        store.add_chunks(chunks)
    
    stats = store.get_stats()
    logger.info(f"✅ Vectorstore prêt:")
    logger.info(f"   - Documents: {stats['total_documents']}")
    logger.info(f"   - Concepts: {stats['total_concepts']}")
    
    return store


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # S'assurer que la clé OpenAI est disponible
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY non trouvée")
        exit(1)
    
    print("🧠 Test du Vectorstore Intelligent")
    print("=" * 60)
    
    # Créer des chunks de test
    test_chunks = [
        {
            "type": "section",
            "text": "The tender submission deadline is January 15, 2025. All proposals must be submitted electronically through our portal before 5:00 PM CET. Late submissions will not be accepted.",
            "metadata": {
                "source": "tender_doc.pdf",
                "section": "2.1 Timeline",
                "page": 5
            }
        },
        {
            "type": "section", 
            "text": "The total budget for this project is EUR 2.5 million. Payment terms are 30% advance, 40% on delivery, and 30% after acceptance. Penalties apply for late delivery.",
            "metadata": {
                "source": "tender_doc.pdf",
                "section": "3.2 Financial Terms",
                "page": 12
            }
        }
    ]
    
    # Créer le store
    store = create_intelligent_store(
        test_chunks,
        analyze_on_index=True
    )
    
    # Test de recherche
    print("\n🔍 Test de recherche intelligente")
    print("-" * 40)
    
    queries = [
        "What is the deadline for submission?",
        "What are the payment terms?",
        "Budget and financial conditions"
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        results = store.intelligent_search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Result {i}:")
            print(f"   Score: {result['score']:.2f}")
            print(f"   Source: {result['source']['breadcrumb']}")
            print(f"   Text: {result['text'][:150]}...")
            print(f"   Confidence: {result['confidence']}%")
    
    # Afficher les stats
    print("\n📊 Statistiques finales:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")