# utils/enhanced_retrieval.py
"""
Système de retrieval simplifié et corrigé pour GPT-4o-mini
Utilise la nouvelle syntaxe LangChain sans deprecation warnings
"""

import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Résultat de retrieval"""
    query: str
    answer: str
    chunks: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        return {
            'query': self.query,
            'answer': self.answer,
            'chunks_used': len(self.chunks),
            'sources': self.sources,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }


class IntelligentRetriever:
    """
    Retriever simplifié sans syntaxe dépréciée
    """
    
    def __init__(
        self,
        vectorstore,
        llm_model: str = "gpt-4o-mini",
        enable_refinement: bool = False,
        enable_multi_hop: bool = False,
        max_context_size: int = 4000
    ):
        """
        Initialise le retriever
        
        Args:
            vectorstore: VectorStore configuré
            llm_model: Modèle LLM (gpt-4o-mini)
            enable_refinement: Ignoré (pour compatibilité)
            enable_multi_hop: Ignoré (pour compatibilité)
            max_context_size: Taille max du contexte
        """
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.1,
            max_tokens=1500
        )
        self.max_context_size = max_context_size
        
        # Stats simples
        self.stats = {
            'queries_processed': 0,
            'chunks_retrieved': 0
        }
        
        logger.info(f"✅ IntelligentRetriever initialisé avec {llm_model}")
    
    def retrieve_and_answer(
        self,
        query: str,
        category: Optional[str] = None,
        require_source: bool = True,
        max_chunks: int = 5
    ) -> RetrievalResult:
        """
        Pipeline simplifié : Recherche → Réponse
        
        Args:
            query: Question
            category: Catégorie (optionnel)
            require_source: Exiger les sources
            max_chunks: Nombre max de chunks
            
        Returns:
            RetrievalResult avec réponse
        """
        self.stats['queries_processed'] += 1
        
        logger.info(f"🔍 Requête: {query[:100]}...")
        
        try:
            # 1. Analyser la requête (simplifié, sans JSON parsing)
            query_analysis = self._analyze_query_simple(query)
            
            # 2. Rechercher dans le vectorstore
            retrieved_chunks = self._search_vectorstore(
                query,
                query_analysis,
                max_chunks
            )
            self.stats['chunks_retrieved'] += len(retrieved_chunks)
            
            # 3. Générer une réponse simple
            answer_data = self._generate_answer_simple(
                query,
                retrieved_chunks,
                require_source
            )
            
            # 4. Extraire les sources
            sources = self._extract_sources(retrieved_chunks)
            
            # Créer le résultat
            result = RetrievalResult(
                query=query,
                answer=answer_data['answer'],
                chunks=retrieved_chunks,
                sources=sources,
                confidence=answer_data['confidence'],
                metadata={
                    'category': category,
                    'chunks_analyzed': len(retrieved_chunks),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Réponse générée (confiance: {result.confidence:.0%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur dans retrieve_and_answer: {e}")
            # Retourner un résultat vide en cas d'erreur
            return RetrievalResult(
                query=query,
                answer="Une erreur s'est produite lors de la recherche.",
                chunks=[],
                sources=[],
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _analyze_query_simple(self, query: str) -> Dict[str, Any]:
        """
        Analyse simplifiée de la requête SANS parsing JSON
        """
        # Analyse basique sans appel LLM pour éviter les erreurs JSON
        query_lower = query.lower()
        
        # Détection simple du type
        intent_type = "factual"
        if any(word in query_lower for word in ['compare', 'versus', 'difference']):
            intent_type = "comparative"
        elif any(word in query_lower for word in ['analyze', 'explain', 'why']):
            intent_type = "analytical"
        
        # Extraction basique des concepts (mots clés)
        words = query.split()
        concepts = [w for w in words if len(w) > 4][:5]  # Mots de plus de 4 lettres
        
        # Type de réponse attendu
        expected_type = "description"
        if any(word in query_lower for word in ['when', 'date', 'deadline']):
            expected_type = "date"
        elif any(word in query_lower for word in ['how much', 'amount', 'cost', 'price']):
            expected_type = "amount"
        elif any(word in query_lower for word in ['list', 'enumerate', 'what are']):
            expected_type = "list"
        
        return {
            'intent_type': intent_type,
            'concepts': concepts,
            'search_terms': concepts,
            'expected_answer_type': expected_type,
            'complexity': 'simple',
            'original_query': query
        }
    
    def _search_vectorstore(
        self,
        query: str,
        query_analysis: Dict,
        max_chunks: int
    ) -> List[Dict[str, Any]]:
        """
        Recherche simple dans le vectorstore
        """
        try:
            # Utiliser intelligent_search si disponible
            if hasattr(self.vectorstore, 'intelligent_search'):
                results = self.vectorstore.intelligent_search(
                    query=query,
                    k=max_chunks,
                    refine_with_llm=False  # Éviter les appels LLM supplémentaires
                )
            else:
                # Fallback sur similarity_search standard
                docs = self.vectorstore.similarity_search(query, k=max_chunks)
                results = []
                for doc in docs:
                    results.append({
                        'text': doc.page_content,
                        'source': {
                            'document': doc.metadata.get('source', 'Unknown'),
                            'section': doc.metadata.get('section', ''),
                            'page': doc.metadata.get('page', 0),
                            'chunk_id': doc.metadata.get('chunk_id', ''),
                            'breadcrumb': doc.metadata.get('breadcrumb', '')
                        },
                        'metadata': doc.metadata,
                        'score': 0.5,  # Score par défaut
                        'confidence': 0.5
                    })
            
            logger.info(f"   📄 {len(results)} chunks récupérés")
            return results
            
        except Exception as e:
            logger.error(f"Erreur recherche vectorstore: {e}")
            return []
    
    def _generate_answer_simple(
        self,
        query: str,
        chunks: List[Dict],
        require_source: bool
    ) -> Dict[str, Any]:
        """
        Génère une réponse simple sans template complexe
        """
        if not chunks:
            return {
                'answer': "Aucune information trouvée pour cette question.",
                'confidence': 0.0
            }
        
        # Préparer le contexte
        context = self._prepare_context_simple(chunks)
        
        # Prompt simple et direct
        prompt = f"""Answer this question based on the provided context.

Question: {query}

Context:
{context}

Provide a precise and factual answer. If the information is not in the context, state that clearly.

Answer:"""
        
        try:
            # Utiliser invoke() au lieu de chain.run()
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
            
            # Calculer une confiance simple
            confidence = 0.7  # Base
            if len(chunks) >= 3:
                confidence += 0.1
            if len(answer) > 100:
                confidence += 0.1
            if "pas" in answer.lower() and "trouvé" in answer.lower():
                confidence = 0.3
            
            return {
                'answer': answer,
                'confidence': min(1.0, confidence)
            }
            
        except Exception as e:
            logger.error(f"Erreur génération réponse: {e}")
            return {
                'answer': f"Erreur lors de la génération: {str(e)}",
                'confidence': 0.0
            }
    
    def _prepare_context_simple(self, chunks: List[Dict]) -> str:
        """
        Prépare le contexte de manière simple
        """
        context_parts = []
        total_size = 0
        
        for i, chunk in enumerate(chunks):
            # Info source
            source = chunk.get('source', {})
            source_info = f"[Source: {source.get('document', 'Unknown')}]"
            
            # Texte du chunk
            text = chunk.get('text', '')
            if len(text) > 500:
                text = text[:500] + "..."
            
            chunk_text = f"{source_info}\n{text}"
            
            # Vérifier la taille totale
            if total_size + len(chunk_text) > self.max_context_size:
                break
            
            context_parts.append(chunk_text)
            total_size += len(chunk_text)
        
        return "\n---\n".join(context_parts)
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extrait les sources des chunks
        """
        sources = []
        seen = set()
        
        for chunk in chunks:
            source = chunk.get('source', {})
            source_key = f"{source.get('document')}_{source.get('page')}"
            
            if source_key not in seen:
                sources.append({
                    'document': source.get('document', 'Unknown'),
                    'section': source.get('section', ''),
                    'page': source.get('page', 0),
                    'chunk_id': source.get('chunk_id', ''),
                    'breadcrumb': source.get('breadcrumb', ''),
                    'confidence': chunk.get('confidence', 0.5),
                    'score': chunk.get('score', 0.5)
                })
                seen.add(source_key)
        
        return sources[:5]  # Max 5 sources
    
    def process_questions_batch(
        self,
        questions: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[RetrievalResult]:
        """
        Traite plusieurs questions (compatibilité)
        """
        results = []
        total = len(questions)
        
        logger.info(f"📋 Traitement de {total} questions")
        
        for i, q in enumerate(questions, 1):
            logger.info(f"[{i}/{total}] {q.get('question', '')[:50]}...")
            
            result = self.retrieve_and_answer(
                query=q.get('question', ''),
                category=q.get('category'),
                require_source=q.get('require_source', True)
            )
            results.append(result)
        
        logger.info(f"✅ {total} questions traitées")
        return results
    
    def clear_cache(self):
        """Vide les caches (compatibilité)"""
        logger.info("Cache vidé")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les stats"""
        return self.stats.copy()


# Factory function pour compatibilité
def create_intelligent_retriever(vectorstore, **kwargs) -> IntelligentRetriever:
    """Crée un retriever configuré"""
    return IntelligentRetriever(vectorstore, **kwargs)


# Test
if __name__ == "__main__":
    import os
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY non trouvée")
        exit(1)
    
    print("🧠 Test Enhanced Retrieval Simplifié")
    print("=" * 60)
    
    # Test basique sans vectorstore réel
    class MockVectorstore:
        def similarity_search(self, query, k=5):
            # Retourne des documents factices
            from types import SimpleNamespace
            return [
                SimpleNamespace(
                    page_content="Test content about deadlines and dates.",
                    metadata={'source': 'test.pdf', 'page': 1}
                )
            ]
    
    vectorstore = MockVectorstore()
    retriever = IntelligentRetriever(vectorstore)
    
    # Test simple
    result = retriever.retrieve_and_answer(
        query="What is the deadline?",
        category="timeline"
    )
    
    print(f"Question: {result.query}")
    print(f"Réponse: {result.answer}")
    print(f"Confiance: {result.confidence:.0%}")
    print("✅ Test terminé")