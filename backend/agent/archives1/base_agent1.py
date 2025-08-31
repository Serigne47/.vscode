# agents/base_agent.py
"""
Classe de base pour tous les agents d'extraction AO
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.documents import Document
import json
import re
import logging

logger = logging.getLogger(__name__)

class BaseExtractionAgent(ABC):
    """
    Classe abstraite définissant l'interface commune pour tous les agents d'extraction
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        Initialisation de l'agent de base
        
        Args:
            model: Modèle OpenAI à utiliser
            temperature: Température pour la génération (0 = déterministe)
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=4000
        )
        self.category = self.__class__.__name__.replace("Agent", "").lower()
        self.extraction_prompt = self._create_extraction_prompt()
        self.validation_patterns = self._define_validation_patterns()
        
    @abstractmethod
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Création du prompt spécifique à chaque agent
        """
        pass
    
    @abstractmethod
    def _define_validation_patterns(self) -> Dict[str, str]:
        """
        Définition des patterns regex pour validation
        """
        pass
    
    @abstractmethod
    def _post_process(self, raw_extraction: Dict) -> Dict:
        """
        Post-traitement spécifique à chaque type d'extraction
        """
        pass
    
    async def extract(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Pipeline d'extraction complet
        
        Args:
            chunks: Documents à analyser
            
        Returns:
            Dictionnaire avec les informations extraites et métadonnées
        """
        try:
            # 1. Préparation du contexte
            context = self._prepare_context(chunks)
            
            # 2. Extraction LLM
            llm_result = await self._llm_extract(context)
            
            # 3. Extraction par patterns
            pattern_result = self._pattern_extract(chunks)
            
            # 4. Fusion des résultats
            merged = self._merge_results(llm_result, pattern_result)
            
            # 5. Post-traitement
            processed = self._post_process(merged)
            
            # 6. Calcul du score de confiance
            confidence = self._calculate_confidence(processed, chunks)
            
            return {
                "data": processed,
                "confidence": confidence,
                "sources": self._extract_sources(chunks),
                "category": self.category
            }
            
        except Exception as e:
            logger.error(f"Erreur dans {self.category} agent: {e}")
            return {
                "data": {},
                "confidence": 0,
                "error": str(e),
                "category": self.category
            }
    
    def _prepare_context(self, chunks: List[Document]) -> str:
        """
        Prépare le contexte pour l'extraction LLM
        """
        # Limite à 15000 caractères pour rester dans les limites
        context = ""
        for chunk in chunks:
            if len(context) + len(chunk.page_content) > 15000:
                break
            context += f"\n---\n{chunk.page_content}"
        return context
    
    async def _llm_extract(self, context: str) -> Dict:
        """
        Extraction via LLM
        """
        messages = self.extraction_prompt.format_messages(context=context)
        response = await self.llm.ainvoke(messages)
        
        try:
            # Parse JSON response
            content = response.content
            # Nettoyer le markdown si présent
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Impossible de parser JSON pour {self.category}")
            return {}
    
    def _pattern_extract(self, chunks: List[Document]) -> Dict:
        """
        Extraction par regex patterns
        """
        results = {}
        full_text = " ".join([c.page_content for c in chunks])
        
        for field, pattern in self.validation_patterns.items():
            matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if matches:
                results[field] = matches
                
        return results
    
    def _merge_results(self, llm_result: Dict, pattern_result: Dict) -> Dict:
        """
        Fusion intelligente des résultats LLM et patterns
        """
        merged = llm_result.copy()
        
        # Enrichir avec les patterns trouvés
        for field, values in pattern_result.items():
            if field not in merged or not merged[field]:
                merged[field] = values[0] if len(values) == 1 else values
            elif isinstance(merged[field], list):
                # Dédupliquer et merger
                existing = set(merged[field]) if isinstance(merged[field][0], str) else merged[field]
                for v in values:
                    if v not in existing:
                        merged[field].append(v)
        
        return merged
    
    def _calculate_confidence(self, extraction: Dict, chunks: List[Document]) -> float:
        """
        Calcul du score de confiance basé sur plusieurs facteurs
        """
        confidence = 0.5  # Base
        
        # Bonus si données trouvées
        if extraction and any(extraction.values()):
            confidence += 0.2
        
        # Bonus si validation patterns match
        pattern_matches = sum(1 for k, v in extraction.items() 
                             if k in self.validation_patterns and v)
        confidence += (pattern_matches / max(len(self.validation_patterns), 1)) * 0.3
        
        return min(confidence, 1.0)
    
    def _extract_sources(self, chunks: List[Document]) -> List[Dict]:
        """
        Extraction des sources pour traçabilité
        """
        sources = []
        for chunk in chunks[:3]:  # Top 3 chunks les plus pertinents
            sources.append({
                "content_preview": chunk.page_content[:200],
                "metadata": chunk.metadata
            })
        return sources