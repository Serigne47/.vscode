# agents/orchestrator.py
"""
Orchestrateur principal pour la coordination des agents d'extraction
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

from .identity_agent_arch import IdentityExtractionAgent
from .volume_agent import VolumeExtractionAgent
from .financial_agent import FinancialExtractionAgent
from .legal_agent import LegalExtractionAgent
from .operational_agent import OperationalExtractionAgent
from .timeline_agent import TimelineExtractionAgent

logger = logging.getLogger(__name__)

class AOExtractionOrchestrator:
    """
    Orchestrateur principal coordonnant tous les agents spécialisés
    """
    
    def __init__(self, vectorstore=None, config: Optional[Dict] = None):
        """
        Initialisation de l'orchestrateur
        
        Args:
            vectorstore: Base vectorielle pour le RAG
            config: Configuration optionnelle
        """
        self.config = config or {}
        self.vectorstore = vectorstore
        
        # LLM pour la synthèse finale
        self.synthesis_llm = ChatOpenAI(
            model=self.config.get('model', 'gpt-4o-mini'),
            temperature=0.1,
            max_tokens=4000
        )
        
        # Initialisation des agents spécialisés
        self.specialized_agents = {
            "identity": IdentityExtractionAgent(),
            "volume": VolumeExtractionAgent(),
            "financial": FinancialExtractionAgent(),
            "legal": LegalExtractionAgent(),
            "operational": OperationalExtractionAgent(),
            "timeline": TimelineExtractionAgent()
        }
        
        # Statistiques d'exécution
        self.stats = {
            'start_time': None,
            'end_time': None,
            'documents_processed': 0,
            'chunks_analyzed': 0,
            'agents_run': 0
        }
    
    async def analyze_ao(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Pipeline principal d'analyse d'appel d'offres
        
        Args:
            documents: Liste des documents à analyser
            
        Returns:
            Rapport complet d'analyse
        """
        self.stats['start_time'] = datetime.now()
        self.stats['documents_processed'] = len(documents)
        
        try:
            # Phase 1: Analyse de structure
            logger.info("Phase 1: Analyse de la structure documentaire")
            structure = await self._analyze_document_structure(documents)
            
            # Phase 2: Classification et routage
            logger.info("Phase 2: Classification et routage des chunks")
            routed_chunks = self._route_chunks_to_agents(documents, structure)
            
            # Phase 3: Extraction parallèle par agents
            logger.info("Phase 3: Extraction multi-agent parallèle")
            extraction_results = await self._parallel_extraction(routed_chunks)
            
            # Phase 4: Consolidation et validation croisée
            logger.info("Phase 4: Consolidation et validation")
            consolidated = self._consolidate_results(extraction_results)
            
            # Phase 5: Enrichissement et scoring
            logger.info("Phase 5: Enrichissement et scoring")
            enriched = await self._enrich_and_score(consolidated, documents)
            
            # Phase 6: Génération du rapport final
            logger.info("Phase 6: Génération du rapport")
            report = self._generate_final_report(enriched)
            
            self.stats['end_time'] = datetime.now()
            self.stats['processing_time'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            return {
                'status': 'success',
                'report': report,
                'metadata': self.stats,
                'confidence_global': self._calculate_global_confidence(report)
            }
            
        except Exception as e:
            logger.error(f"Erreur dans l'orchestration: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'partial_results': {}
            }
    
    async def _analyze_document_structure(self, documents: List[Document]) -> Dict:
        """
        Analyse la structure globale des documents
        """
        structure_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Analyse la structure de ces documents d'appel d'offres.
            
            Identifie:
            1. Les sections principales (cahier des charges, annexes, formulaires)
            2. La langue principale
            3. Le type de document (RFP, RFQ, ITB)
            4. L'organisation générale
            
            Retourne un JSON avec la structure identifiée."""),
            HumanMessage(content="{sample}")
        ])
        
        # Échantillon pour analyse rapide
        sample = "\n".join([doc.page_content[:500] for doc in documents[:5]])
        
        messages = structure_prompt.format_messages(sample=sample)
        response = await self.synthesis_llm.ainvoke(messages)
        
        try:
            return json.loads(response.content)
        except:
            return {'type': 'standard', 'sections': [], 'language': 'fr'}
    
    def _route_chunks_to_agents(self, documents: List[Document], structure: Dict) -> Dict[str, List[Document]]:
        """
        Route intelligemment les chunks vers les agents appropriés
        """
        routed = {agent: [] for agent in self.specialized_agents.keys()}
        
        # Keywords pour le routage
        routing_keywords = {
            'identity': ['client', 'émetteur', 'référence', 'appel offre', 'ao', 'rfp', 'tender', 'soumission'],
            'volume': ['volume', 'teu', 'tonnage', 'palette', 'quantité', 'm3', 'container', 'prévision'],
            'financial': ['prix', 'tarif', 'facturation', 'paiement', 'devise', 'garantie', 'coût', 'budget'],
            'legal': ['responsabilité', 'clause', 'résiliation', 'pénalité', 'assurance', 'juridique', 'loi'],
            'operational': ['transport', 'livraison', 'entrepôt', 'température', 'gdp', 'service', 'kpi'],
            'timeline': ['date', 'délai', 'deadline', 'échéance', 'calendrier', 'planning', 'durée']
        }
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            
            # Score de pertinence pour chaque agent
            scores = {}
            for agent, keywords in routing_keywords.items():
                score = sum(1 for kw in keywords if kw in content_lower)
                scores[agent] = score
            
            # Route vers les 3 agents les plus pertinents
            top_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for agent, score in top_agents:
                if score > 0:
                    routed[agent].append(doc)
        
        # Log distribution
        for agent, chunks in routed.items():
            logger.info(f"Agent {agent}: {len(chunks)} chunks assignés")
            self.stats['chunks_analyzed'] += len(chunks)
        
        return routed
    
    async def _parallel_extraction(self, routed_chunks: Dict[str, List[Document]]) -> Dict[str, Any]:
        """
        Exécution parallèle des agents sur leurs chunks respectifs
        """
        extraction_tasks = []
        
        for agent_name, chunks in routed_chunks.items():
            if chunks:  # Ne lancer que si des chunks sont assignés
                agent = self.specialized_agents[agent_name]
                task = self._run_agent_with_logging(agent_name, agent, chunks)
                extraction_tasks.append(task)
                self.stats['agents_run'] += 1
        
        # Exécution parallèle
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # Structurer les résultats
        extraction_results = {}
        for i, (agent_name, chunks) in enumerate([(k, v) for k, v in routed_chunks.items() if v]):
            if isinstance(results[i], Exception):
                logger.error(f"Erreur agent {agent_name}: {results[i]}")
                extraction_results[agent_name] = {'error': str(results[i])}
            else:
                extraction_results[agent_name] = results[i]
        
        return extraction_results
    
    async def _run_agent_with_logging(self, name: str, agent, chunks: List[Document]):
        """
        Exécute un agent avec logging
        """
        logger.info(f"Démarrage agent {name}")
        start = datetime.now()
        
        try:
            result = await agent.extract(chunks)
            duration = (datetime.now() - start).total_seconds()
            logger.info(f"Agent {name} terminé en {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Erreur agent {name}: {e}")
            raise
    
    def _consolidate_results(self, extraction_results: Dict[str, Any]) -> Dict:
        """
        Consolide les résultats de tous les agents
        """
        consolidated = {
            'identity': {},
            'volumes': {},
            'financial': {},
            'legal': {},
            'operational': {},
            'timeline': {},
            'confidence_scores': {},
            'errors': []
        }
        
        for agent_name, result in extraction_results.items():
            if 'error' in result:
                consolidated['errors'].append({
                    'agent': agent_name,
                    'error': result['error']
                })
            else:
                # Extraire les données
                if 'data' in result:
                    consolidated[agent_name] = result['data']
                
                # Extraire le score de confiance
                if 'confidence' in result:
                    consolidated['confidence_scores'][agent_name] = result['confidence']
        
        # Validation croisée
        consolidated['cross_validation'] = self._cross_validate(consolidated)
        
        return consolidated
    
    def _cross_validate(self, data: Dict) -> Dict:
        """
        Validation croisée entre les données des différents agents
        """
        validations = {
            'coherent': True,
            'issues': [],
            'warnings': []
        }
        
        # Vérifier cohérence volumes/financial
        if 'volumes' in data and 'financial' in data:
            # Si grandes quantités mais pas de dégressivité tarifaire
            if data['volumes'].get('aggregates', {}).get('total_teu_annuel', 0) > 10000:
                if not data['financial'].get('structure_tarifaire', {}).get('degressivite'):
                    validations['warnings'].append(
                        "Volumes importants sans dégressivité tarifaire mentionnée"
                    )
        
        # Vérifier cohérence timeline/legal
        if 'timeline' in data and 'legal' in data:
            # Durée contrat vs préavis résiliation
            if data['timeline'].get('contrat', {}).get('duree_initiale_mois'):
                duree = data['timeline']['contrat']['duree_initiale_mois']
                preavis = data['legal'].get('resiliation', {}).get('pour_convenance', {}).get('preavis')
                if preavis and '6 mois' in str(preavis) and duree < 12:
                    validations['issues'].append(
                        "Préavis de 6 mois pour contrat de moins d'un an"
                    )
        
        # Vérifier cohérence operational/legal
        if 'operational' in data and 'legal' in data:
            # GDP requis vs assurances
            if data['operational'].get('temperature', {}).get('gdp_requis'):
                if not data['legal'].get('assurances', {}).get('marchandises'):
                    validations['warnings'].append(
                        "GDP requis mais assurance marchandises non spécifiée"
                    )
        
        validations['coherent'] = len(validations['issues']) == 0
        
        return validations
    
    async def _enrich_and_score(self, consolidated: Dict, documents: List[Document]) -> Dict:
        """
        Enrichissement et scoring global
        """
        enriched = consolidated.copy()
        
        # Calcul de la complétude
        enriched['completeness'] = self._calculate_completeness(consolidated)
        
        # Identification des données manquantes critiques
        enriched['missing_critical'] = self._identify_missing_critical(consolidated)
        
        # Score de risque global
        enriched['risk_score'] = self._calculate_risk_score(consolidated)
        
        # Recommandations prioritaires
        enriched['recommendations'] = await self._generate_recommendations(consolidated)
        
        # Points d'attention urgents
        enriched['urgent_actions'] = self._identify_urgent_actions(consolidated)
        
        return enriched
    
    def _calculate_completeness(self, data: Dict) -> Dict:
        """
        Calcule le score de complétude des données
        """
        sections = {
            'identity': ['client', 'reference_ao', 'canal_reponse'],
            'volumes': ['volumes_conteneurs', 'volumes_poids', 'aggregates'],
            'financial': ['facturation', 'paiement', 'devise'],
            'legal': ['responsabilites', 'assurances', 'gouvernance'],
            'operational': ['transport', 'services', 'kpi_sla'],
            'timeline': ['soumission', 'contrat']
        }
        
        completeness = {}
        total_fields = 0
        found_fields = 0
        
        for section, required_fields in sections.items():
            section_data = data.get(section, {})
            section_found = sum(1 for field in required_fields if field in section_data and section_data[field])
            section_total = len(required_fields)
            
            completeness[section] = {
                'score': (section_found / section_total * 100) if section_total > 0 else 0,
                'found': section_found,
                'total': section_total
            }
            
            total_fields += section_total
            found_fields += section_found
        
        completeness['global'] = {
            'score': (found_fields / total_fields * 100) if total_fields > 0 else 0,
            'found': found_fields,
            'total': total_fields
        }
        
        return completeness
    
    def _identify_missing_critical(self, data: Dict) -> List[str]:
        """
        Identifie les informations critiques manquantes
        """
        critical = []
        
        # Identité
        if not data.get('identity', {}).get('client', {}).get('nom'):
            critical.append("Nom du client")
        if not data.get('identity', {}).get('reference_ao', {}).get('numero'):
            critical.append("Référence AO")
        
        # Timeline
        if not data.get('timeline', {}).get('soumission', {}).get('date_limite_principale'):
            critical.append("Date limite de soumission")
        
        # Volumes
        if not data.get('volumes', {}).get('aggregates'):
            critical.append("Volumes totaux")
        
        # Financial
        if not data.get('financial', {}).get('paiement', {}).get('delai_jours'):
            critical.append("Délai de paiement")
        
        return critical
    
    def _calculate_risk_score(self, data: Dict) -> Dict:
        """
        Calcule le score de risque global
        """
        risk_score = 0
        risk_factors = []
        
        # Risques juridiques
        if 'legal' in data and 'risk_analysis' in data['legal']:
            legal_risk = data['legal']['risk_analysis'].get('score', 0)
            risk_score += legal_risk
            if legal_risk > 5:
                risk_factors.append(f"Risque juridique élevé (score: {legal_risk})")
        
        # Risques financiers
        if 'financial' in data and 'risk_assessment' in data['financial']:
            financial_risk = data['financial']['risk_assessment'].get('score', 0)
            risk_score += financial_risk
            if financial_risk > 5:
                risk_factors.append(f"Risque financier élevé (score: {financial_risk})")
        
        # Risques opérationnels
        if 'operational' in data and 'complexity_analysis' in data['operational']:
            if data['operational']['complexity_analysis'].get('level') in ['ÉLEVÉ', 'TRÈS ÉLEVÉ']:
                risk_score += 5
                risk_factors.append("Complexité opérationnelle élevée")
        
        # Risques timeline
        if 'timeline' in data and 'urgencies' in data['timeline']:
            urgent_count = len(data['timeline']['urgencies'])
            if urgent_count > 0:
                risk_score += urgent_count * 2
                risk_factors.append(f"{urgent_count} échéances urgentes")
        
        # Déterminer le niveau de risque
        if risk_score >= 25:
            level = "CRITIQUE"
        elif risk_score >= 15:
            level = "ÉLEVÉ"
        elif risk_score >= 8:
            level = "MODÉRÉ"
        else:
            level = "FAIBLE"
        
        return {
            'score': risk_score,
            'level': level,
            'factors': risk_factors
        }