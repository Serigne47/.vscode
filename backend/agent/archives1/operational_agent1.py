# agents/operational_agent.py
"""
Agent opérationnel optimisé pour extraction des exigences d'exécution
Spécialisé dans les modes de transport, services, SLA/KPI et contraintes opérationnelles
"""
import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from .base_agent import YAMLBaseAgent, QuestionResult, ExtractionStatus, ExtractionSource

logger = logging.getLogger(__name__)

class OperationalExtractionAgent(YAMLBaseAgent):
    """
    Agent d'extraction opérationnelle pour appels d'offres logistiques
    
    Spécialisations:
    - Modes de transport (air, sea, road, rail, multimodal)
    - Services logistiques requis
    - Exigences opérationnelles et certifications
    - SLA/KPI et métriques de performance
    """
    
    def __init__(
        self,
        config_path: str = "config/prompts/execution_questions.yaml",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        enable_cache: bool = True,
        enable_parallel: bool = True,
        use_advanced_extraction: bool = True
    ):
        """Initialise l'agent opérationnel"""
        super().__init__(
            config_path=config_path,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            enable_cache=enable_cache,
            enable_parallel=enable_parallel,
            max_parallel=4
        )
        
        self.use_advanced = use_advanced_extraction
        
        if self.use_advanced:
            # LLM spécialisé pour extraction de métriques
            self.metrics_llm = ChatOpenAI(
                model=model,
                temperature=0.0,
                max_tokens=3000
            )
        
        # Catalogues opérationnels
        self.operational_catalog = self._init_operational_catalog()
        self.patterns = self._init_patterns()
        
        logger.info(f"✅ OperationalExtractionAgent initialized")
    
    def _init_operational_catalog(self) -> Dict[str, Any]:
        """Initialise les catalogues de référence opérationnels"""
        return {
            'transport_modes': {
                'air': ['air', 'aircraft', 'aviation', 'flight', 'airline', 'iata', 'awb'],
                'sea': ['sea', 'ocean', 'maritime', 'vessel', 'container', 'port', 'fcl', 'lcl'],
                'road': ['road', 'truck', 'ltl', 'ftl', 'highway', 'vehicle', 'cmr'],
                'rail': ['rail', 'train', 'railway', 'intermodal', 'wagon'],
                'multimodal': ['multimodal', 'intermodal', 'combined', 'door-to-door']
            },
            'services': {
                'freight_forwarding': ['freight forward', 'forwarding', 'ff'],
                'customs': ['customs', 'clearance', 'brokerage', 'import', 'export'],
                'warehousing': ['warehouse', 'storage', 'inventory', 'stock'],
                'cross_docking': ['cross-dock', 'cross dock', 'transhipment'],
                'last_mile': ['last mile', 'final delivery', 'home delivery'],
                'reverse_logistics': ['reverse', 'returns', 'recall'],
                'consolidation': ['consolidat', 'groupage', 'lcl'],
                'tracking': ['track', 'trace', 'visibility', 'monitoring'],
                'packaging': ['pack', 'repack', 'label', 'palletiz'],
                'distribution': ['distribut', 'delivery', 'dispatch']
            },
            'requirements': {
                'temperature': ['temperature', 'cold chain', 'reefer', 'frozen', 'chilled', 'gdp'],
                'tracking': ['gps', 'rfid', 'real-time', 'track', 'trace', 'iot'],
                'it_systems': ['edi', 'api', 'wms', 'tms', 'erp', 'integration', 'interface'],
                'certifications': ['iso', 'gdp', 'gmp', 'iata', 'aeo', 'c-tpat', 'sqas', 'haccp'],
                'documentation': ['document', 'paperwork', 'certificate', 'manifest', 'invoice'],
                'languages': ['multilingual', 'language', 'english', 'french', 'spanish'],
                'presence': ['physical presence', 'on-site', 'local office', 'representative']
            },
            'kpi_metrics': {
                'delivery': ['on-time', 'otif', 'delivery rate', 'lead time', 'transit time'],
                'quality': ['damage', 'loss', 'claim', 'complaint', 'error rate'],
                'service': ['fill rate', 'order accuracy', 'response time', 'resolution'],
                'system': ['uptime', 'availability', 'sla', 'performance'],
                'cost': ['cost reduction', 'savings', 'efficiency']
            }
        }
    
    def _init_patterns(self) -> Dict[str, str]:
        """Initialise les patterns regex pour extraction"""
        return {
            # Métriques avec valeurs
            'percentage_metric': r'(\d{1,3}(?:[.,]\d+)?)\s*%\s*(?:minimum|target|required|sla)?',
            'time_metric': r'(?:within|under|max|maximum)?\s*(\d+)\s*(?:hours?|days?|minutes?)',
            'temperature_range': r'([-+]?\d+)\s*(?:to|à|-)\s*([-+]?\d+)\s*°?[CF]',
            
            # Services et modes
            'transport_mode': r'(?:by|via|using|mode[:\s])?\s*(air|sea|road|rail|truck|vessel|aircraft)',
            'service_mention': r'(?:service[s]?[:\s]|require[s]?|need[s]?)\s*([^.;,]+)',
            
            # Certifications
            'certification': r'(?:certified?|certification)\s*(?:in|for|to)?\s*([A-Z]+[\w\-]*\d*)',
            'standard': r'(?:ISO|GDP|GMP|IATA|AEO|C-TPAT|SQAS|HACCP)\s*[\d\-:]*'
        }
    
    # ============================================================================
    # EXTRACTION AVANCÉE
    # ============================================================================
    
    async def _extract_single_question(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Override: Extraction spécialisée pour questions opérationnelles"""
        if not self.use_advanced:
            return await super()._extract_single_question(question_config, context, chunks)
        
        question_id = question_config.id
        
        # Router vers la stratégie appropriée
        if 'transport_mode' in question_id:
            return await self._extract_transport_modes(question_config, context, chunks)
        elif 'services' in question_id:
            return await self._extract_services(question_config, context, chunks)
        elif 'requirements' in question_id:
            return await self._extract_requirements(question_config, context, chunks)
        elif 'sla' in question_id or 'kpi' in question_id:
            return await self._extract_metrics(question_config, context, chunks)
        else:
            return await self._generic_operational_extraction(question_config, context, chunks)
    
    # ============================================================================
    # STRATÉGIES SPÉCIALISÉES PAR TYPE
    # ============================================================================
    
    async def _extract_transport_modes(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction spécialisée des modes de transport"""
        modes_found = set()
        sources = []
        
        # 1. Détection par patterns et keywords
        for mode, keywords in self.operational_catalog['transport_modes'].items():
            for chunk in chunks:
                content_lower = chunk.page_content.lower()
                for keyword in keywords:
                    if keyword in content_lower:
                        # Vérifier le contexte
                        if self._validate_mode_context(keyword, content_lower):
                            modes_found.add(mode)
                            # Extraire snippet de contexte
                            idx = content_lower.find(keyword)
                            snippet = chunk.page_content[max(0, idx-30):idx+50]
                            sources.append(ExtractionSource(
                                document=chunk.metadata.get('source', 'document'),
                                context_snippet=snippet[:50],
                                confidence=0.9
                            ))
                            break
        
        # 2. Extraction LLM pour validation et modes manqués
        llm_prompt = f"""
        {question_config.system_prompt}
        
        VALIDATION: Confirm these detected transport modes are correct:
        {list(modes_found)}
        
        Also check for any missed modes or multimodal requirements.
        Be precise - only include explicitly mentioned modes.
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=llm_prompt),
                HumanMessage(content=context[:8000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                # Fusionner avec détection
                llm_modes = parsed['answer'] if isinstance(parsed['answer'], list) else [parsed['answer']]
                modes_found.update([m.lower() for m in llm_modes if m])
                
                # Ajouter sources LLM
                if parsed.get('sources'):
                    for src in parsed['sources'][:2]:
                        sources.append(ExtractionSource(
                            document=src.get('document', 'document'),
                            context_snippet=src.get('context_snippet', ''),
                            confidence=0.85
                        ))
        except Exception as e:
            logger.debug(f"LLM validation failed: {e}")
        
        # Construire résultat
        if modes_found:
            return QuestionResult(
                question_id=question_config.id,
                answer=list(modes_found),
                sources=sources[:5],
                status=ExtractionStatus.SUCCESS,
                confidence=0.9 if len(sources) > 0 else 0.7,
                metadata={'detection_method': 'hybrid'}
            )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No transport modes found"
        )
    
    async def _extract_services(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction des services logistiques requis"""
        services_found = set()
        sources = []
        
        # 1. Détection par catalogue
        for service, keywords in self.operational_catalog['services'].items():
            for chunk in chunks:
                content_lower = chunk.page_content.lower()
                for keyword in keywords:
                    if keyword in content_lower:
                        services_found.add(service.replace('_', ' '))
                        # Une source par service suffit
                        if service not in [s.metadata.get('service') for s in sources if hasattr(s, 'metadata')]:
                            sources.append(self._create_source(chunk, keyword))
                        break
        
        # 2. Extraction LLM pour services complexes
        service_prompt = f"""
        {question_config.system_prompt}
        
        Additional guidance:
        - Look for both explicit service mentions and scope of work descriptions
        - Include value-added services if mentioned
        - Distinguish between required and optional services
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=service_prompt),
                HumanMessage(content=context[:10000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                llm_services = parsed['answer'] if isinstance(parsed['answer'], list) else [parsed['answer']]
                services_found.update(llm_services)
                
                # Sources LLM
                if parsed.get('sources'):
                    sources.extend([
                        ExtractionSource(
                            document=s.get('document', 'document'),
                            context_snippet=s.get('context_snippet', ''),
                            confidence=0.8
                        ) for s in parsed['sources'][:3]
                    ])
        except Exception as e:
            logger.debug(f"Service extraction failed: {e}")
        
        if services_found:
            # Nettoyer et dédupliquer
            cleaned_services = self._clean_services_list(list(services_found))
            
            return QuestionResult(
                question_id=question_config.id,
                answer=cleaned_services,
                sources=sources[:5],
                status=ExtractionStatus.SUCCESS,
                confidence=min(0.95, 0.7 + len(services_found) * 0.05),
                metadata={'services_count': len(cleaned_services)}
            )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No services identified"
        )
    
    async def _extract_requirements(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction des exigences opérationnelles"""
        requirements = defaultdict(list)
        sources = []
        
        # 1. Détection structurée par catégorie
        for req_type, keywords in self.operational_catalog['requirements'].items():
            for chunk in chunks:
                content = chunk.page_content
                content_lower = content.lower()
                
                for keyword in keywords:
                    if keyword in content_lower:
                        # Extraire le contexte détaillé
                        requirement_detail = self._extract_requirement_detail(
                            content, keyword, req_type
                        )
                        if requirement_detail:
                            requirements[req_type].append(requirement_detail)
                            sources.append(self._create_source(chunk, keyword))
        
        # 2. Détection de certifications par pattern
        cert_pattern = self.patterns['certification']
        for chunk in chunks:
            cert_matches = re.finditer(cert_pattern, chunk.page_content, re.IGNORECASE)
            for match in cert_matches:
                cert = match.group(1)
                if self._is_valid_certification(cert):
                    requirements['certifications'].append(cert)
        
        # 3. Extraction LLM pour requirements complexes
        req_prompt = f"""
        {question_config.system_prompt}
        
        Focus on:
        - Technical requirements (IT, systems, equipment)
        - Operational constraints (temperature, timing, handling)
        - Compliance requirements (certifications, standards)
        - Resource requirements (personnel, facilities)
        """
        
        try:
            response = await self.metrics_llm.ainvoke([
                SystemMessage(content=req_prompt),
                HumanMessage(content=context[:10000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                llm_reqs = parsed['answer'] if isinstance(parsed['answer'], list) else [parsed['answer']]
                requirements['other'].extend(llm_reqs)
                
                if parsed.get('sources'):
                    sources.extend([
                        ExtractionSource(
                            document=s.get('document', 'document'),
                            context_snippet=s.get('context_snippet', ''),
                            confidence=0.85
                        ) for s in parsed['sources'][:2]
                    ])
        except Exception as e:
            logger.debug(f"Requirements extraction failed: {e}")
        
        # Formater le résultat
        if requirements:
            formatted_reqs = self._format_requirements(dict(requirements))
            
            return QuestionResult(
                question_id=question_config.id,
                answer=formatted_reqs,
                sources=sources[:5],
                status=ExtractionStatus.SUCCESS,
                confidence=0.85,
                metadata={
                    'requirement_categories': list(requirements.keys()),
                    'total_requirements': sum(len(v) for v in requirements.values())
                }
            )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No operational requirements found"
        )
    
    async def _extract_metrics(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction des SLA/KPI avec valeurs cibles"""
        metrics = []
        sources = []
        
        # 1. Extraction par patterns de métriques
        for chunk in chunks:
            content = chunk.page_content
            
            # Pourcentages (ex: 95% on-time delivery)
            pct_matches = re.finditer(self.patterns['percentage_metric'], content)
            for match in pct_matches:
                metric_context = content[max(0, match.start()-50):match.end()+50]
                metric = self._parse_metric_context(metric_context, match.group(1), '%')
                if metric:
                    metrics.append(metric)
                    sources.append(self._create_source(chunk, match.group(0)))
            
            # Métriques temporelles (ex: within 24 hours)
            time_matches = re.finditer(self.patterns['time_metric'], content)
            for match in time_matches:
                metric_context = content[max(0, match.start()-50):match.end()+50]
                metric = self._parse_metric_context(metric_context, match.group(1), 'time')
                if metric:
                    metrics.append(metric)
                    sources.append(self._create_source(chunk, match.group(0)))
        
        # 2. Détection par keywords KPI
        for kpi_type, keywords in self.operational_catalog['kpi_metrics'].items():
            for chunk in chunks:
                content_lower = chunk.page_content.lower()
                for keyword in keywords:
                    if keyword in content_lower:
                        # Chercher une valeur associée
                        kpi_detail = self._extract_kpi_detail(chunk.page_content, keyword)
                        if kpi_detail:
                            metrics.append(kpi_detail)
                            sources.append(self._create_source(chunk, keyword))
        
        # 3. Extraction LLM pour KPIs complexes
        kpi_prompt = f"""
        {question_config.system_prompt}
        
        Extract ONLY quantifiable metrics with their target values.
        Include:
        - The metric name
        - The target value or threshold
        - Whether it's minimum/maximum/target
        
        Examples: "95% on-time delivery", "customs clearance within 4 hours"
        """
        
        try:
            response = await self.metrics_llm.ainvoke([
                SystemMessage(content=kpi_prompt),
                HumanMessage(content=context[:10000])
            ])
            
            parsed = self._parse_json_response(response.content)
            if parsed and parsed.get('answer'):
                llm_metrics = parsed['answer'] if isinstance(parsed['answer'], list) else [parsed['answer']]
                
                # Dédupliquer avec métriques existantes
                for metric in llm_metrics:
                    if not self._is_duplicate_metric(metric, metrics):
                        metrics.append(metric)
                
                if parsed.get('sources'):
                    sources.extend([
                        ExtractionSource(
                            document=s.get('document', 'document'),
                            context_snippet=s.get('context_snippet', ''),
                            confidence=0.9
                        ) for s in parsed['sources'][:2]
                    ])
        except Exception as e:
            logger.debug(f"KPI extraction failed: {e}")
        
        if metrics:
            # Structurer les métriques
            structured_metrics = self._structure_metrics(metrics)
            
            return QuestionResult(
                question_id=question_config.id,
                answer=structured_metrics,
                sources=sources[:5],
                status=ExtractionStatus.SUCCESS,
                confidence=0.9,
                metadata={
                    'metrics_count': len(metrics),
                    'has_targets': any('target' in str(m).lower() for m in metrics)
                }
            )
        
        return QuestionResult(
            question_id=question_config.id,
            answer=None,
            status=ExtractionStatus.FAILED,
            error="No SLA/KPI metrics found"
        )
    
    async def _generic_operational_extraction(
        self,
        question_config,
        context: str,
        chunks: List[Document]
    ) -> QuestionResult:
        """Extraction générique pour questions opérationnelles"""
        # Utiliser la méthode parent avec enrichissement
        enhanced_prompt = f"""
        {question_config.system_prompt}
        
        OPERATIONAL CONTEXT:
        Focus on concrete, actionable requirements.
        Distinguish between mandatory and optional elements.
        Include specific values, thresholds, or standards when mentioned.
        """
        
        question_config.system_prompt = enhanced_prompt
        return await super()._extract_single_question(question_config, context, chunks)
    
    # ============================================================================
    # MÉTHODES UTILITAIRES
    # ============================================================================
    
    def _validate_mode_context(self, keyword: str, content: str) -> bool:
        """Valide que le keyword est bien un mode de transport dans le contexte"""
        # Éviter les faux positifs
        false_positive_contexts = [
            'email', 'mail address', 'airline code', 'port of', 'airport code'
        ]
        
        keyword_idx = content.find(keyword)
        surrounding = content[max(0, keyword_idx-30):keyword_idx+30].lower()
        
        return not any(fp in surrounding for fp in false_positive_contexts)
    
    def _create_source(self, chunk: Document, keyword: str) -> ExtractionSource:
        """Crée une source d'extraction"""
        content = chunk.page_content
        idx = content.lower().find(keyword.lower())
        snippet = content[max(0, idx-20):idx+30] if idx >= 0 else keyword
        
        return ExtractionSource(
            document=chunk.metadata.get('source', 'document'),
            context_snippet=snippet[:50],
            confidence=0.85
        )
    
    def _clean_services_list(self, services: List[str]) -> List[str]:
        """Nettoie et déduplique la liste des services"""
        cleaned = []
        seen = set()
        
        for service in services:
            # Normaliser
            normalized = service.lower().strip().replace('_', ' ')
            
            # Éviter les doublons sémantiques
            if normalized not in seen and len(normalized) > 2:
                cleaned.append(service)
                seen.add(normalized)
                
                # Ajouter les variantes pour éviter duplications
                seen.add(normalized.replace(' ', '_'))
                seen.add(normalized.replace('-', ' '))
        
        return cleaned
    
    def _extract_requirement_detail(
        self,
        content: str,
        keyword: str,
        req_type: str
    ) -> Optional[str]:
        """Extrait le détail d'une exigence"""
        keyword_idx = content.lower().find(keyword.lower())
        if keyword_idx < 0:
            return None
        
        # Extraire la phrase complète
        start = content.rfind('.', 0, keyword_idx) + 1
        end = content.find('.', keyword_idx)
        if end == -1:
            end = keyword_idx + 100
        
        requirement = content[start:end].strip()
        
        # Filtrer les requirements trop courts ou non pertinents
        if len(requirement) < 10 or len(requirement) > 500:
            return None
        
        # Formater selon le type
        if req_type == 'certifications':
            # Extraire juste la certification
            cert_match = re.search(r'(ISO[\s\-]?\d+|GDP|GMP|IATA|AEO)', requirement, re.IGNORECASE)
            if cert_match:
                return cert_match.group(1)
        elif req_type == 'temperature':
            # Inclure la plage de température
            temp_match = re.search(r'[-+]?\d+\s*°?[CF]', requirement)
            if temp_match:
                return f"Temperature control: {requirement[:100]}"
        
        return requirement[:200] if len(requirement) > 200 else requirement
    
    def _is_valid_certification(self, cert: str) -> bool:
        """Vérifie si une certification est valide"""
        valid_certs = [
            'ISO', 'GDP', 'GMP', 'IATA', 'AEO', 'C-TPAT', 
            'SQAS', 'HACCP', 'OHSAS', 'TAPA', 'CEIV'
        ]
        return any(vc in cert.upper() for vc in valid_certs)
    
    def _format_requirements(self, requirements: Dict) -> List[str]:
        """Formate les exigences pour la sortie"""
        formatted = []
        
        for category, items in requirements.items():
            if items:
                # Dédupliquer
                unique_items = list(set(items))
                
                # Formater selon la catégorie
                if category == 'certifications':
                    formatted.extend([f"Certification: {item}" for item in unique_items])
                elif category == 'temperature':
                    formatted.extend(unique_items)  # Déjà formaté
                else:
                    formatted.extend(unique_items[:5])  # Limiter le nombre
        
        return formatted
    
    def _parse_metric_context(
        self,
        context: str,
        value: str,
        metric_type: str
    ) -> Optional[str]:
        """Parse le contexte d'une métrique"""
        context_lower = context.lower()
        
        # Identifier le type de métrique
        if metric_type == '%':
            if 'on-time' in context_lower or 'otif' in context_lower:
                return f"On-time delivery: {value}%"
            elif 'fill rate' in context_lower:
                return f"Fill rate: {value}%"
            elif 'accuracy' in context_lower:
                return f"Accuracy: {value}%"
            elif 'uptime' in context_lower or 'availability' in context_lower:
                return f"System availability: {value}%"
        elif metric_type == 'time':
            unit = 'hours' if 'hour' in context_lower else 'days' if 'day' in context_lower else 'minutes'
            if 'customs' in context_lower:
                return f"Customs clearance: within {value} {unit}"
            elif 'response' in context_lower:
                return f"Response time: {value} {unit}"
            elif 'delivery' in context_lower or 'transit' in context_lower:
                return f"Delivery time: {value} {unit}"
        
        # Métrique générique
        return f"{context[:50]}: {value}{metric_type}"
    
    def _extract_kpi_detail(self, content: str, keyword: str) -> Optional[str]:
        """Extrait le détail d'un KPI"""
        keyword_idx = content.lower().find(keyword.lower())
        if keyword_idx < 0:
            return None
        
        # Chercher une valeur numérique proche
        search_window = content[max(0, keyword_idx-50):keyword_idx+100]
        
        # Patterns pour valeurs
        value_patterns = [
            r'(\d{1,3}(?:[.,]\d+)?)\s*%',
            r'(\d+)\s*(?:hours?|days?|minutes?)',
            r'(?:target|minimum|maximum)[:\s]+(\d+(?:[.,]\d+)?)'
        ]
        
        for pattern in value_patterns:
            match = re.search(pattern, search_window, re.IGNORECASE)
            if match:
                return f"{keyword}: {match.group(0)}"
        
        return None
    
    def _is_duplicate_metric(self, metric: str, existing_metrics: List) -> bool:
        """Vérifie si une métrique est un doublon"""
        metric_lower = str(metric).lower()
        
        for existing in existing_metrics:
            existing_lower = str(existing).lower()
            
            # Vérification simple de similarité
            if metric_lower in existing_lower or existing_lower in metric_lower:
                return True
            
            # Vérifier les valeurs numériques identiques
            metric_numbers = re.findall(r'\d+', metric_lower)
            existing_numbers = re.findall(r'\d+', existing_lower)
            
            if metric_numbers and metric_numbers == existing_numbers:
                # Même nombre, vérifier le contexte
                common_words = set(metric_lower.split()) & set(existing_lower.split())
                if len(common_words) > 2:
                    return True
        
        return False
    
    def _structure_metrics(self, metrics: List) -> List[str]:
        """Structure et catégorise les métriques"""
        structured = {
            'delivery': [],
            'quality': [],
            'service': [],
            'operational': [],
            'other': []
        }
        
        for metric in metrics:
            metric_str = str(metric).lower()
            
            if any(kw in metric_str for kw in ['delivery', 'on-time', 'otif', 'transit']):
                structured['delivery'].append(metric)
            elif any(kw in metric_str for kw in ['damage', 'loss', 'claim', 'error']):
                structured['quality'].append(metric)
            elif any(kw in metric_str for kw in ['response', 'resolution', 'fill rate']):
                structured['service'].append(metric)
            elif any(kw in metric_str for kw in ['customs', 'temperature', 'system']):
                structured['operational'].append(metric)
            else:
                structured['other'].append(metric)
        
        # Flatten en gardant les catégories non vides
        result = []
        for category, items in structured.items():
            if items:
                result.extend(items)
        
        return result