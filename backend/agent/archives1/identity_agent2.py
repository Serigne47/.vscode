# agents/identity_agent.py
"""
Agent spécialisé dans l'extraction de l'identité de l'appel d'offres
"""
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from .base_agent1 import BaseExtractionAgent
import re
from datetime import datetime

class IdentityExtractionAgent(BaseExtractionAgent):
    """
    Agent pour extraire les informations d'identification de l'AO
    """
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Prompt spécialisé pour l'extraction d'identité
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Tu es un expert en analyse d'appels d'offres.
            
            MISSION: Extraire TOUTES les informations d'identification de l'appel d'offres.
            
            INFORMATIONS À EXTRAIRE:
            1. CLIENT/ÉMETTEUR:
               - Nom complet de l'entreprise
               - Groupe/Maison mère si mentionné
               - Division/Département émetteur
               - Contact principal (nom, email, téléphone)
            
            2. RÉFÉRENCE AO:
               - Numéro de référence officiel
               - Titre/Intitulé de l'AO
               - Type de procédure (ouvert, restreint, négocié)
               - Numéro de lot si applicable
            
            3. CANAUX DE RÉPONSE:
               - Plateforme de soumission (nom, URL)
               - Email de soumission
               - Adresse physique si remise papier
               - Format requis (PDF, plateforme spécifique)
            
            4. LIVRABLES ATTENDUS:
               - Liste des documents à fournir
               - Format requis pour chaque document
               - Langue de réponse exigée
               - Nombre d'exemplaires si applicable
            
            5. TYPE DE MARCHÉ:
               - Accord-cadre / Marché ponctuel
               - Mono-attributaire / Multi-attributaire
               - Marché à bons de commande
               - Exclusivité ou non
            
            6. PÉRIMÈTRE:
               - Pays/Régions concernés
               - Sites/Entrepôts mentionnés
               - Corridors de transport
            
            RÈGLES D'EXTRACTION:
            - Extraire les informations EXACTES du document
            - Indiquer "non_specifie" si l'information n'est pas trouvée
            - Pour les contacts, extraire tous les détails disponibles
            - Pour les références, prendre la référence la plus officielle
            
            FORMAT JSON STRICT:
            {
                "client": {
                    "nom": "string",
                    "groupe": "string ou null",
                    "division": "string ou null",
                    "contact": {
                        "nom": "string ou null",
                        "email": "string ou null",
                        "telephone": "string ou null"
                    }
                },
                "reference_ao": {
                    "numero": "string",
                    "titre": "string",
                    "type_procedure": "string",
                    "lot": "string ou null"
                },
                "canal_reponse": {
                    "plateforme": "string ou null",
                    "email": "string ou null",
                    "adresse": "string ou null",
                    "format": "string"
                },
                "livrables": [
                    {
                        "document": "string",
                        "format": "string",
                        "obligatoire": true/false
                    }
                ],
                "type_marche": {
                    "nature": "accord-cadre|ponctuel|bons_commande",
                    "attribution": "mono|multi",
                    "exclusif": true/false/null
                },
                "perimetre": {
                    "pays": ["liste des pays"],
                    "regions": ["liste des régions"],
                    "sites": ["liste des sites"],
                    "corridors": ["liste des corridors"]
                }
            }
            """),
            HumanMessage(content="Contexte du document:\n{context}")
        ])
    
    def _define_validation_patterns(self) -> Dict[str, str]:
        """
        Patterns regex pour validation des données d'identité
        """
        return {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}',
            "reference": r'(?:RFP|AO|TENDER|APPEL)[\s\-:#]*([A-Z0-9\-\/]+)',
            "company": r'(?:Client|Émetteur|Entreprise|Société|Company|Donneur d\'ordre)[\s:]*([A-Z][A-Za-z\s\-&]+)',
            "platform": r'(?:plateforme|platform|portail|portal)[\s:]*([A-Za-z0-9\-\.]+)',
            "country": r'\b(?:France|Belgique|Allemagne|Espagne|Italie|Portugal|UK|Germany|Spain|Italy|Belgium|Netherlands)\b'
        }
    
    def _post_process(self, raw_extraction: Dict) -> Dict:
        """
        Post-traitement spécifique pour les données d'identité
        """
        processed = raw_extraction.copy()
        
        # Nettoyer les emails
        if 'client' in processed and 'contact' in processed['client']:
            contact = processed['client']['contact']
            if contact and 'email' in contact and contact['email']:
                # Valider format email
                email = contact['email']
                if not re.match(self.validation_patterns['email'], email):
                    contact['email'] = None
        
        # Normaliser les pays
        if 'perimetre' in processed and 'pays' in processed['perimetre']:
            pays = processed['perimetre']['pays']
            if isinstance(pays, list):
                # Normalisation des noms de pays
                country_mapping = {
                    'uk': 'United Kingdom',
                    'gb': 'United Kingdom',
                    'deutschland': 'Germany',
                    'allemagne': 'Germany',
                    'espagne': 'Spain',
                    'españa': 'Spain'
                }
                normalized = []
                for p in pays:
                    p_lower = p.lower()
                    normalized.append(country_mapping.get(p_lower, p))
                processed['perimetre']['pays'] = normalized
        
        # Déterminer le type de marché si non spécifié
        if 'type_marche' not in processed or not processed['type_marche']:
            processed['type_marche'] = self._infer_market_type(raw_extraction)
        
        # Enrichir avec les patterns trouvés
        if hasattr(self, '_last_pattern_results'):
            if 'reference' in self._last_pattern_results:
                if 'reference_ao' not in processed:
                    processed['reference_ao'] = {}
                if 'numero' not in processed['reference_ao']:
                    processed['reference_ao']['numero'] = self._last_pattern_results['reference'][0]
        
        return processed
    
    def _infer_market_type(self, data: Dict) -> Dict:
        """
        Infère le type de marché à partir du contexte
        """
        market_type = {
            "nature": "non_specifie",
            "attribution": "non_specifie",
            "exclusif": None
        }
        
        # Analyse du contexte pour déterminer le type
        context_str = str(data).lower()
        
        if 'accord-cadre' in context_str or 'framework' in context_str:
            market_type["nature"] = "accord-cadre"
        elif 'ponctuel' in context_str or 'one-time' in context_str:
            market_type["nature"] = "ponctuel"
        elif 'bons de commande' in context_str:
            market_type["nature"] = "bons_commande"
        
        if 'exclusif' in context_str or 'exclusive' in context_str:
            market_type["exclusif"] = True
        elif 'non exclusif' in context_str or 'non-exclusive' in context_str:
            market_type["exclusif"] = False
        
        return market_type
    
    def _pattern_extract(self, chunks) -> Dict:
        """
        Override pour stocker les résultats des patterns
        """
        results = super()._pattern_extract(chunks)
        self._last_pattern_results = results
        return results