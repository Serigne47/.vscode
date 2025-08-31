# agents/financial_agent.py
"""
Agent spécialisé dans l'extraction des conditions financières
"""
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from .base_agent1 import BaseExtractionAgent
import re

class FinancialExtractionAgent(BaseExtractionAgent):
    """
    Agent pour extraire les conditions financières et commerciales
    """
    
    def _create_extraction_prompt(self) -> ChatPromptTemplate:
        """
        Prompt pour extraction financière
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""Tu es un expert financier spécialisé dans l'analyse d'appels d'offres.
            
            MISSION: Extraire TOUTES les conditions financières et commerciales.
            
            ÉLÉMENTS FINANCIERS À EXTRAIRE:
            
            1. MODALITÉS DE FACTURATION:
               - Fréquence (mensuelle, trimestrielle, à la prestation)
               - Mode (EDI, portail, email, papier)
               - Entité de facturation (centralisée, par site, par pays)
               - Format requis (PDF, XML, UBL, autre)
               - Pièces justificatives requises
            
            2. CONDITIONS DE PAIEMENT:
               - Délai de paiement (30, 45, 60, 90, 120 jours)
               - Base de calcul (date facture, fin de mois, réception)
               - Escompte pour paiement anticipé
               - Pénalités de retard
               - Mode de règlement (virement, LCR, autre)
            
            3. DEVISE ET CHANGE:
               - Devise(s) de facturation (EUR, USD, GBP, autre)
               - Devise de paiement
               - Taux de change applicable
               - Clause de révision monétaire
               - Hedging requis
            
            4. GARANTIES ET CAUTIONS:
               - Garantie bancaire montant/pourcentage
               - Caution de bonne exécution
               - Retenue de garantie
               - Garantie maison-mère
               - Assurance crédit requise
            
            5. RÉVISION DE PRIX:
               - Formule de révision
               - Indices de référence
               - Fréquence de révision
               - Fuel surcharge / BAF
               - CAF (Currency Adjustment Factor)
               - Indexation sur inflation
            
            6. STRUCTURE TARIFAIRE:
               - Type de tarification (forfait, unitaire, mixte)
               - Minimum de facturation
               - Dégressivité volumes
               - Tarifs par corridor/destination
               - Frais additionnels possibles
            
            7. CLAUSES FINANCIÈRES SPÉCIALES:
               - Open book / Transparence des coûts
               - Cost plus / Management fee
               - Gain sharing / Partage des économies
               - Clause de benchmarking
               - Audit des coûts
            
            8. RISQUES FINANCIERS:
               - Clause de sauvegarde
               - Plafond de responsabilité financière
               - Franchise
               - Exclusions
            
            FORMAT JSON STRICT:
            {
                "facturation": {
                    "frequence": "mensuelle|trimestrielle|autre",
                    "mode": "EDI|portail|email",
                    "entite": "centralisee|par_site",
                    "format": "PDF|XML|autre",
                    "justificatifs": ["liste des documents requis"]
                },
                "paiement": {
                    "delai_jours": 60,
                    "base_calcul": "date_facture|fin_mois|reception",
                    "escompte": "2% si paiement 10 jours",
                    "penalites_retard": "taux BCE + 10 points",
                    "mode": "virement|LCR"
                },
                "devise": {
                    "facturation": "EUR",
                    "paiement": "EUR",
                    "taux_change": "BCE J-1",
                    "clause_revision": true,
                    "hedging_requis": false
                },
                "garanties": {
                    "bancaire": {
                        "montant": 100000,
                        "pourcentage": "10% CA annuel",
                        "type": "première demande"
                    },
                    "caution_execution": true,
                    "retenue_garantie": "5%",
                    "garantie_maison_mere": false
                },
                "revision_prix": {
                    "formule": "description de la formule",
                    "indices": ["Gazole", "CNR"],
                    "frequence": "trimestrielle",
                    "fuel_surcharge": true,
                    "baf": "formule BAF",
                    "indexation": "ICC"
                },
                "structure_tarifaire": {
                    "type": "unitaire|forfait|mixte",
                    "minimum_facturation": 500,
                    "degressivite": [
                        {"seuil": 1000, "remise": "5%"},
                        {"seuil": 5000, "remise": "10%"}
                    ]
                },
                "clauses_speciales": {
                    "open_book": false,
                    "cost_plus": false,
                    "gain_sharing": "50/50",
                    "benchmarking": "annuel",
                    "audit_costs": true
                },
                "risques": {
                    "plafond_responsabilite": "1M EUR",
                    "franchise": 5000,
                    "clause_sauvegarde": true
                },
                "source_clauses": ["extraits exacts des clauses importantes"]
            }
            """),
            HumanMessage(content="Analyse ces documents et extrais toutes les conditions financières:\n{context}")
        ])
    
    def _define_validation_patterns(self) -> Dict[str, str]:
        """
        Patterns pour données financières
        """
        return {
            "payment_terms": r'(\d+)\s*(?:jours?|days?)\s*(?:nets?|fin de mois|end of month)?',
            "currency": r'\b(EUR|USD|GBP|CHF|CNY|JPY|€|\$|£)\b',
            "percentage": r'(\d+[\.,]?\d*)\s*%',
            "amount": r'(\d+[\s\.,]*\d*)\s*(?:EUR|€|USD|\$|k€|K€|M€|MEUR)',
            "discount": r'(?:escompte|remise|discount)\s*(?:de\s*)?(\d+[\.,]?\d*)\s*%',
            "penalty": r'(?:pénalité|penalty|intérêt)\s*(?:de\s*)?(?:retard\s*)?(\d+[\.,]?\d*)\s*%',
            "guarantee": r'(?:garantie|caution)\s*(?:bancaire)?\s*(?:de\s*)?(\d+[\s\.,]*\d*)',
            "revision": r'(?:révision|revision|indexation|ajustement)\s*(?:des?\s*)?prix',
            "fuel": r'(?:fuel|gazole|gasoil|BAF|bunker)\s*(?:surcharge|ajustement)?',
            "invoice": r'(?:factur|invoice)',
            "payment_delay": r'(?:délai|delay|terme)\s*(?:de\s*)?(?:paiement|payment)'
        }
    
    def _post_process(self, raw_extraction: Dict) -> Dict:
        """
        Post-traitement des données financières
        """
        processed = raw_extraction.copy()
        
        # Normalisation des délais de paiement
        if 'paiement' in processed and 'delai_jours' in processed['paiement']:
            delai = processed['paiement']['delai_jours']
            if isinstance(delai, str):
                # Extraire le nombre de jours
                match = re.search(r'(\d+)', delai)
                if match:
                    processed['paiement']['delai_jours'] = int(match.group(1))
        
        # Calcul du risque financier
        processed['risk_assessment'] = self._assess_financial_risk(processed)
        
        # Conversion des montants
        processed = self._normalize_amounts(processed)
        
        # Identification des points d'attention
        processed['attention_points'] = self._identify_attention_points(processed)
        
        return processed
    
    def _assess_financial_risk(self, data: Dict) -> Dict:
        """
        Évalue le niveau de risque financier
        """
        risk_score = 0
        risk_factors = []
        
        # Délai de paiement
        if 'paiement' in data and 'delai_jours' in data['paiement']:
            delai = data['paiement']['delai_jours']
            if isinstance(delai, int):
                if delai > 60:
                    risk_score += 2
                    risk_factors.append(f"Délai de paiement long: {delai} jours")
                elif delai > 90:
                    risk_score += 3
                    risk_factors.append(f"Délai de paiement très long: {delai} jours")
        
        # Garantie bancaire
        if 'garanties' in data and 'bancaire' in data['garanties']:
            risk_score += 2
            risk_factors.append("Garantie bancaire requise")
        
        # Open book
        if 'clauses_speciales' in data:
            if data['clauses_speciales'].get('open_book'):
                risk_score += 2
                risk_factors.append("Clause Open Book")
        
        # Révision de prix
        if 'revision_prix' in data:
            if not data['revision_prix'].get('fuel_surcharge'):
                risk_score += 1
                risk_factors.append("Pas de fuel surcharge")
        
        # Calcul du niveau de risque
        risk_level = "faible"
        if risk_score >= 7:
            risk_level = "élevé"
        elif risk_score >= 4:
            risk_level = "moyen"
        
        return {
            "score": risk_score,
            "level": risk_level,
            "factors": risk_factors
        }
    
    def _normalize_amounts(self, data: Dict) -> Dict:
        """
        Normalise tous les montants en EUR
        """
        # Table de conversion simplifiée
        exchange_rates = {
            'USD': 0.92,
            'GBP': 1.17,
            'CHF': 1.02
        }
        
        def convert_amount(amount, currency):
            if currency in exchange_rates:
                return amount * exchange_rates[currency]
            return amount
        
        # Convertir garanties
        if 'garanties' in data and 'bancaire' in data['garanties']:
            garantie = data['garanties']['bancaire']
            if 'montant' in garantie and isinstance(garantie['montant'], (int, float)):
                devise = data.get('devise', {}).get('facturation', 'EUR')
                if devise != 'EUR':
                    garantie['montant_eur'] = convert_amount(garantie['montant'], devise)
        
        return data
    
    def _identify_attention_points(self, data: Dict) -> List[str]:
        """
        Identifie les points nécessitant une attention particulière
        """
        points = []
        
        # Vérifier devise
        if 'devise' in data:
            if data['devise'].get('facturation') != 'EUR':
                points.append(f"Facturation en {data['devise']['facturation']} - risque de change")
        
        # Vérifier pénalités
        if 'paiement' in data and 'penalites_retard' in data['paiement']:
            points.append("Pénalités de retard applicables")
        
        # Vérifier minimum de facturation
        if 'structure_tarifaire' in data:
            if 'minimum_facturation' in data['structure_tarifaire']:
                min_fact = data['structure_tarifaire']['minimum_facturation']
                points.append(f"Minimum de facturation: {min_fact}")
        
        # Vérifier clauses spéciales
        if 'clauses_speciales' in data:
            if data['clauses_speciales'].get('benchmarking'):
                points.append("Clause de benchmarking - révision tarifaire possible")
        
        return points