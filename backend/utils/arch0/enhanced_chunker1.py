# utils/enhanced_chunker.py
"""
Chunker amélioré avec extraction de structure et gestion avancée des tableaux
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    PyPDFLoader,
    Docx2txtLoader
)

try:
    from unstructured.partition.auto import partition
    from unstructured.chunking.title import chunk_by_title
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    logging.warning("⚠️ Unstructured non installé. Utilisation des loaders basiques.")

logger = logging.getLogger(__name__)

class EnhancedChunker:
    """
    Chunker intelligent avec préservation du contexte sémantique et extraction de structure
    """
    
    def __init__(self, 
                 chunk_size: int = 1500,
                 chunk_overlap: int = 300,
                 use_unstructured: bool = True):
        """
        Initialise le chunker avancé
        
        Args:
            chunk_size: Taille des chunks
            chunk_overlap: Chevauchement entre chunks
            use_unstructured: Utiliser Unstructured si disponible
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_unstructured = use_unstructured and UNSTRUCTURED_AVAILABLE
        
        # Splitters adaptés selon le type de contenu
        self.splitters = {
            "narrative": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", ", ", " "],
                length_function=len
            ),
            "technical": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 1.3,  # Plus grand pour contenu technique
                chunk_overlap=chunk_overlap * 1.5,
                separators=["\n\n", "\n", ":", ";", ". ", ", "],
                length_function=len
            ),
            "table": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 2,  # Beaucoup plus grand pour les tableaux
                chunk_overlap=chunk_overlap * 2,
                separators=["\n\n", "\n", "|", "\t"],
                length_function=len
            )
        }
        
        logger.info(f"✅ EnhancedChunker initialisé (Unstructured: {self.use_unstructured})")
    
    def extract_with_structure(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extraction avec préservation de la structure du document
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            Liste d'éléments structurés
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")
        
        logger.info(f"📄 Extraction de: {file_path.name}")
        
        # Utiliser Unstructured si disponible et activé
        if self.use_unstructured:
            try:
                return self._extract_with_unstructured(file_path)
            except Exception as e:
                logger.warning(f"Échec Unstructured, bascule sur extraction basique: {e}")
        
        # Fallback sur extraction basique
        return self._extract_basic(file_path)
    
    def _extract_with_unstructured(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extraction avancée avec Unstructured
        """
        try:
            # Partition avec détection de structure
            elements = partition(
                filename=str(file_path),
                strategy="hi_res",  # Haute résolution pour meilleure détection
                infer_table_structure=True,  # Extraction des tableaux
                include_page_breaks=True,
                extract_images_in_pdf=False,  # Pour l'instant, pas d'images
                languages=["fra", "eng"]  # Support multilingue
            )
            
            structured_elements = []
            current_section = ""
            current_subsection = ""
            
            for element in elements:
                # Déterminer le type d'élément
                element_type = element.category if hasattr(element, 'category') else 'text'
                element_text = str(element)
                
                # Mise à jour des sections
                if element_type == "Title":
                    current_section = element_text
                    current_subsection = ""
                elif element_type == "Header":
                    current_subsection = element_text
                
                # Créer l'élément structuré
                element_dict = {
                    "type": element_type,
                    "text": element_text,
                    "metadata": {
                        "source": file_path.name,
                        "section": current_section,
                        "subsection": current_subsection,
                        "page": getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') else None
                    }
                }
                
                # Traitement spécial pour les tableaux
                if element_type == "Table":
                    element_dict["parsed_table"] = self._parse_table_structure(element_text)
                    element_dict["metadata"]["is_table"] = True
                
                # Détection de patterns importants
                element_dict["metadata"]["contains_volumes"] = self._detect_volumes(element_text)
                element_dict["metadata"]["contains_dates"] = self._detect_dates(element_text)
                element_dict["metadata"]["contains_amounts"] = self._detect_amounts(element_text)
                
                structured_elements.append(element_dict)
            
            logger.info(f"  → {len(structured_elements)} éléments extraits avec Unstructured")
            return structured_elements
            
        except Exception as e:
            logger.error(f"Erreur Unstructured: {e}")
            raise
    
    def _extract_basic(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extraction basique sans Unstructured
        """
        ext = file_path.suffix.lower()
        
        try:
            # Choisir le loader approprié
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(file_path))
            elif ext in [".xlsx", ".xls"]:
                # Pour Excel, on peut essayer pandas
                return self._extract_excel_with_pandas(file_path)
            else:
                raise ValueError(f"Format non supporté: {ext}")
            
            # Charger le document
            pages = loader.load()
            
            structured_elements = []
            for page_num, page in enumerate(pages, 1):
                # Découper le contenu en sections logiques
                sections = self._split_into_sections(page.page_content)
                
                for section_num, section_text in enumerate(sections):
                    element = {
                        "type": self._detect_content_type(section_text),
                        "text": section_text,
                        "metadata": {
                            "source": file_path.name,
                            "page": page_num,
                            "section_num": section_num,
                            "contains_volumes": self._detect_volumes(section_text),
                            "contains_dates": self._detect_dates(section_text),
                            "contains_amounts": self._detect_amounts(section_text)
                        }
                    }
                    
                    # Détecter si c'est un tableau
                    if self._is_table(section_text):
                        element["parsed_table"] = self._parse_table_structure(section_text)
                        element["metadata"]["is_table"] = True
                    
                    structured_elements.append(element)
            
            logger.info(f"  → {len(structured_elements)} éléments extraits (basique)")
            return structured_elements
            
        except Exception as e:
            logger.error(f"Erreur extraction basique: {e}")
            return []
    
    def _extract_excel_with_pandas(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extraction spéciale pour fichiers Excel avec pandas
        """
        try:
            excel_file = pd.ExcelFile(file_path)
            structured_elements = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Convertir le DataFrame en texte structuré
                table_text = df.to_string()
                
                element = {
                    "type": "table",
                    "text": table_text,
                    "metadata": {
                        "source": file_path.name,
                        "sheet": sheet_name,
                        "is_table": True,
                        "rows": len(df),
                        "columns": list(df.columns)
                    },
                    "parsed_table": df.to_dict('records')
                }
                
                # Détecter les patterns importants dans les données
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        element["metadata"][f"has_numeric_{col}"] = True
                
                structured_elements.append(element)
            
            return structured_elements
            
        except Exception as e:
            logger.error(f"Erreur extraction Excel: {e}")
            return []
    
    def _split_into_sections(self, text: str) -> List[str]:
        """
        Découpe intelligente en sections
        """
        # Patterns de séparation de sections
        section_patterns = [
            r'\n\d+\.\s+[A-Z]',  # Sections numérotées
            r'\n[A-Z]{2,}\s*\n',  # Titres en majuscules
            r'\n={3,}\n',  # Séparateurs
            r'\n-{3,}\n'
        ]
        
        sections = []
        current_section = []
        lines = text.split('\n')
        
        for line in lines:
            # Vérifier si c'est un début de section
            is_section_start = any(re.match(pattern, '\n' + line) for pattern in section_patterns)
            
            if is_section_start and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections if sections else [text]
    
    def _detect_content_type(self, text: str) -> str:
        """
        Détecte le type de contenu
        """
        text_lower = text.lower()
        
        # Détection basée sur les mots-clés
        if any(word in text_lower for word in ['tableau', 'table', '|', '\t\t']):
            return 'table'
        elif any(word in text_lower for word in ['article', 'clause', 'section']):
            return 'legal'
        elif any(word in text_lower for word in ['prix', 'tarif', 'coût', 'eur', 'usd']):
            return 'financial'
        elif any(word in text_lower for word in ['volume', 'teu', 'tonnage', 'palette']):
            return 'volume'
        elif any(word in text_lower for word in ['date', 'délai', 'échéance']):
            return 'timeline'
        else:
            return 'narrative'
    
    def _is_table(self, text: str) -> bool:
        """
        Détecte si le texte est un tableau
        """
        indicators = [
            text.count('|') > 3,  # Pipes pour colonnes
            text.count('\t') > 5,  # Tabs pour colonnes
            bool(re.search(r'\d+\s+\d+\s+\d+', text)),  # Séries de nombres
            'total' in text.lower() and any(c.isdigit() for c in text)
        ]
        return sum(indicators) >= 2
    
    def _parse_table_structure(self, table_text: str) -> Optional[pd.DataFrame]:
        """
        Parse un tableau en DataFrame
        """
        try:
            lines = table_text.strip().split('\n')
            
            # Détecter le séparateur
            if '|' in table_text:
                separator = '|'
            elif '\t' in table_text:
                separator = '\t'
            else:
                separator = r'\s{2,}'  # Au moins 2 espaces
            
            # Parser les lignes
            data = []
            headers = None
            
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = re.split(separator, line.strip())
                parts = [p.strip() for p in parts if p.strip()]
                
                if not headers:
                    headers = parts
                else:
                    data.append(parts)
            
            if headers and data:
                # Ajuster les longueurs si nécessaire
                max_len = max(len(headers), max(len(row) for row in data))
                headers += [''] * (max_len - len(headers))
                
                for i in range(len(data)):
                    data[i] += [''] * (max_len - len(data[i]))
                
                df = pd.DataFrame(data, columns=headers)
                return df
                
        except Exception as e:
            logger.debug(f"Impossible de parser le tableau: {e}")
            return None
    
    def _detect_volumes(self, text: str) -> bool:
        """
        Détecte la présence de volumes
        """
        volume_patterns = [
            r'\d+[\s,\.]*\d*\s*(?:TEU|FEU|m3|m³|tonnes?|palettes?)',
            r'volume|quantity|quantité'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in volume_patterns)
    
    def _detect_dates(self, text: str) -> bool:
        """
        Détecte la présence de dates
        """
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(?:janvier|février|mars|avril|mai|juin|juillet|août|septemb|octob|novemb|decemb|january|february|march|april|may|june|july|august)',
            r'(?:date|deadline|échéance|délai)'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns)
    
    def _detect_amounts(self, text: str) -> bool:
        """
        Détecte la présence de montants
        """
        amount_patterns = [
            r'\d+[\s,\.]*\d*\s*(?:€|EUR|USD|\$|GBP|£)',
            r'(?:prix|tarif|coût|montant|budget)'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in amount_patterns)
    
    def process_folder(self, folder_path: Path) -> List[Dict[str, Any]]:
        """
        Traite tous les fichiers d'un dossier
        
        Args:
            folder_path: Chemin du dossier
            
        Returns:
            Liste de tous les éléments extraits
        """
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Dossier invalide: {folder_path}")
        
        all_elements = []
        supported_extensions = {'.pdf', '.docx', '.doc', '.xlsx', '.xls'}
        
        for file_path in folder_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    elements = self.extract_with_structure(file_path)
                    all_elements.extend(elements)
                except Exception as e:
                    logger.error(f"Erreur sur {file_path.name}: {e}")
        
        logger.info(f"✅ {len(all_elements)} éléments extraits de {folder_path}")
        return all_elements
    
if __name__ == "__main__":
    print("🚀 Enhanced Chunker - Démarrage")
    
    try:
        # Créer l'instance
        chunker = EnhancedChunker()
        print("✅ Chunker initialisé")
        
        # Dossier à traiter (modifiez selon vos besoins)
        folder_path = Path(r"C:\Users\serigne.faye\OneDrive - MSC\Bureau\Dossier de travail\MISSIONS\B PILOTAGE PROJETS INNO\2.6 PoC Power Platform\9 Agent Analyse AO\Z DOCUMENTATION\Reckitt Tender documentation 1")  # ou Path("./test_documents")
        
        # Créer le dossier de test s'il n'existe pas
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
            print(f"📁 Dossier créé: {folder_path}")
            print("⚠️  Ajoutez des documents PDF/Word/Excel dans ce dossier")
        
        print(f"📂 Traitement du dossier: {folder_path.absolute()}")
        
        # Lister les fichiers détectés
        supported_extensions = {'.pdf', '.docx', '.doc', '.xlsx', '.xls'}
        files_found = [f for f in folder_path.rglob('*') 
                      if f.suffix.lower() in supported_extensions]
        
        if not files_found:
            print("⚠️  Aucun fichier supporté trouvé!")
            print(f"   Formats supportés: {', '.join(supported_extensions)}")
            print(f"   Dossier scanné: {folder_path.absolute()}")
        else:
            print(f"📄 {len(files_found)} fichier(s) détecté(s):")
            for f in files_found:
                print(f"   - {f.name}")
        
        # Traiter les fichiers
        print("\n🔄 Début de l'extraction...")
        elements = chunker.process_folder(folder_path)
        
        print(f"\n✅ Extraction terminée!")
        print(f"📊 {len(elements)} éléments extraits au total")
        
        # Statistiques par type
        if elements:
            types_count = {}
            for elem in elements:
                elem_type = elem.get('type', 'unknown')
                types_count[elem_type] = types_count.get(elem_type, 0) + 1
            
            print("\n📈 Répartition par type:")
            for elem_type, count in types_count.items():
                print(f"   {elem_type}: {count}")
        
        # Sauvegarder les résultats
        output_file = Path("extracted_elements.json")
        if elements:
            import json
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(elements, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Résultats sauvegardés dans: {output_file.absolute()}")
        
        # Afficher quelques exemples
        if elements:
            print(f"\n🔍 Aperçu des premiers éléments:")
            for i, elem in enumerate(elements[:3]):
                print(f"\n--- Élément {i+1} ---")
                print(f"Type: {elem.get('type', 'N/A')}")
                print(f"Source: {elem.get('metadata', {}).get('source', 'N/A')}")
                print(f"Texte: {elem.get('text', '')[:100]}...")
        
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé: {e}")
    except ValueError as e:
        print(f"❌ Erreur de configuration: {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Enhanced Chunker - Terminé")