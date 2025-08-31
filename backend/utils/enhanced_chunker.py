# utils/enhanced_chunker.py
"""
Chunker intelligent optimisé pour l'analyse par LLM
Préserve le contexte, la structure et les relations sémantiques
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
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
    Chunker intelligent pour maximiser la compréhension des LLMs
    Philosophie : Grands chunks contextuels > Petits fragments
    """
    
    # Tailles optimales pour GPT-4
    DEFAULT_CHUNK_SIZE = 2000  # 8k caractères pour contexte riche
    DEFAULT_OVERLAP = 250     # Overlap généreux pour continuité
    MAX_CHUNK_SIZE = 3000     # Maximum pour chunks très importants
    FULL_DOC_THRESHOLD = 3000 # Si < 3k, garder document entier
    
    def __init__(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_OVERLAP,
        use_unstructured: bool = True,
        preserve_structure: bool = True,
        intelligent_mode: bool = True
    ):
        """
        Initialise le chunker intelligent
        
        Args:
            chunk_size: Taille cible des chunks (2000 par défaut)
            chunk_overlap: Chevauchement entre chunks (300 par défaut)
            use_unstructured: Utiliser Unstructured si disponible
            preserve_structure: Préserver la structure du document
            intelligent_mode: Mode intelligent (sections complètes, contexte préservé)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_unstructured = use_unstructured and UNSTRUCTURED_AVAILABLE
        self.preserve_structure = preserve_structure
        self.intelligent_mode = intelligent_mode
        
        # Splitters selon stratégie
        self.splitters = {
            # Pour sections narratives : préserver les paragraphes
            "narrative": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n\n", "\n\n", "\n", ". "],  # Priorité aux doubles sauts
                length_function=len,
                is_separator_regex=False
            ),
            # Pour contenu structuré : grandes sections
            "structured": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 1.5,  # Plus grand pour structuré
                chunk_overlap=chunk_overlap * 1.5,
                separators=["\n\n\n", "\n\n", "\n"],  # Ne pas couper les phrases
                length_function=len
            ),
            # Pour tableaux : jamais couper
            "table": RecursiveCharacterTextSplitter(
                chunk_size=self.MAX_CHUNK_SIZE,  # Maximum pour tableaux
                chunk_overlap=0,  # Pas d'overlap pour tableaux
                separators=["\n\n\n"],  # Seulement si vraiment nécessaire
                length_function=len
            )
        }
        
        logger.info(f"✅ EnhancedChunker initialisé - Mode: {'Intelligent' if intelligent_mode else 'Standard'}")
        logger.info(f"   Chunk size: {chunk_size} | Overlap: {chunk_overlap}")
    
    # ============================================================================
    # MÉTHODE PRINCIPALE : EXTRACTION INTELLIGENTE
    # ============================================================================
    
    def extract_for_llm(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extraction optimisée pour analyse LLM
        Retourne des sections larges et contextuelles
        
        Args:
            file_path: Chemin du fichier à traiter
            
        Returns:
            Liste de sections/chunks intelligents
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {file_path}")
        
        logger.info(f"📄 Extraction intelligente de: {file_path.name}")
        
        # 1. Charger le document complet
        full_content = self._load_full_document(file_path)
        
        if not full_content:
            logger.error(f"Impossible de charger {file_path.name}")
            return []
        
        # 2. Analyser la taille du document
        doc_size = len(full_content)
        logger.info(f"   Taille du document: {doc_size:,} caractères")
        
        # 3. Si petit document, le retourner entier
        if doc_size <= self.FULL_DOC_THRESHOLD:
            logger.info(f"   → Document court, extraction complète")
            return self._create_full_document_chunk(file_path, full_content)
        
        # 4. Si document large, extraction intelligente
        if self.intelligent_mode:
            logger.info(f"   → Mode intelligent: extraction par sections sémantiques")
            return self._intelligent_extraction(file_path, full_content)
        else:
            logger.info(f"   → Mode standard: chunking classique optimisé")
            return self._standard_extraction(file_path, full_content)
    
    # ============================================================================
    # EXTRACTION DOCUMENT COMPLET
    # ============================================================================
    
    def _create_full_document_chunk(
        self, 
        file_path: Path, 
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Retourne le document entier comme un seul chunk
        """
        doc_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return [{
            "type": "full_document",
            "text": content,
            "metadata": {
                "source": file_path.name,
                "doc_id": doc_hash,
                "is_complete": True,
                "chunk_strategy": "full_document",
                "total_pages": self._estimate_pages(content),
                "char_count": len(content),
                "sections": self._detect_main_sections(content)
            }
        }]
    
    # ============================================================================
    # EXTRACTION INTELLIGENTE PAR SECTIONS
    # ============================================================================
    
    def _intelligent_extraction(
        self, 
        file_path: Path, 
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Extraction intelligente qui préserve les sections sémantiques
        """
        # 1. Identifier la structure du document
        doc_structure = self._analyze_document_structure(content)
        
        # 2. Extraire les sections principales
        main_sections = self._extract_semantic_sections(content, doc_structure)
        
        # 3. Créer les chunks intelligents
        intelligent_chunks = []
        
        for section_name, section_content in main_sections.items():
            section_size = len(section_content)
            
            # Si la section est petite, la garder entière
            if section_size <= self.chunk_size * 1.5:
                intelligent_chunks.append(self._create_section_chunk(
                    file_path, section_name, section_content, is_complete=True
                ))
            
            # Si section large, diviser intelligemment
            else:
                subsections = self._split_large_section(section_content)
                for i, subsection in enumerate(subsections):
                    intelligent_chunks.append(self._create_section_chunk(
                        file_path, 
                        f"{section_name} (Part {i+1}/{len(subsections)})",
                        subsection,
                        is_complete=(len(subsections) == 1)
                    ))
        
        # 4. Ajouter les tableaux comme chunks séparés
        tables = self._extract_tables(content)
        for i, table in enumerate(tables):
            intelligent_chunks.append(self._create_table_chunk(
                file_path, f"Table_{i+1}", table
            ))
        
        logger.info(f"   ✅ {len(intelligent_chunks)} chunks intelligents créés")
        return intelligent_chunks
    
    # ============================================================================
    # ANALYSE DE STRUCTURE
    # ============================================================================
    
    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """
        Analyse la structure du document pour extraction intelligente
        """
        structure = {
            "type": "unknown",
            "has_toc": False,
            "sections": [],
            "hierarchy_depth": 0,
            "special_elements": {
                "tables": 0,
                "lists": 0,
                "headers": 0
            }
        }
        
        lines = content.split('\n')
        
        # Détecter le type de document
        content_lower = content[:5000].lower()  # Analyser le début
        
        if any(term in content_lower for term in ['tender', 'rfp', 'rfq', 'proposal']):
            structure["type"] = "tender"
        elif any(term in content_lower for term in ['contract', 'agreement', 'terms']):
            structure["type"] = "contract"
        elif any(term in content_lower for term in ['report', 'analysis', 'study']):
            structure["type"] = "report"
        
        # Identifier les sections principales
        section_patterns = [
            (r'^#{1,3}\s+(.+)$', 'markdown'),  # Markdown headers
            (r'^(\d+\.?\s+[A-Z].+)$', 'numbered'),  # 1. Section
            (r'^([A-Z][A-Z\s]{2,})$', 'capitals'),  # SECTION TITLE
            (r'^(ARTICLE|SECTION|CHAPTER)\s+\d+', 'formal')  # Formal sections
        ]
        
        for line in lines:
            for pattern, style in section_patterns:
                if re.match(pattern, line.strip()):
                    structure["sections"].append({
                        "title": line.strip(),
                        "style": style
                    })
                    break
        
        # Compter les éléments spéciaux
        structure["special_elements"]["tables"] = content.count('|') // 10  # Estimation
        structure["special_elements"]["lists"] = len(re.findall(r'^\s*[-•*]\s+', content, re.MULTILINE))
        structure["special_elements"]["headers"] = len(structure["sections"])
        
        return structure
    
    # ============================================================================
    # EXTRACTION DE SECTIONS SÉMANTIQUES
    # ============================================================================
    
    def _extract_semantic_sections(
        self, 
        content: str, 
        structure: Dict
    ) -> Dict[str, str]:
        """
        Extrait les sections sémantiques basées sur le type de document
        """
        sections = {}
        
        # Patterns pour documents de tender/RFP
        if structure["type"] == "tender":
            section_keywords = {
                "identity": ["client", "issuer", "company", "tender", "reference", "rfp", "rfq"],
                "timeline": ["date", "deadline", "submission", "calendar", "schedule", "milestone"],
                "requirements": ["requirement", "must", "shall", "mandatory", "required", "deliverable"],
                "scope": ["scope", "objective", "service", "work", "description", "overview"],
                "commercial": ["price", "cost", "payment", "invoice", "budget", "financial"],
                "legal": ["terms", "conditions", "liability", "insurance", "compliance", "gdpr"],
                "technical": ["technical", "specification", "system", "platform", "technology"],
                "evaluation": ["evaluation", "criteria", "scoring", "assessment", "selection"]
            }
        else:
            # Sections génériques
            section_keywords = {
                "introduction": ["introduction", "overview", "summary", "abstract"],
                "main_content": ["description", "detail", "content", "body"],
                "conclusion": ["conclusion", "summary", "recommendation"]
            }
        
        # Découper le contenu en sections basées sur les mots-clés
        remaining_content = content
        
        for section_name, keywords in section_keywords.items():
            section_content = self._extract_section_by_keywords(
                remaining_content, keywords
            )
            if section_content:
                sections[section_name] = section_content
        
        # Si pas de sections identifiées, diviser en parties égales
        if not sections:
            sections = self._divide_equally(content)
        
        return sections
    
    def _extract_section_by_keywords(
        self, 
        content: str, 
        keywords: List[str]
    ) -> str:
        """
        Extrait une section basée sur des mots-clés
        """
        lines = content.split('\n')
        section_lines = []
        in_section = False
        keyword_density_threshold = 2  # Au moins 2 mots-clés pour commencer
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Compter les mots-clés dans la ligne
            keyword_count = sum(1 for kw in keywords if kw in line_lower)
            
            # Commencer la section si assez de mots-clés
            if keyword_count >= keyword_density_threshold and not in_section:
                in_section = True
                # Inclure quelques lignes avant pour le contexte
                start_idx = max(0, i - 2)
                section_lines.extend(lines[start_idx:i])
            
            if in_section:
                section_lines.append(line)
                
                # Arrêter si on trouve une nouvelle section majeure
                if re.match(r'^(SECTION|ARTICLE|CHAPTER|\d+\.)\s+', line):
                    if len(section_lines) > 10:  # Si on a assez de contenu
                        break
        
        return '\n'.join(section_lines) if len(section_lines) > 5 else ""
    
    # ============================================================================
    # DIVISION INTELLIGENTE
    # ============================================================================
    
    def _split_large_section(self, content: str) -> List[str]:
        """
        Divise une grande section en sous-sections cohérentes
        """
        # Ne jamais créer de chunks trop petits
        min_chunk_size = self.chunk_size // 2
        
        # Chercher les points de division naturels
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            # Si ajouter ce paragraphe dépasse la taille cible
            if current_size + para_size > self.chunk_size and current_size >= min_chunk_size:
                # Sauvegarder le chunk actuel
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = para_size
            else:
                current_chunk.append(paragraph)
                current_size += para_size
        
        # Ajouter le dernier chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks if chunks else [content]
    
    # ============================================================================
    # EXTRACTION DE TABLEAUX
    # ============================================================================
    
    def _extract_tables(self, content: str) -> List[str]:
        """
        Extrait les tableaux du document
        """
        tables = []
        
        # Pattern pour détecter les tableaux
        table_patterns = [
            r'(\|[^\n]+\|[\n\s]*)+',  # Markdown tables
            r'(?:(?:\t[^\t\n]+)+\n)+',  # Tab-separated
            r'(?:(?:\s{2,}[^\s]+)+\n)+'  # Space-aligned
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                table_text = match.group()
                if len(table_text) > 100:  # Ignorer les petits matches
                    tables.append(table_text)
        
        return tables
    
    # ============================================================================
    # CRÉATION DE CHUNKS
    # ============================================================================
    
    def _create_section_chunk(
        self,
        file_path: Path,
        section_name: str,
        content: str,
        is_complete: bool = True
    ) -> Dict[str, Any]:
        """
        Crée un chunk pour une section
        """
        return {
            "type": "section",
            "text": content,
            "metadata": {
                "source": file_path.name,
                "section": section_name,
                "is_complete_section": is_complete,
                "chunk_strategy": "semantic_section",
                "char_count": len(content),
                "has_context": True,
                "contains_dates": self._detect_dates(content),
                "contains_amounts": self._detect_amounts(content),
                "contains_requirements": self._detect_requirements(content)
            }
        }
    
    def _create_table_chunk(
        self,
        file_path: Path,
        table_name: str,
        content: str
    ) -> Dict[str, Any]:
        """
        Crée un chunk spécial pour un tableau
        """
        return {
            "type": "table",
            "text": content,
            "metadata": {
                "source": file_path.name,
                "section": table_name,
                "is_table": True,
                "chunk_strategy": "complete_table",
                "char_count": len(content),
                "preserve_formatting": True
            }
        }
    
    # ============================================================================
    # EXTRACTION STANDARD OPTIMISÉE
    # ============================================================================
    
    def _standard_extraction(
        self, 
        file_path: Path, 
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Extraction standard mais avec chunks larges
        """
        # Utiliser le splitter adapté
        splitter = self.splitters["structured"]
        
        # Créer les chunks
        chunks = splitter.split_text(content)
        
        result_chunks = []
        for i, chunk_text in enumerate(chunks):
            result_chunks.append({
                "type": "standard",
                "text": chunk_text,
                "metadata": {
                    "source": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_strategy": "standard_large",
                    "char_count": len(chunk_text),
                    "is_complete": False  # Les chunks standard ne sont pas complets
                }
            })
        
        return result_chunks
    
    # ============================================================================
    # MÉTHODES UTILITAIRES
    # ============================================================================
    
    def _load_full_document(self, file_path: Path) -> str:
        """
        Charge le document complet en mémoire
        """
        ext = file_path.suffix.lower()
        
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                return "\n\n".join([page.page_content for page in pages])
            
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(file_path))
                doc = loader.load()
                return "\n\n".join([page.page_content for page in doc])
            
            elif ext in [".txt", ".md"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            else:
                logger.warning(f"Format non supporté: {ext}")
                return ""
                
        except Exception as e:
            logger.error(f"Erreur chargement {file_path}: {e}")
            return ""
    
    def _detect_main_sections(self, content: str) -> List[str]:
        """
        Détecte les sections principales du document
        """
        sections = []
        
        # Patterns pour identifier les sections
        section_patterns = [
            r'^(?:\d+\.?\s+)?([A-Z][A-Za-z\s]{3,50})$',  # Titres
            r'^(?:SECTION|ARTICLE|CHAPTER)\s+\d+:\s*(.+)$'  # Sections formelles
        ]
        
        for line in content.split('\n')[:100]:  # Analyser le début
            for pattern in section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    sections.append(match.group(1) if match.lastindex else line.strip())
                    break
        
        return sections[:10]  # Max 10 sections principales
    
    def _estimate_pages(self, content: str) -> int:
        """
        Estime le nombre de pages
        """
        # Approximation : ~3000 caractères par page
        return max(1, len(content) // 3000)
    
    def _divide_equally(self, content: str) -> Dict[str, str]:
        """
        Divise le contenu en parties égales
        """
        total_length = len(content)
        num_parts = max(1, total_length // self.chunk_size)
        
        sections = {}
        part_size = total_length // num_parts
        
        for i in range(num_parts):
            start = i * part_size
            end = start + part_size if i < num_parts - 1 else total_length
            sections[f"part_{i+1}"] = content[start:end]
        
        return sections
    
    def _detect_dates(self, text: str) -> bool:
        """Détecte la présence de dates"""
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(?:deadline|date|due|submission)'
        ]
        return any(re.search(pattern, text.lower()) for pattern in date_patterns)
    
    def _detect_amounts(self, text: str) -> bool:
        """Détecte la présence de montants"""
        amount_patterns = [
            r'\d+[\s,\.]*\d*\s*(?:€|EUR|USD|\$|GBP|£)',
            r'(?:price|cost|budget|payment|fee)'
        ]
        return any(re.search(pattern, text.lower()) for pattern in amount_patterns)
    
    def _detect_requirements(self, text: str) -> bool:
        """Détecte la présence d'exigences"""
        requirement_keywords = ['must', 'shall', 'required', 'mandatory', 'should', 'need to']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in requirement_keywords)
    
    # ============================================================================
    # MÉTHODE PUBLIQUE PRINCIPALE
    # ============================================================================
    
    def process_folder(
        self, 
        folder_path: Path,
        output_mode: str = "intelligent"
    ) -> List[Dict[str, Any]]:
        """
        Traite tous les fichiers d'un dossier
        
        Args:
            folder_path: Chemin du dossier
            output_mode: 'intelligent', 'full_doc', ou 'standard'
            
        Returns:
            Liste de tous les chunks extraits
        """
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Dossier invalide: {folder_path}")
        
        all_chunks = []
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
        
        files_found = list(folder_path.rglob('*'))
        files_to_process = [f for f in files_found if f.suffix.lower() in supported_extensions]
        
        logger.info(f"📁 {len(files_to_process)} fichiers à traiter")
        
        for file_path in files_to_process:
            try:
                logger.info(f"📄 Traitement de {file_path.name}")
                
                if output_mode == "intelligent":
                    chunks = self.extract_for_llm(file_path)
                elif output_mode == "full_doc":
                    content = self._load_full_document(file_path)
                    chunks = self._create_full_document_chunk(file_path, content)
                else:
                    content = self._load_full_document(file_path)
                    chunks = self._standard_extraction(file_path, content)
                
                all_chunks.extend(chunks)
                logger.info(f"   ✓ {len(chunks)} chunks créés")
                
            except Exception as e:
                logger.error(f"   ✗ Erreur sur {file_path.name}: {e}")
        
        logger.info(f"✅ Total: {len(all_chunks)} chunks créés depuis {len(files_to_process)} fichiers")
        return all_chunks


# ============================================================================
# POINT D'ENTRÉE POUR TEST
# ============================================================================

if __name__ == "__main__":
    import json
    
    print("🚀 Enhanced Chunker v2 - Mode Intelligence LLM")
    print("=" * 60)
    
    # Configuration
    folder_path = Path(r"C:\Users\serigne.faye\OneDrive - MSC\Bureau\Dossier de travail\MISSIONS\B PILOTAGE PROJETS INNO\2.6 PoC Power Platform\9 Agent Analyse AO\Z DOCUMENTATION\Reckitt Tender documentation 1")
    
    # Créer le chunker optimisé
    chunker = EnhancedChunker(
        chunk_size=2000,  # Chunks larges
        chunk_overlap=250,  # Bon overlap
        intelligent_mode=True  # Mode intelligent activé
    )
    
    print(f"📂 Dossier: {folder_path}")
    print(f"⚙️ Configuration:")
    print(f"   - Chunk size: {chunker.chunk_size:,} caractères")
    print(f"   - Overlap: {chunker.chunk_overlap:,} caractères")
    print(f"   - Mode: {'Intelligent' if chunker.intelligent_mode else 'Standard'}")
    print()
    
    try:
        # Extraction intelligente
        print("🔄 Extraction en cours...")
        chunks = chunker.process_folder(folder_path, output_mode="intelligent")
        
        # Statistiques
        print(f"\n📊 Résultats:")
        print(f"   - Chunks créés: {len(chunks)}")
        
        if chunks:
            avg_size = sum(c['metadata']['char_count'] for c in chunks) / len(chunks)
            print(f"   - Taille moyenne: {avg_size:,.0f} caractères")
            
            # Types de chunks
            types = {}
            for chunk in chunks:
                chunk_type = chunk.get('type', 'unknown')
                types[chunk_type] = types.get(chunk_type, 0) + 1
            
            print(f"\n📈 Répartition par type:")
            for chunk_type, count in types.items():
                print(f"   - {chunk_type}: {count}")
            
            # Sauvegarder
            output_file = Path("chunks_intelligent.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 Chunks sauvegardés dans: {output_file}")
            
            # Aperçu
            print(f"\n🔍 Aperçu du premier chunk:")
            first = chunks[0]
            print(f"   Type: {first['type']}")
            print(f"   Taille: {first['metadata']['char_count']:,} caractères")
            print(f"   Section: {first['metadata'].get('section', 'N/A')}")
            print(f"   Texte: {first['text'][:200]}...")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()