"""Document loading and processing for various file types"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib
from datetime import datetime

# Security
class SecurityError(Exception):
    """Raised when security violation is detected"""
    pass

# Document processing imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader
)
from bs4 import BeautifulSoup
import requests
import re

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document data structure"""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    chunks: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for document"""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        return f"doc_{content_hash}"

class DocumentLoader:
    """Loads and processes various document types"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """Initialize document loader"""
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for text splitting
        if separators is None:
            self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        else:
            self.separators = separators
        
        # Initialize text splitters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )
        
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Supported file extensions
        self.supported_extensions = {
            '.pdf': self.load_pdf,
            '.txt': self.load_text,
            '.md': self.load_markdown,
            '.docx': self.load_docx,
            '.doc': self.load_docx,
            '.html': self.load_html
        }
        
        logger.info(f"DocumentLoader initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def load_file(self, file_path: Union[str, Path]) -> Document:
        """Load a document from file"""
        
        file_path = Path(file_path)
        
        # Security: Prevent path traversal attacks
        safe_path = file_path.resolve()
        
        # Check if path is within allowed directories
        allowed_dirs = [
            Path.cwd(),
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path("/tmp"),
        ]
        
        # Python 3.9+ has is_relative_to, older versions need manual check
        is_safe = False
        for allowed_dir in allowed_dirs:
            if not allowed_dir.exists():
                continue
            try:
                # Try Python 3.9+ method
                if hasattr(safe_path, 'is_relative_to'):
                    if safe_path.is_relative_to(allowed_dir):
                        is_safe = True
                        break
                else:
                    # Fallback for older Python
                    try:
                        safe_path.relative_to(allowed_dir)
                        is_safe = True
                        break
                    except ValueError:
                        continue
            except Exception:
                continue
        
        if not is_safe:
            # Additional check for common directories
            if safe_path.is_file() and safe_path.parent == Path.cwd():
                is_safe = True
            elif ".." in str(file_path):
                raise SecurityError(f"Path traversal attempt detected: {file_path}")
            elif not is_safe:
                raise SecurityError(f"Access denied: Path '{safe_path}' is outside allowed directories")
        
        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: {safe_path}")
        
        # Get file extension
        ext = file_path.suffix.lower()
        
        if ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Load using appropriate loader
        loader_func = self.supported_extensions[ext]
        document = loader_func(file_path)
        
        # Add common metadata
        document.metadata.update({
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": ext,
            "loaded_at": datetime.now().isoformat(),
            "file_size": file_path.stat().st_size
        })
        
        return document
    
    def load_pdf(self, file_path: Path) -> Document:
        """Load PDF document"""
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            # Combine all pages
            full_text = "\n\n".join([page.page_content for page in pages])
            
            # Extract metadata
            metadata = {
                "page_count": len(pages),
                "type": "pdf"
            }
            
            # Try to extract additional metadata from first page
            if pages:
                metadata.update(pages[0].metadata)
            
            return Document(content=full_text, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise
    
    def load_text(self, file_path: Path) -> Document:
        """Load plain text document"""
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            
            if documents:
                return Document(
                    content=documents[0].page_content,
                    metadata={"type": "text"}
                )
            
            # Fallback to direct reading
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return Document(content=content, metadata={"type": "text"})
            
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            raise
    
    def load_markdown(self, file_path: Path) -> Document:
        """Load Markdown document"""
        try:
            # Try UnstructuredMarkdownLoader first
            try:
                loader = UnstructuredMarkdownLoader(str(file_path))
                documents = loader.load()
                
                if documents:
                    return Document(
                        content=documents[0].page_content,
                        metadata={"type": "markdown"}
                    )
            except:
                pass
            
            # Fallback to direct reading
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract headers as metadata
            headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
            
            return Document(
                content=content,
                metadata={
                    "type": "markdown",
                    "headers": ", ".join(headers[:10]) if headers else ""  # First 10 headers as string
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load Markdown file {file_path}: {e}")
            raise
    
    def load_docx(self, file_path: Path) -> Document:
        """Load Word document"""
        try:
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            
            if documents:
                return Document(
                    content=documents[0].page_content,
                    metadata={"type": "docx"}
                )
            
            raise ValueError("Failed to load DOCX content")
            
        except Exception as e:
            logger.error(f"Failed to load DOCX file {file_path}: {e}")
            raise
    
    def load_html(self, file_path: Path) -> Document:
        """Load HTML document"""
        try:
            loader = UnstructuredHTMLLoader(str(file_path))
            documents = loader.load()
            
            if documents:
                content = documents[0].page_content
            else:
                # Fallback to BeautifulSoup
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    content = soup.get_text()
            
            return Document(
                content=content,
                metadata={"type": "html"}
            )
            
        except Exception as e:
            logger.error(f"Failed to load HTML file {file_path}: {e}")
            raise
    
    def load_url(self, url: str) -> Document:
        """Load content from URL"""
        try:
            # Fetch content
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            content = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract metadata
            title = soup.find("title")
            title_text = title.string if title else "Unknown"
            
            return Document(
                content=content,
                metadata={
                    "type": "web",
                    "url": url,
                    "title": title_text,
                    "fetched_at": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load URL {url}: {e}")
            raise
    
    def chunk_document(
        self,
        document: Document,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """Split document into chunks"""
        
        # Use custom or default parameters
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        # Choose splitter based on document type
        if document.metadata.get("type") == "markdown":
            splitter = self.markdown_splitter
        else:
            splitter = self.text_splitter
        
        # Split text
        chunks = splitter.split_text(document.content)
        
        # Create Document objects for each chunk
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_total": len(chunks),
                "parent_doc_id": document.doc_id
            })
            
            chunked_docs.append(
                Document(
                    content=chunk,
                    metadata=chunk_metadata
                )
            )
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunked_docs
    
    def load_directory(
        self,
        directory: Union[str, Path],
        glob_pattern: str = "**/*",
        recursive: bool = True
    ) -> List[Document]:
        """Load all documents from a directory"""
        
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        
        # Find all files matching pattern
        if recursive:
            files = directory.glob(glob_pattern)
        else:
            files = directory.glob(glob_pattern.replace("**/", ""))
        
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self.load_file(file_path)
                    documents.append(doc)
                    logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    def process_documents(
        self,
        documents: Union[Document, List[Document]],
        chunk: bool = True
    ) -> List[Document]:
        """Process documents with chunking"""
        
        if isinstance(documents, Document):
            documents = [documents]
        
        processed = []
        
        for doc in documents:
            if chunk:
                chunks = self.chunk_document(doc)
                processed.extend(chunks)
            else:
                processed.append(doc)
        
        return processed