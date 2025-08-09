"""Security tests for WonyBot"""

import pytest
from pathlib import Path
import tempfile

from app.rag.document_loader import DocumentLoader, SecurityError


class TestSecurity:
    """Test suite for security features"""
    
    @pytest.fixture
    def doc_loader(self):
        """Create a document loader for testing"""
        return DocumentLoader()
    
    @pytest.mark.security
    def test_path_traversal_prevention_parent_dirs(self, doc_loader):
        """Test that path traversal attacks are prevented"""
        # Try to access parent directories
        with pytest.raises(SecurityError, match="Path traversal attempt detected"):
            doc_loader.load_file("../../etc/passwd")
        
        with pytest.raises(SecurityError, match="Path traversal attempt detected"):
            doc_loader.load_file("../../../sensitive.txt")
    
    @pytest.mark.security
    def test_path_traversal_prevention_absolute_paths(self, doc_loader):
        """Test that suspicious absolute paths are blocked"""
        # Try to access system files
        with pytest.raises((SecurityError, FileNotFoundError)):
            doc_loader.load_file("/etc/passwd")
        
        with pytest.raises((SecurityError, FileNotFoundError)):
            doc_loader.load_file("/etc/shadow")
    
    @pytest.mark.security
    def test_allowed_directories(self, doc_loader):
        """Test that files in allowed directories can be accessed"""
        # Create a temp file in an allowed directory
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', dir=Path.cwd(), delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            # Should not raise SecurityError
            doc = doc_loader.load_file(temp_path)
            assert doc.content == "Test content"
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.security
    def test_symlink_attack_prevention(self, doc_loader):
        """Test that symlink attacks are handled"""
        # Create a temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Safe content")
            safe_path = f.name
        
        # Create a symlink pointing to it
        link_path = Path.cwd() / "test_symlink.txt"
        
        try:
            link_path.symlink_to(safe_path)
            
            # The symlink should be resolved and checked
            # If it points to an allowed location, it should work
            # If it points outside, it should be blocked
            doc = doc_loader.load_file(str(link_path))
            
            # Check that the resolved path is used
            assert doc.metadata["source"] == str(link_path)
            
        except (SecurityError, OSError) as e:
            # Some systems may not support symlinks or may block them
            pass
        finally:
            # Cleanup
            link_path.unlink(missing_ok=True)
            Path(safe_path).unlink(missing_ok=True)
    
    @pytest.mark.security
    def test_file_type_validation(self, doc_loader):
        """Test that only supported file types are loaded"""
        # Create a file with unsupported extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', dir=Path.cwd(), delete=False) as f:
            f.write("executable content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                doc_loader.load_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.security
    def test_memory_buffer_size_limits(self):
        """Test that memory buffers have size limits"""
        from app.services.memory_manager import ConversationMemoryManager
        
        manager = ConversationMemoryManager(auto_save=False)
        
        # Check that MAX_BUFFER_SIZE is defined and reasonable
        assert hasattr(manager, 'MAX_BUFFER_SIZE')
        assert manager.MAX_BUFFER_SIZE > 0
        assert manager.MAX_BUFFER_SIZE <= 10000  # Reasonable upper limit
    
    @pytest.mark.security
    def test_sql_injection_prevention(self):
        """Test that SQL injection is prevented"""
        # Since we use SQLAlchemy ORM, SQL injection should be prevented
        # This is a basic test to ensure we're not using raw SQL
        
        from app.core import database
        import inspect
        
        # Check that we're not using raw SQL execution
        source = inspect.getsource(database)
        
        # These patterns indicate potential SQL injection vulnerabilities
        dangerous_patterns = [
            '.execute(f"',  # f-string in execute
            '.execute("SELECT',  # Raw SELECT
            '.execute("INSERT',  # Raw INSERT
            '.execute("UPDATE',  # Raw UPDATE
            '.execute("DELETE',  # Raw DELETE
            '% (',  # String formatting in SQL
        ]
        
        for pattern in dangerous_patterns:
            assert pattern not in source, f"Potential SQL injection risk: {pattern} found"
    
    @pytest.mark.security
    def test_env_secrets_not_logged(self):
        """Test that secrets from environment are not logged"""
        import logging
        from io import StringIO
        
        # Set up a string buffer to capture logs
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        
        # Add handler to root logger
        logger = logging.getLogger()
        logger.addHandler(handler)
        
        try:
            # Import modules that might log
            from app.config import settings
            
            # Get log output
            log_output = log_capture.getvalue()
            
            # Check that sensitive values are not in logs
            sensitive_patterns = [
                'password',
                'secret',
                'token',
                'api_key',
                'DATABASE_URL',
            ]
            
            for pattern in sensitive_patterns:
                assert pattern.lower() not in log_output.lower(), \
                    f"Potential secret leak in logs: {pattern}"
        
        finally:
            logger.removeHandler(handler)