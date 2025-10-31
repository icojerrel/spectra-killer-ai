"""
Secure Credential Management System
Handles encrypted storage and retrieval of sensitive credentials
"""

import os
import json
import base64
from typing import Optional, Dict, Any
from pathlib import Path
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)


class CredentialManager:
    """
    Secure credential manager with encryption support
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize credential manager
        
        Args:
            master_key: Master encryption key (defaults to environment variable)
        """
        self.master_key = master_key or os.getenv('SPECTRA_MASTER_KEY')
        if not self.master_key:
            self.master_key = self._generate_or_load_master_key()
        
        self.cipher = self._create_cipher()
        self.credential_store = self._get_credential_store_path()
        
    def _generate_or_load_master_key(self) -> str:
        """Generate or load master encryption key"""
        key_path = Path.home() / '.spectra_killer' / 'master.key'
        key_path.parent.mkdir(exist_ok=True)
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                return f.read().decode()
        else:
            # Generate new key
            key = Fernet.generate_key().decode()
            with open(key_path, 'w') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(key_path, 0o600)
            logger.info("Generated new master encryption key")
            return key
    
    def _create_cipher(self) -> Fernet:
        """Create Fernet cipher from master key"""
        try:
            return Fernet(self.master_key.encode())
        except Exception as e:
            logger.error(f"Failed to create cipher: {e}")
            raise ValueError("Invalid master key format")
    
    def _get_credential_store_path(self) -> Path:
        """Get credential store file path"""
        store_dir = Path.home() / '.spectra_killer' / 'credentials'
        store_dir.mkdir(exist_ok=True)
        return store_dir / 'encrypted_store.json'
    
    def store_credential(self, service: str, credential_data: Dict[str, Any]) -> bool:
        """
        Encrypt and store credential data
        
        Args:
            service: Service name (e.g., 'openrouter', 'mt5')
            credential_data: Dictionary containing credential information
            
        Returns:
            True if successful
        """
        try:
            # Load existing store
            store = self._load_store()
            
            # Encrypt credential data
            json_data = json.dumps(credential_data)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            
            # Store encrypted credential
            store[service] = base64.b64encode(encrypted_data).decode()
            
            # Save encrypted store
            self._save_store(store)
            
            logger.info(f"Credential stored for service: {service}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store credential for {service}: {e}")
            return False
    
    def get_credential(self, service: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt credential data
        
        Args:
            service: Service name
            
        Returns:
            Credential data dictionary or None if not found
        """
        try:
            store = self._load_store()
            
            if service not in store:
                logger.warning(f"No credential found for service: {service}")
                return None
            
            # Decrypt credential data
            encrypted_data = base64.b64decode(store[service])
            decrypted_data = self.cipher.decrypt(encrypted_data).decode()
            credential_data = json.loads(decrypted_data)
            
            logger.debug(f"Credential retrieved for service: {service}")
            return credential_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve credential for {service}: {e}")
            return None
    
    def delete_credential(self, service: str) -> bool:
        """
        Delete credential data
        
        Args:
            service: Service name
            
        Returns:
            True if successful
        """
        try:
            store = self._load_store()
            
            if service in store:
                del store[service]
                self._save_store(store)
                logger.info(f"Credential deleted for service: {service}")
                return True
            else:
                logger.warning(f"No credential found to delete for service: {service}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete credential for {service}: {e}")
            return False
    
    def list_services(self) -> list:
        """
        List all stored service names
        
        Returns:
            List of service names
        """
        try:
            store = self._load_store()
            return list(store.keys())
        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            return []
    
    def _load_store(self) -> Dict[str, str]:
        """Load encrypted credential store"""
        if not self.credential_store.exists():
            return {}
        
        try:
            with open(self.credential_store, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Credential store corrupted, creating new store")
            return {}
    
    def _save_store(self, store: Dict[str, str]) -> None:
        """Save encrypted credential store"""
        try:
            with open(self.credential_store, 'w') as f:
                json.dump(store, f, indent=2)
            # Set restrictive permissions
            os.chmod(self.credential_store, 0o600)
        except IOError as e:
            logger.error(f"Failed to save credential store: {e}")
            raise
    
    def rotate_master_key(self, new_master_key: str) -> bool:
        """
        Rotate master encryption key and re-encrypt all credentials
        
        Args:
            new_master_key: New master key
            
        Returns:
            True if successful
        """
        try:
            # Load all credentials with old key
            old_cipher = self.cipher
            credentials = {}
            
            for service in self.list_services():
                credential_data = self.get_credential(service)
                if credential_data:
                    credentials[service] = credential_data
            
            # Update to new key
            self.master_key = new_master_key
            self.cipher = self._create_cipher()
            
            # Re-encrypt and store all credentials
            for service, credential_data in credentials.items():
                self.store_credential(service, credential_data)
            
            logger.info("Master key rotation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate master key: {e}")
            # Restore old key on failure
            self.master_key = os.getenv('SPECTRA_MASTER_KEY', self.master_key)
            self.cipher = old_cipher
            return False
    
    @staticmethod
    def generate_secure_password(length: int = 32) -> str:
        """
        Generate a secure random password
        
        Args:
            length: Password length
            
        Returns:
            Secure password string
        """
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def hash_api_key(api_key: str, salt: Optional[str] = None) -> str:
        """
        Create hash of API key for verification
        
        Args:
            api_key: API key to hash
            salt: Optional salt value
            
        Returns:
            Hashed API key
        """
        if salt is None:
            salt = os.getenv('API_KEY_SALT', 'spectra_default_salt')
        
        return hashlib.pbkdf2_hmac('sha256', 
                                  api_key.encode('utf-8'), 
                                  salt.encode('utf-8'), 
                                  100000).hex()


# Global credential manager instance
_credential_manager: Optional[CredentialManager] = None


def get_credential_manager() -> CredentialManager:
    """Get global credential manager instance"""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager


def secure_get_env(key: str, service: Optional[str] = None) -> Optional[str]:
    """
    Securely get environment variable or credential
    
    Args:
        key: Environment variable key or credential field
        service: Service name for credential lookup
        
    Returns:
        Credential value or None
    """
    # First try environment variable
    value = os.getenv(key)
    if value:
        return value
    
    # Then try credential manager
    if service:
        manager = get_credential_manager()
        cred_data = manager.get_credential(service)
        if cred_data and key in cred_data:
            return cred_data[key]
    
    return None
