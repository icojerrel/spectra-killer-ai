"""
Authentication and Authorization System
JWT-based authentication with role-based access control
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import hashlib
import secrets
from functools import wraps
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)


class Role:
    """User roles"""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


class Permission:
    """Permissions"""
    READ_MARKET_DATA = "read_market_data"
    PLACE_TRADES = "place_trades"
    MODIFY_STRATEGIES = "modify_strategies"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"
    VIEW_LOGS = "view_logs"
    TRADING_HISTORY = "trading_history"
    RISK_MANAGEMENT = "risk_management"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.READ_MARKET_DATA,
        Permission.PLACE_TRADES,
        Permission.MODIFY_STRATEGIES,
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_USERS,
        Permission.VIEW_LOGS,
        Permission.TRADING_HISTORY,
        Permission.RISK_MANAGEMENT,
    ],
    Role.TRADER: [
        Permission.READ_MARKET_DATA,
        Permission.PLACE_TRDES,
        Permission.VIEW_ANALYTICS,
        Permission.TRADING_HISTORY,
        Permission.RISK_MANAGEMENT,
    ],
    Role.VIEWER: [
        Permission.READ_MARKET_DATA,
        Permission.VIEW_ANALYTICS,
        Permission.TRADING_HISTORY,
    ],
    Role.API_CLIENT: [
        Permission.READ_MARKET_DATA,
        Permission.PLACE_TRADES,
    ],
}


class User:
    """User model"""
    
    def __init__(self, user_id: str, username: str, role: str, 
                 permissions: List[str], enabled: bool = True):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.permissions = permissions
        self.enabled = enabled
        self.last_login = None
        self.api_keys = []
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role"""
        return self.role == role
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role,
            'permissions': self.permissions,
            'enabled': self.enabled,
            'last_login': self.last_login,
        }


class UserManager:
    """User management system"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default system users"""
        # Create admin user
        admin_user = User(
            user_id="admin_001",
            username="admin",
            role=Role.ADMIN,
            permissions=ROLE_PERMISSIONS[Role.ADMIN],
        )
        self.users[admin_user.user_id] = admin_user
        
        # Create demo trader
        trader_user = User(
            user_id="trader_001", 
            username="demo_trader",
            role=Role.TRADER,
            permissions=ROLE_PERMISSIONS[Role.TRADER],
        )
        self.users[trader_user.user_id] = trader_user
    
    def create_user(self, username: str, role: str, enabled: bool = True) -> User:
        """Create new user"""
        user_id = f"{role}_{secrets.token_hex(4)}"
        permissions = ROLE_PERMISSIONS.get(role, [])
        
        user = User(
            user_id=user_id,
            username=username,
            role=role,
            permissions=permissions,
            enabled=enabled,
        )
        
        self.users[user_id] = user
        logger.info(f"Created user: {username} with role: {role}")
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate API key for user"""
        if user_id not in self.users:
            raise ValueError(f"User not found: {user_id}")
        
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        self.api_keys[api_key] = user_id
        self.users[user_id].api_keys.append(api_key)
        
        logger.info(f"Generated API key for user: {user_id}")
        return api_key
    
    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        user_id = self.api_keys.get(api_key)
        if user_id:
            return self.users.get(user_id)
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if api_key in self.api_keys:
            user_id = self.api_keys[api_key]
            del self.api_keys[api_key]
            
            # Remove from user's API keys
            if user_id in self.users:
                self.users[user_id].api_keys = [
                    key for key in self.users[user_id].api_keys if key != api_key
                ]
            
            logger.info(f"Revoked API key for user: {user_id}")
            return True
        return False


class TokenManager:
    """JWT token management"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.getenv('JWT_SECRET_KEY', 
                                                 secrets.token_urlsafe(32))
        self.algorithm = 'HS256'
        self.token_expiry = timedelta(hours=24)
        self.refresh_expiry = timedelta(days=30)
    
    def generate_access_token(self, user: User) -> str:
        """Generate JWT access token"""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role,
            'permissions': user.permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + self.token_expiry,
            'type': 'access'
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def generate_refresh_token(self, user: User) -> str:
        """Generate JWT refresh token"""
        payload = {
            'user_id': user.user_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + self.refresh_expiry,
            'type': 'refresh'
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token"""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get('type') != 'refresh':
            return None
        
        user_id = payload.get('user_id')
        # In a real implementation, you'd retrieve user from database
        # For now, we'll create a minimal user object
        user = User(
            user_id=user_id,
            username="unknown",
            role="viewer",
            permissions=[]
        )
        
        return self.generate_access_token(user)


class AuthenticationManager:
    """Main authentication manager"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.user_manager = UserManager()
        self.token_manager = TokenManager(secret_key)
        
        # FastAPI security scheme
        self.security = HTTPBearer()
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        user = self.user_manager.get_user_by_api_key(api_key)
        if user and user.enabled:
            user.last_login = datetime.utcnow()
            return user
        return None
    
    def authenticate_token(self, token: str) -> Optional[User]:
        """Authenticate using JWT token"""
        payload = self.token_manager.verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get('user_id')
        if payload.get('type') != 'access':
            return None
        
        user = self.user_manager.get_user(user_id)
        if user and user.enabled:
            user.last_login = datetime.utcnow()
            return user
        return None
    
    def authenticate_credentials(self, username: str, password: str) -> Optional[str]:
        """Authenticate using username/password and return access token"""
        user = self.user_manager.get_user_by_username(username)
        if not user or not user.enabled:
            return None
        
        # In a real implementation, you'd check against stored password hash
        # For now, we'll use a simple demo password
        if self._verify_password(password, username):
            return self.token_manager.generate_access_token(user)
        return None
    
    def _verify_password(self, password: str, username: str) -> bool:
        """Simple password verification (demo only)"""
        # In production, use proper password hashing
        demo_passwords = {
            'admin': 'admin123',
            'demo_trader': 'demo123'
        }
        return demo_passwords.get(username) == password
    
    def login_user(self, username: str, password: str) -> Optional[Dict]:
        """Login user and return tokens"""
        access_token = self.authenticate_credentials(username, password)
        if not access_token:
            return None
        
        user = self.user_manager.get_user_by_username(username)
        refresh_token = self.token_manager.generate_refresh_token(user)
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'user': user.to_dict()
        }
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> User:
        """FastAPI dependency to get current authenticated user"""
        token = credentials.credentials
        user = self.authenticate_token(token)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user


# Global authentication manager instance
_auth_manager: Optional[AuthenticationManager] = None


def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from FastAPI dependency injection
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            if not current_user.has_permission(permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission required: {permission}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(role: str):
    """Decorator to require specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )
            
            if not current_user.has_role(role):
                raise HTTPException(
                    status_code=403,
                    detail=f"Role required: {role}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# FastAPI dependency helpers
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> User:
    """Get current authenticated user"""
    auth_manager = get_auth_manager()
    return auth_manager.get_current_user(credentials)


async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current user with admin role verification"""
    if not current_user.has_role(Role.ADMIN):
        raise HTTPException(
            status_code=403,
            detail="Admin role required"
        )
    return current_user


async def get_trader_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current user with trader role verification"""
    if not current_user.has_role(Role.TRADER) and not current_user.has_role(Role.ADMIN):
        raise HTTPException(
            status_code=403,
            detail="Trader role required"
        )
    return current_user
