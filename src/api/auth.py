"""
Auth utilities for PRISM API using Auth0 JWKS (RS256).
"""
import os
import time
import threading
from functools import wraps
from flask import request, jsonify, g
import requests
from jose import jwt

AUTH0_DOMAIN = os.environ.get('AUTH0_DOMAIN')
AUTH0_AUDIENCE = os.environ.get('AUTH0_AUDIENCE')
AUTH0_ALGORITHMS = [alg.strip() for alg in os.environ.get('AUTH0_ALGORITHMS', 'RS256').split(',') if alg.strip()]
AUTH0_ENABLED = os.environ.get('AUTH0_ENABLED', 'false').lower() == 'true'


class AuthError(Exception):
    def __init__(self, error, status_code):
        self.error = error
        self.status_code = status_code


_jwks_cache = None
_jwks_cache_expiry = 0
_jwks_lock = threading.Lock()
_JWKS_CACHE_TTL = 900


def get_jwks():
    global _jwks_cache, _jwks_cache_expiry
    now = time.time()
    if _jwks_cache is not None and now < _jwks_cache_expiry:
        return _jwks_cache
    with _jwks_lock:
        if _jwks_cache is not None and now < _jwks_cache_expiry:
            return _jwks_cache
        jwks_url = f'https://{AUTH0_DOMAIN}/.well-known/jwks.json'
        r = requests.get(jwks_url, timeout=5)
        r.raise_for_status()
        _jwks_cache = r.json()
        _jwks_cache_expiry = now + _JWKS_CACHE_TTL
        return _jwks_cache


def verify_jwt(token):
    jwks = get_jwks()
    unverified_header = jwt.get_unverified_header(token)
    rsa_key = None
    for key in jwks.get('keys', []):
        if key.get('kid') == unverified_header.get('kid'):
            rsa_key = {'kty': key['kty'], 'kid': key['kid'], 'use': key['use'], 'n': key['n'], 'e': key['e']}
            break
    if not rsa_key:
        raise AuthError({'code': 'invalid_header', 'description': 'Unable to find appropriate key'}, 401)
    try:
        return jwt.decode(token, rsa_key, algorithms=AUTH0_ALGORITHMS,
                          audience=AUTH0_AUDIENCE, issuer=f'https://{AUTH0_DOMAIN}/')
    except jwt.ExpiredSignatureError:
        raise AuthError({'code': 'token_expired', 'description': 'token is expired'}, 401)
    except jwt.JWTClaimsError as e:
        raise AuthError({'code': 'invalid_claims', 'description': str(e)}, 401)
    except Exception as e:
        raise AuthError({'code': 'invalid_token', 'description': str(e)}, 401)


def get_token_auth_header():
    auth = request.headers.get('Authorization', None)
    if not auth:
        raise AuthError({'code': 'authorization_header_missing', 'description': 'Authorization header missing'}, 401)
    parts = auth.split()
    if parts[0].lower() != 'bearer' or len(parts) != 2:
        raise AuthError({'code': 'invalid_header', 'description': 'Authorization header must be Bearer token'}, 401)
    return parts[1]


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_enabled = os.environ.get('AUTH0_ENABLED', 'false').lower() == 'true'
        if not auth_enabled:
            return f(*args, **kwargs)
        if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
            raise AuthError({'code': 'config_error', 'description': 'AUTH0 not configured'}, 500)
        try:
            token = get_token_auth_header()
            payload = verify_jwt(token)
            g.current_user = payload
        except AuthError as err:
            return jsonify({'status': 'error', 'message': err.error.get('description', 'Unauthorized')}), err.status_code
        return f(*args, **kwargs)
    return decorated