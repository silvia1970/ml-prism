"""
Auth utilities for PRISM API using Auth0 JWKS (RS256)

This module provides a `requires_auth` decorator that validates
an incoming `Authorization: Bearer <token>` header against the
Auth0 JWKS endpoint. Verification is optional and controlled by
the `AUTH0_ENABLED` environment variable (default: false for dev).

Dependencies (install in your API virtualenv):
  pip install python-jose[cryptography] requests

Configuration (env vars):
  AUTH0_DOMAIN - e.g. prism.eu.auth0.com
  AUTH0_AUDIENCE - e.g. https://api.prism.local
  AUTH0_ALGORITHMS - comma separated, default RS256
  AUTH0_ENABLED - 'true' to enable enforcement
"""
import os
import json
import time
import threading
from functools import wraps
from flask import request, jsonify, g
import requests
from jose import jwt
from jose.utils import base64url_decode

AUTH0_DOMAIN = os.environ.get('AUTH0_DOMAIN')
AUTH0_AUDIENCE = os.environ.get('AUTH0_AUDIENCE')
AUTH0_ALGORITHMS = [alg.strip() for alg in os.environ.get('AUTH0_ALGORITHMS', 'RS256').split(',') if alg.strip()]
AUTH0_ENABLED = os.environ.get('AUTH0_ENABLED', 'false').lower() == 'true'

# Access token lifetime in hours (for development: long-lived tokens).
# Set ACCESS_TOKEN_EXPIRY_HOURS in your .env to override.
# Note: this value is only used when generating tokens via a custom
# token endpoint or Auth0 Management API; for Auth0-issued tokens the
# actual expiry is controlled in the Auth0 dashboard
# (Applications → <app> → Settings → Token Expiration).
# For development convenience the default here is 8 hours.
ACCESS_TOKEN_EXPIRY_HOURS = int(os.environ.get('ACCESS_TOKEN_EXPIRY_HOURS', 8))


class AuthError(Exception):
    def __init__(self, error, status_code):
        self.error = error
        self.status_code = status_code


def _validate_auth0_config() -> None:
    """Fail fast when Auth0 enforcement is enabled but configuration is incomplete."""
    if not AUTH0_DOMAIN or not AUTH0_AUDIENCE:
        raise AuthError(
            {'code': 'config_error', 'description': 'AUTH0_DOMAIN and AUTH0_AUDIENCE must be configured when AUTH0_ENABLED=true'},
            500,
        )
    if AUTH0_ALGORITHMS != ['RS256']:
        raise AuthError(
            {'code': 'config_error', 'description': 'Only RS256 is supported for Auth0 JWT validation'},
            500,
        )


def get_token_auth_header():
    auth = request.headers.get('Authorization', None)
    if not auth:
        raise AuthError({'code': 'authorization_header_missing', 'description': 'Authorization header is expected'}, 401)

    parts = auth.split()

    if parts[0].lower() != 'bearer':
        raise AuthError({'code': 'invalid_header', 'description': 'Authorization header must start with Bearer'}, 401)
    elif len(parts) == 1:
        raise AuthError({'code': 'invalid_header', 'description': 'Token not found'}, 401)
    elif len(parts) > 2:
        raise AuthError({'code': 'invalid_header', 'description': 'Authorization header must be Bearer token'}, 401)

    token = parts[1]
    return token


# Cached JWKS data and expiry timestamp
_jwks_cache = None
_jwks_cache_expiry = 0
_jwks_lock = threading.Lock()
# Cache TTL: 15 minutes (JWKS keys rotate rarely)
_JWKS_CACHE_TTL = 900


def get_jwks():
    global _jwks_cache, _jwks_cache_expiry
    now = time.time()

    # Fast path: return cached JWKS if still valid (no lock needed for reads)
    if _jwks_cache is not None and now < _jwks_cache_expiry:
        return _jwks_cache

    with _jwks_lock:
        # Re-check after acquiring lock (double-checked locking)
        if _jwks_cache is not None and now < _jwks_cache_expiry:
            return _jwks_cache

        if not AUTH0_DOMAIN:
            raise AuthError({'code': 'config_error', 'description': 'AUTH0_DOMAIN not configured'}, 500)
        jwks_url = f'https://{AUTH0_DOMAIN}/.well-known/jwks.json'
        try:
            r = requests.get(jwks_url, timeout=5)
            r.raise_for_status()
            _jwks_cache = r.json()
            _jwks_cache_expiry = now + _JWKS_CACHE_TTL
            return _jwks_cache
        except Exception as e:
            raise AuthError({'code': 'jwks_fetch_failed', 'description': f'Could not fetch JWKS: {e}'}, 503)


def verify_jwt(token):
    jwks = get_jwks()
    unverified_header = jwt.get_unverified_header(token)
    rsa_key = {}

    for key in jwks.get('keys', []):
        if key.get('kid') == unverified_header.get('kid'):
            rsa_key = {
                'kty': key.get('kty'),
                'kid': key.get('kid'),
                'use': key.get('use'),
                'n': key.get('n'),
                'e': key.get('e')
            }
            break

    if not rsa_key:
        raise AuthError({'code': 'invalid_header', 'description': 'Unable to find appropriate key'}, 401)

    try:
        # Build a PEM public key from the JWKS 'n' and 'e' values using jose
        # python-jose will accept the jwk dict directly as the key parameter
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=AUTH0_ALGORITHMS,
            audience=AUTH0_AUDIENCE,
            issuer=f'https://{AUTH0_DOMAIN}/'
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError({'code': 'token_expired', 'description': 'token is expired'}, 401)
    except jwt.JWTClaimsError as e:
        raise AuthError({'code': 'invalid_claims', 'description': str(e)}, 401)
    except Exception as e:
        raise AuthError({'code': 'invalid_header', 'description': str(e)}, 401)


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Read AUTH0_ENABLED dynamically to allow runtime overrides (e.g. tests)
        auth_enabled = os.environ.get('AUTH0_ENABLED', 'false').lower() == 'true'
        if not auth_enabled:
            return f(*args, **kwargs)

        _validate_auth0_config()

        try:
            token = get_token_auth_header()
            payload = verify_jwt(token)
            # attach payload to flask.g for downstream use
            g.current_user = payload
        except AuthError as err:
            return jsonify({'status': 'error', 'message': err.error.get('description', 'Unauthorized')}), err.status_code

        return f(*args, **kwargs)

    return decorated
