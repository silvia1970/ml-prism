# Auth0 Setup and Local Integration Guide

Questa guida spiega i passaggi per creare un tenant Auth0, registrare un'API e una Single Page Application, e configurare l'app PRISM per l'autenticazione locale.

1) Creare Tenant Auth0
  - Vai su https://auth0.com e crea un account / tenant
  - Scegli il nome (es. prism-dev)

2) Creare una API (Identifier)
  - Dashboard → APIs → Create API
  - Name: PRISM API
  - Identifier: `https://api.prism.local` (questo sarà l'AUDIENCE)
  - Signing Algorithm: RS256

3) Creare una Application (Single Page App)
  - Dashboard → Applications → Create Application
  - Type: Single Page Application
  - Settings:
    - Allowed Callback URLs: `http://localhost:5173`
    - Allowed Logout URLs: `http://localhost:5173`
    - Allowed Web Origins: `http://localhost:5173`

4) Recupera i valori necessari
  - Domain (es. `prism.eu.auth0.com`)
  - Client ID (della SPA)
  - Audience (l'Identifier dell'API creato, es. `https://api.prism.local`)

5) Configura le variabili d'ambiente locali
  - Nel frontend, crea file `.env` nella cartella `frontend/` con:
    VITE_AUTH0_DOMAIN=your-domain.auth0.com
    VITE_AUTH0_CLIENT_ID=your-client-id
    VITE_AUTH0_AUDIENCE=https://api.prism.local
    VITE_AUTH0_REDIRECT_URI=http://localhost:5173
    VITE_API_BASE_URL=http://localhost:5000

  - Nel backend, esporta le variabili (o aggiungile a `.env` nella root):
    AUTH0_DOMAIN=your-domain.auth0.com
    AUTH0_AUDIENCE=https://api.prism.local
    AUTH0_ENABLED=true

6) Installare dipendenze
  - Backend (virtualenv attivo):
    ```bash
    pip install python-jose[cryptography] requests
    ```

  - Frontend:
    ```bash
    cd frontend
    npm install
    ```

7) Avviare e testare
  - Avvia backend:
    ```bash
    ./start_backend.sh
    ```
  - Avvia frontend:
    ```bash
    cd frontend
    npm run dev
    ```
  - Apri `http://localhost:5173`, clicca "Accedi con Auth0" → effettua il login su Auth0 → al ritorno l'app dovrebbe recuperare il token e aggiungerlo alle chiamate API.

8) Debug
  - Se il backend restituisce 401, verifica che `AUTH0_AUDIENCE` corrisponda all'audience nel token (campo `aud`) e che `AUTH0_DOMAIN` sia corretto.
  - Verifica che `AUTH0_ENABLED=true` sia impostato solo quando vuoi enforceare l'autenticazione.

Se vuoi, procedo io con la creazione dei file e con una verifica passo‑passo (tu mi fornisci i valori `Domain`, `Client ID` e `Audience`), oppure ti guido mentre li inserisci nella dashboard Auth0.
