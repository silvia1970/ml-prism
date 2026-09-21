# Changelog

Tutte le modifiche notevoli al progetto PRISM ML saranno documentate in questo file.

Il formato si basa su [Keep a Changelog](https://keepachangelog.com/it/1.0.0/),
e questo progetto aderisce a [Semantic Versioning](https://semver.org/lang/it/).

## [3.5.0] — 2026-07-02

### Rimosso

- **Frontend React** rimosso completamente (`frontend/`, `docker/frontend/`, `frontend_specifics/`). Il progetto è ora backend-only.
- **Legacy script**: `run_Prism.py` (API standalone deprecata), `utils.py` root-level (analisi dati non usata dall'API).
- **Docker frontend**: `docker/frontend/Dockerfile`, `docker/frontend/nginx.conf`. Il `docker-compose.yml` gestisce solo backend + MinIO.
- **Test obsoleti**: `tests/backend/test_api_simple.py`, `test_v3_compliance.py`, `test_csv_template.py`. Shell test script rimossi.
- **Documentazione obsoleta**: 20 file docs/ consolidati in README.md, API_DOCUMENTATION.md, CHANGELOG.md, AUTH0_SETUP.md.

### Modificato

- **Circular import fix**: `process_records_batch` spostato in `api/blueprints/shared.py`. Import lazy in `api/blueprints/__init__.py`.
- **Blueprint `data.py`**: accetta sia `db` che `db_name` nel payload per backward compatibility.
- **Test suite**: `test_all_apis.py` aggiornato al formato risposta API corrente (53 test, 52 passanti).
- **docker-compose.yml**: rimosso servizio frontend. CORS default senza `localhost:3003`.
- **.gitignore**: rimosse entry Node.js/frontend non piu necessarie.
- **Documentazione**: README.md, ENDPOINTS.md, api/README.md, .env.example completamente riscritti.

## [3.4.1] - 2026-05-27

### Fixed – Hardening sicurezza e robustezza

- **SEC-8** `api/app.py` ora applica headers di sicurezza e cache-control su tutte le risposte:
  `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`,
  `Cache-Control`, `Pragma`, `Expires`. In ambiente prod, se la connessione è HTTPS,
  viene aggiunto anche `Strict-Transport-Security`.
- **SEC-9** `api/app.py` valida la configurazione di runtime all'avvio: se `AUTH0_ENABLED=true`
  ma mancano `AUTH0_DOMAIN` o `AUTH0_AUDIENCE`, l'app fallisce subito invece di restare in uno
  stato parzialmente configurato.
- **SEC-10** `api/auth.py` ora normalizza e limita gli algoritmi JWT supportati a `RS256`, con
  validazione esplicita della configurazione Auth0 quando l'enforcement è attivo.
- **SEC-11** `api/app.py` segnala esplicitamente l'uso del placeholder `SECRET_KEY` di default e
  `api/minio_storage.py` avvisa quando MinIO è avviato con credenziali deboli (`minioadmin`).


## [3.4.0] - 2026-05-26

### Fixed – Bug critici

- **BUG-DB-1** `database._init_single_db`: bug di indentazione che lasciava tutte le istruzioni
  `CREATE TABLE`, `CREATE INDEX`, `ALTER TABLE` e `conn.commit()` **fuori** dal blocco
  `with sqlite3.connect(db_path) as conn:`. Le DDL venivano eseguite sulla connessione ancora aperta
  ma fuori dal contesto della transazione esplicita; ora sono correttamente dentro il `with`.
- **BUG-PAGE-1** Mismatch chiavi di paginazione nei blueprint (`stats.py`, `patients.py`,
  `submissions.py`): il metodo `result.get('total', 0)` restituiva sempre `0` perché il layer
  database usa chiavi diverse (`total_records`, `total_patients`, `total_submissions`). Stesso
  problema per le pagine totali (`pages` vs `total_pages` / `page_total`). Corretti tutti e tre i
  blueprint in modo da leggere le chiavi effettive restituite dal DAL.
- **BUG-PAGE-2** `PatientHistory.jsx` leggeva `data.total_pages` e `data.total_records`
  direttamente sulla risposta API, ma la risposta li wrappa dentro `data.pagination.pages` e
  `data.pagination.total`. Corretti tutti i punti (`loadPatients`, `loadAllRecords`).
- **BUG-AUTH-1** `App.jsx`: l'espressione `!user?.authProvider === 'auth0'` applica `!` solo a
  `user?.authProvider` per precedenza operatori, rendendo la condizione sempre `false`. Il refresh
  token automatico via `setInterval` veniva attivato per **tutti** gli utenti autenticati invece
  che solo per quelli Auth0. Corretto in `user?.authProvider !== 'auth0'`.
- **BUG-URL-1** `CSVUpload.jsx`: il download del template CSV usava
  `` `${import.meta.env.VITE_API_BASE_URL}/api/v1/...` `` che produce la stringa letterale
  `"undefined/api/v1/..."` quando la variabile d'ambiente non è definita (ambienti Docker con
  reverse-proxy same-origin). Corretto usando `VITE_API_BASE_URL ?? 'http://localhost:5000'`
  in allineamento con la logica di `axiosInstance`.

### Improved – Qualità del codice

- **ARCH-DB-1** `database.store_record`: connessione SQLite convertita a context manager
  (`with sqlite3.connect(...) as conn:`) in luogo del pattern manuale `conn.commit()` /
  `conn.close()`. Garantisce rollback automatico in caso di eccezione.
- **ARCH-DB-2** Stesso fix applicato a `database.update_record_score`,
  `database.update_record_score_by_sample`, `database.store_submission` — tutte le operazioni di
  scrittura usano ora context manager SQLite.
- **ARCH-FE-1** `apiClient.js`: le mappe di conversione dei nomi dei campi (`SEPSIEXP_FIELD_MAPPING`,
  `MIMIC_FIELD_MAPPING`) erano dichiarate come variabili locali dentro `submitScore()`,
  ricostruite ad ogni chiamata. Spostate a costanti di modulo per evitare allocazioni inutili.

## [3.3.0] - 2026-05-25

### Fixed – Bug critici (runtime crash)

- **BUG-4** `database.find_submission` ora restituisce un `dict` uniforme invece di una tupla
  `(row_tuple, submission_id)`. Tutti i blueprint che chiamavano `.get('db_name')` sulla risposta
  non lanciano più `AttributeError`.
- **BUG-8** `ChartGenerator.generate_risk_distribution_chart / generate_score_distribution_chart /
  generate_combined_chart` accettano ora entrambe le calling convention:
  - legacy: `(results, db_type, submission_id)`
  - blueprint: `(records=…, db_name=…, title=…)` — signature attesa da `charts.py`
- **BUG-14** Il blueprint `submissions.py` accedeva direttamente a SQLite (`sqlite3.connect(db._get_db_path(...))`
  bypassando il DAL. Aggiunto `Database.delete_submission()` che centralizza la logica di cancellazione
  (SQLite + file JSON + MinIO mirror). Il blueprint ora delega a questo metodo.
- **BUG-16** `run_Prism.py::normalize_single_patient_sepsisexp`: variabile `pat` indefinita
  sostituita con il parametro corretto `single_patient_df`.
- **BUG-1** Sostituito il no-op `.replace('+00:00', '+00:00')` (15+ occorrenze) con
  `.replace('+00:00', 'Z')` in `database.py`, `csv_handler.py`, `app.py`, `blueprints/stats.py`.
- **BUG-2** `app.py __main__`: `debug=True` hardcoded sostituito con auto-detect da `FLASK_ENV`.
- **BUG-12** `torch_models.py::prepare_input_sequence`: `print()` sostituito con `logger.debug()`.

### Fixed – Sicurezza

- **SEC-1** `SECRET_KEY`: rimosso il fallback silenzioso. Se `SECRET_KEY` non è nel `.env`,
  viene emesso un `warnings.warn` esplicito. Non crasha l'avvio ma segnala chiaramente il problema.
- **SEC-2** `app.py` Swagger config: URL Auth0 hardcoded sostituita con `os.getenv('AUTH0_DOMAIN')`.
- **SEC-4** `database._save_record_to_file`: `sample_id` sanitizzato prima dell'uso come nome file
  (solo alfanumerici, `-`, `_`) per prevenire path traversal.
- **SEC-6** `minio_storage._get_client`: rimosso il default `minioadmin` per access/secret key.
  Se le variabili non sono impostate, MinIO viene disabilitato con un log `WARNING`.
- **SEC-7** `blueprints/charts.py`: path traversal mitigato con `os.path.basename()` invece di
  check manuale su `..`/`/`/`\\` (che non copriva URL encoding).

### Fixed – Architettura e qualità

- **ARCH-1** Rimossa la doppia istanziazione di `ModelLoader` in `app.py`. `MLScorer` crea già
  internamente un `ModelLoader`; `app.py` ora riusa `ml_scorer.torch_loader` invece di crearne uno
  separato. Riduzione da 3 → 1 deserializzazione dei `.pth` allo startup.
- **ARCH-13** `torch_models.py`: due `import json` locali dentro `__init__` rimossi; `json` è ora
  importato a livello di modulo come da best practice.
- **ARCH-14** `blueprints/csv.py`: tutti i `from datetime import …`, `from api.field_mappings import …`
  e `import pandas as pd` spostati a livello di modulo (erano eseguiti ad ogni request HTTP).
- **DUPL-6** Loop store→predict→update duplicato in `data.py` e `csv.py` estratto nella funzione
  `process_records_batch()` in `api/blueprints/__init__.py`. Entrambi i blueprint la importano.
- **SMELL-1** `auth.py::get_jwks`: `__import__('time')` sostituito con import di modulo `time`.
- **THREAD-1** `auth.py::get_jwks`: JWKS cache protetta da `threading.Lock` con double-checked
  locking per sicurezza in ambienti multi-worker.
- **THREAD-1** `minio_storage._get_client`: singleton protetto da `threading.Lock` con
  double-checked locking. Rimosso il re-read del env var ad ogni chiamata (`_MINIO_ENABLED`
  ora è costante di modulo).

## [3.0.0] - 2026-02-25

### ⚠ Breaking Changes

#### Score e Classi di Rischio
- **Score format**: ora `0.0–1.0` (probabilità sigmoid grezza) invece di `0–100`
- **Classi di rischio**: `low_risk` / `moderate_risk` / `high_risk` (sostituiscono `positive` / `negative`)
  - `low_risk`: score < 0.50
  - `moderate_risk`: 0.50 ≤ score < 0.75
  - `high_risk`: score ≥ 0.75

#### Database Separati
- **DB MIMIC**: `api_data/mimic.db` (file SQLite dedicato)
- **DB SepsisExp**: `api_data/sepsiexp.db` (file SQLite dedicato)
- Il vecchio `prism.db` è mantenuto come fallback legacy

#### Nomi Campi SepsisExp (DataFlow Schema)
- `heart_rate` → `heartrate`
- `mean_bp` → `meanbp`
- `heart_time_volume` → `hearttimevolume`
- `oxygen_saturation` → `oxygensaturation`
- `delta_temperature` → `deltatemp`
- `mixed_venous_oxygen_saturation` → `oxygenationsaturation`
- `pancreatic_lipase` → `pancreaticlipase`
- `blood_urea_nitrogen` → `bun`
- `procalcitonin` → `pct`
- `bun_creatinine_ratio` → `buncreatinineratio`
- `aspartate_transaminase` → `ast`
- `c_reactive_protein` → `crp`
- `respiratory_minute_volume` → `respiratoryminutevolume`
- `arterial_ph` → `arterialph`
- `partial_pressure_art_o2` → `pa_o2`
- I vecchi nomi snake_case rimangono come alias backward-compat

### Aggiunto

#### Infrastruttura
- **Database separati** per MIMIC e SepsisExp (`mimic.db`, `sepsiexp.db`)
- **Metodo `_get_db_path(db_name)`**: routing automatico al DB corretto
- **`_init_single_db()`**: inizializzazione schema per singolo file SQLite
- **`get_submissions()` cross-DB**: merge e deduplicazione da entrambi i DB

#### Scoring & Classificazione
- **`_classify_risk(prob)`** in `torch_models.py`: helper con soglie DataFlow Schema
- **`_classify_score(score)`** aggiornato in `ml_scorer.py`: stesse soglie

#### API
- **`trace_id`** (UUID v4) nelle risposte 500 per tracciabilità errori
- **404 con timestamp**: messaggio include `at time '...'` quando il parametro è fornito

### Modificato

#### `api/database.py`
- Tutti i metodi ora usano `self._get_db_path(db_name)` invece di `self.db_path`
- Query `get_patient_submissions()`: filtro `high_risk`/`moderate_risk` al posto di `positive`
- Query `get_patient_statistics()`: `class IN ('high_risk', 'moderate_risk')` al posto di vecchi label

#### `api/torch_models.py`
- `predict()`: restituisce `score: round(prob, 4)` (0.0–1.0) con `_classify_risk()`
- `predict_sequence()`: stesse modifiche nel loop windows e summary

#### `api/ml_scorer.py`
- `_classify_score()`: soglie 0.50/0.75, label `low_risk`/`moderate_risk`/`high_risk`
- `_mock_predict()`: stessi formati di output

#### `api/validators.py`
- `SCHEMAS['sepsiexp']`: tutti i nomi campi allineati a DataFlow Schema
- `alt`: reso opzionale (`required: False`)

#### `api/field_mappings.py`
- `SEPSIEXP_JSON_TO_PYTORCH`: chiavi primarie sono i nomi DataFlow
- Alias backward-compat per i vecchi nomi snake_case

#### `api/app.py`
- Swagger enum: `[low_risk, moderate_risk, high_risk]` (era `[positive, negative]`)
- Swagger score range: `0.0–1.0` (era `0–100`)
- Handler 500: include `trace_id: str(uuid.uuid4())`

#### `frontend_specifics/PRISM-DataFlow_Schema.json`
- `prediction_output.class.enum`: `["low_risk", "moderate_risk", "high_risk"]`
- `prediction_output.score`: `min: 0.0, max: 1.0`

## [2.2.0] - 2026-01-13

### Aggiunto

#### Autenticazione Frontend
- **LoginPage.jsx**: Nuova pagina di login con form username/password
- **Session persistence**: Salvataggio sessione in sessionStorage
- **Logout button**: Pulsante logout
- **Credenziali demo**: admin/admin, demo/demo per development
- **Predisposto per Auth0**: Struttura pronta per autenticazione reale

#### UI/UX Improvements
- **Palette colori neutri**: Sostituiti blu/viola con slate/stone professionali
- **Logo Youbiquo**: Aggiunto nell'header (destra) e footer
- **Favicon personalizzato**: /icon/favicon.ico
- **Gauge charts migliorati**: Zone colorate (rosso-verde-rosso) per statistiche
- **Sliding window compatto**: Visualizzazione semplificata delle finestre

#### Funzionalità Predizioni
- **Auto-prediction dopo upload**: Esecuzione automatica predizioni con delay 1500ms
- **Dropdown disabilitato durante auto-load**: UX migliorata durante caricamento
- **Ricerca pazienti autocomplete**: Dropdown con debounce per predizioni e statistiche
- **Parametri fissi**: target_len=24, stride=6 (solo visualizzazione)

### Modificato

#### Frontend
- **App.jsx**: Gestione autenticazione con isAuthenticated state
- **PatientHistory.jsx**: Fix aggiornamento record dopo delete submission
- **App.css**: Pulizia ~550 righe di CSS inutilizzato, nuovi stili login

#### Stili
- **Background neutro**: `--gray-50: #f5f5f4` (stone-100)
- **Gradienti rimossi**: Sostituiti #eff6ff, #dbeafe con toni neutri
- **Login page styles**: Card centrata con header colorato

### Fisso
- **Record count non aggiornato**: Ricarica lista pazienti dopo delete
- **Colori blu/viola residui**: Sostituiti con palette neutra
- **CSS duplicato**: Rimossi blocchi sliding-window-explainer obsoleti

---

## [2.1.0] - 2025-12-15

### Aggiunto

#### Frontend
- **Icone professionali** con libreria `lucide-react` sostituendo tutte le emoji
- **Componente SubmissionEditor**: Modal interattivo per modificare submissions
- **Bottone "Modifica e Ricalcola"** nella pagina risultati CSV
- **Animazioni fluide** per transizioni e feedback utente
- **Loading states migliorati** con spinner professionali
- **Card statistiche animate** nel Dashboard con icone colorate
- **Download template CSV** con gestione blob migliorata

#### Backend API
- **Endpoint PATCH migliorato** `/api/v1/data/<sample_id>` con ricalcolo automatico predizione
- **Endpoint GET** `/api/v1/submissions/<submission_id>` per dettagli submission
- **Endpoint POST** `/api/v1/submissions/<submission_id>/recalculate` per batch update e ricalcolo
- **Validazione robusta** per tutte le operazioni di update
- **Log migliorati** per troubleshooting

#### Utilità
- **Script cleanup_database.py** per pulizia completa database
- **Backup automatico** database prima della pulizia
- **Modalità interattiva e force** per lo script di cleanup

#### Documentazione
- **NEW_FEATURES_V2.1.md**: Documentazione completa nuove funzionalità
- **Esempi API** con curl per tutti i nuovi endpoint
- **Guida utilizzo** SubmissionEditor
- **Best practices** per modifiche batch

### Modificato

#### Frontend
- **Navigazione** con icone e layout migliorato
- **Dashboard** con card rinnovate e icone contestuali
- **CSVUpload** con icone e feedback visivo migliorato
- **Alert e messaggi** con icone appropriate (CheckCircle, AlertCircle, AlertTriangle)
- **Bottoni** con icone integrate per chiarezza

#### Backend API
- **Risposta PATCH** ora include predizione ricalcolata
- **Gestione errori** più granulare con dettagli specifici
- **Performance** ottimizzata per batch operations

#### Stili CSS
- **App.css**: Aggiunto gap per icone nei nav-button
- **Dashboard.css**: Nuovi stili per stat-icon e spinning animation
- **Transizioni smooth** per tutti gli elementi interattivi

### Fisso
- **Validazione campi** nelle operazioni di update
- **Gestione errori** per file non trovati
- **Pulizia risorse** dopo operazioni batch
- **Memory leaks** in componenti modal

## [2.0.0] - 2025-12-10

### Aggiunto
- Sistema completo con Flask REST API
- Frontend React con Vite
- Upload CSV con validazione
- Modelli LSTM PyTorch (MIMIC e SepsisExp)
- Database SQLite + JSON per persistenza
- Generazione grafici automatica
- Dashboard con statistiche
- Storico pazienti
- Form inserimento manuale
- 13 endpoint API RESTful
- Documentazione Swagger
- Test automatizzati

### Caratteristiche Principali
- **MIMIC**: 34 features, LSTM 4 layer
- **SepsisExp**: 29 features, LSTM 4 layer
- **CSV Templates** scaricabili
- **Real-time predictions** con score 0-1
- **Risk classification**: LOW/MODERATE/HIGH
- **CORS** configurabile
- **Environment variables** support

## [1.0.0] - 2025-11-01

### Aggiunto
- Versione iniziale con Tkinter GUI
- Supporto base per MIMIC dataset
- Predizioni ML semplici
- Export risultati CSV

---

## Tipi di modifiche

- **Aggiunto** per nuove funzionalità
- **Modificato** per cambiamenti in funzionalità esistenti
- **Deprecato** per funzionalità che saranno rimosse
- **Rimosso** per funzionalità rimosse
- **Fisso** per bug fix
- **Sicurezza** per vulnerabilità

## Link

- [2.1.0]: https://github.com/yourrepo/prism-ml/releases/tag/v2.1.0
- [2.0.0]: https://github.com/yourrepo/prism-ml/releases/tag/v2.0.0
- [1.0.0]: https://github.com/yourrepo/prism-ml/releases/tag/v1.0.0
