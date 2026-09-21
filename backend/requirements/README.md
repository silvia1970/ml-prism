# Requirements - PRISM ML

Questa cartella contiene tutti i file di dipendenze Python organizzati per caso d'uso.

## 📦 File Disponibili

### `requirements.txt`
**Dipendenze complete per Machine Learning**
- Include tutte le librerie ML (PyTorch, scikit-learn, XGBoost, ecc.)
- Librerie di analisi dati (pandas, numpy, scipy)
- Tools di visualizzazione (matplotlib, seaborn, plotly)
- Librerie di interpretabilità (SHAP, LIME, ELI5)

**Installazione:**
```bash
pip install -r requirements/requirements.txt
```

**Dimensione:** ~2.5GB dopo installazione  
**Tempo:** ~10-15 minuti  
**Uso:** Sviluppo ML completo, training modelli, analisi dati

---

### `requirements-api.txt`
**Dipendenze minime per API in produzione**
- Flask + Flask-CORS (web framework)
- Auth0 authentication (python-jose)
- Data handling (pandas, numpy)
- Documentazione API (flasgger)

**Installazione:**
```bash
pip install -r requirements/requirements-api.txt
```

**Dimensione:** ~200MB dopo installazione  
**Tempo:** ~2-3 minuti  
**Uso:** Deployment API, server produzione

---

### `requirements-dev.txt`
**Tools di sviluppo e testing**
- Testing (pytest, pytest-cov)
- Linting (black, flake8, mypy)
- Development tools (ipython, jupyter)

**Installazione:**
```bash
pip install -r requirements/requirements-dev.txt
```

**Nota:** Questo file include automaticamente `requirements.txt`

**Uso:** Sviluppo locale, testing, CI/CD

---

### `pytorch_deps.txt`
**Dipendenze specifiche PyTorch**
- PyTorch ecosystem
- CUDA dependencies (se necessario)

**Uso:** Installazione personalizzata di PyTorch per GPU/CPU specifici

---

## 🚀 Installazione Rapida

### Setup Completo (Sviluppo ML)
```bash
# Crea virtual environment .venv (se non esiste)
python3 -m venv .venv

# Attiva environment
source .venv/bin/activate

# Installa dipendenze complete
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt
```

### Setup API (Produzione)
```bash
# Crea virtual environment .venv (se non esiste)
python3 -m venv .venv

# Attiva environment
source .venv/bin/activate

# Installa solo dipendenze API
pip install -r requirements/requirements-api.txt
```

### Setup Frontend
```bash
cd frontend
npm install
```

---

## 🔄 Aggiornamento Dipendenze

### Aggiorna requirements.txt
```bash
pip freeze > requirements/requirements.txt
```

### Verifica dipendenze obsolete
```bash
pip list --outdated
```

### Aggiorna tutte le dipendenze
```bash
pip install --upgrade -r requirements/requirements.txt
```

---

## 📊 Dimensioni e Tempi (Riferimento)

| File | Pacchetti | Dimensione | Tempo |
|------|-----------|------------|-------|
| requirements.txt | ~95 | ~2.5GB | ~10-15min |
| requirements-api.txt | ~10 | ~200MB | ~2-3min |
| requirements-dev.txt | ~105 | ~2.6GB | ~12-17min |

*Tempi stimati su connessione 50Mbps, variabili in base alla cache pip*

---

## ⚠️ Note Importali

1. **Virtual Environment**: Sempre usare un virtual environment per evitare conflitti
2. **Python Version**: Testato con Python 3.8+, raccomandato 3.10+
3. **Auth0**: `python-jose[cryptography]` richiede compilazione C (potrebbe richiedere build-essential)
4. **PyTorch**: Per installazioni GPU-specific, vedere `pytorch_deps.txt`
5. **Windows**: Alcuni pacchetti potrebbero richiedere Visual C++ Build Tools

---

## 🐛 Troubleshooting

### Errore: "No module named 'dotenv'"
```bash
pip install python-dotenv
```

### Errore: "python-jose compilation failed"
```bash
# Ubuntu/Debian
sudo apt-get install build-essential libssl-dev libffi-dev python3-dev

# Mac
brew install openssl

# Poi reinstalla
pip install python-jose[cryptography]
```

### Errore: "Cannot import torch"
```bash
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
