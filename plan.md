# Implementation Plan: Peer-Aware ASAG System

## Project Overview
Automated Short Answer Grading (ASAG) system using a two-pass peer-aware methodology.
Target: RMSE < 0.85 on Mohler dataset. Deployable on minimal hardware (4GB RAM, no GPU).

---

## Architecture Summary

```
Student Responses
      │
      ▼
[Pass 1: Dynamic Key Builder]
  - Preprocess all responses
  - Blend reference answer + corpus TF stats (÷ strictness)
  - Demote question words (/1.8)
      │
      ▼
[Pass 2: Similarity Grader]
  - TF-IDF vectorize each response
  - Cosine similarity vs dynamic key
  - Polynomial regression (degree 7) → grade [0–5]
      │
      ▼
Grade Output
```

---

## Phase 1 — Core Algorithm (Priority: Critical)

### 1.1 Preprocessing Module (`src/preprocessing.py`)
- [ ] Lowercase + punctuation removal
- [ ] Tokenization → stopword removal (NLTK)
- [ ] Porter stemming
- [ ] Question-word demotion (identify overlap between question tokens and response tokens)

### 1.2 TF-IDF Vectorization (`src/vectorizer.py`)
- [ ] Build vocabulary from full response corpus
- [ ] Compute TF per document, IDF across corpus
- [ ] Return sparse CSR matrix
- [ ] Use scikit-learn `TfidfVectorizer` as backend (configure `sublinear_tf=False`)

### 1.3 Dynamic Key Builder (`src/key_builder.py`)
```python
def build_dynamic_key(reference, corpus, strictness=20):
    key = binary_vector(reference, vocabulary)
    corpus_tf = sum(term_frequencies(corpus)) / strictness
    key = key + corpus_tf
    key[question_words] /= 1.8
    return key
```
- [ ] Binary vector for reference answer terms
- [ ] Aggregate corpus TF, scale by `1/strictness`
- [ ] Additive blend, then question-word demotion
- [ ] Expose `strictness` as configurable parameter (default 20)

### 1.4 Similarity Calculator (`src/similarity.py`)
```python
# response_vec = binary_presence + tfidf_weights
similarity = dot(response_vec, key_vec) / (norm(response_vec) * norm(key_vec))
# NaN → 0.0
```
- [ ] Combine binary + TF-IDF for response vector
- [ ] L2-normalize both vectors before dot product
- [ ] Guard: return 0.0 on zero-vector

### 1.5 Grade Mapper (`src/grade_mapper.py`)
- [ ] Train polynomial regression (degree 7) on Mohler similarity→score pairs
- [ ] `PolynomialFeatures` + `LinearRegression` from scikit-learn
- [ ] Clip output to `[0, max_score]`
- [ ] Persist model with `joblib`

### 1.6 Benchmark Evaluation (`scripts/evaluate.py`)
- [ ] Load Mohler dataset (2,264 responses, 79 questions)
- [ ] Question-level cross-validation (train on N-1 questions, test on 1)
- [ ] Report RMSE, Pearson correlation
- [ ] Target: RMSE ≤ 0.81

---

## Phase 2 — Data Layer (Priority: High)

### 2.1 Database Schema (`db/schema.sql`)
Tables:
- `questions` (id, text, reference_answer, subject, created_at)
- `responses` (id, question_id, student_id, text, submitted_at)
- `grades` (id, response_id, job_id, similarity_score, predicted_grade, model_version)
- `grading_jobs` (id, question_id, status, strictness, created_at, completed_at)
- `users` (id, email, role[admin/teacher/student], hashed_password)

### 2.2 ORM Models (`src/models.py`)
- [ ] SQLAlchemy 2.0 declarative models for all tables
- [ ] Alembic migrations

---

## Phase 3 — API Layer (Priority: High)

### 3.1 FastAPI Application (`src/api/`)

| Endpoint | Method | Handler |
|---|---|---|
| `/api/v1/questions` | POST | Create question + reference |
| `/api/v1/questions/{id}` | GET | Fetch question |
| `/api/v1/responses` | POST | Submit response |
| `/api/v1/grade/single` | POST | Immediate single grade |
| `/api/v1/grade/batch` | POST | Enqueue batch job |
| `/api/v1/jobs/{id}` | GET | Job status |
| `/api/v1/results/{job_id}` | GET | Results |
| `/api/v1/export/{job_id}` | GET | CSV/Excel download |

### 3.2 Authentication
- [ ] JWT tokens (python-jose + passlib)
- [ ] Role middleware: admin / teacher / student
- [ ] Rate limiting (slowapi)

### 3.3 Async Batch Processing
- [ ] Celery worker consuming Redis queue
- [ ] Checkpoint progress in `grading_jobs` table
- [ ] Retry on transient failure (max 3)

---

## Phase 4 — Frontend (`frontend/`)

### 4.1 React + Vite Setup
- [ ] Pages: Login, Dashboard, Questions, Grade Batch, Results, Admin
- [ ] Material-UI components
- [ ] Redux Toolkit for state (auth, jobs, grades)
- [ ] Axios for API calls

### 4.2 Key Teacher Flows
1. Upload questions CSV → preview → confirm
2. Upload responses CSV → configure strictness → grade → monitor job
3. Review grades table → override individual grades → export

---

## Phase 5 — Cloud Deployment (Priority: Medium)

### Gaps vs. original sketch (now resolved below)
| Gap | Resolution |
|---|---|
| No resource provisioning steps | §5.0 prerequisites + §5.1 resource group |
| No container registry | §5.2 ACR build + push |
| No managed DB / Redis | §5.3 Flexible Server + Azure Cache |
| No secrets management | §5.4 Key Vault + App Settings |
| No API/frontend hosting | §5.5 App Service |
| No Databricks provisioning | §5.6 Databricks workspace setup |
| No CI/CD pipeline | §5.7 GitHub Actions workflow |
| No monitoring | §5.8 Application Insights |
| Docker Compose was listed as cloud | Moved to local dev only (§5.9) |

---

### 5.0 Prerequisites
- [ ] Azure CLI installed and logged in (`az login`)
- [ ] Docker Desktop running locally
- [ ] `pyproject.toml` dependencies locked (`uv lock` or `pip freeze > requirements.txt`)
- [ ] Polynomial regression model trained and serialised (`grade_mapper.joblib`)
- [ ] Environment variables template committed as `.env.example` (never `.env`)

---

### 5.1 Azure Databricks Notebook (`notebooks/asag_spark.py`)
- [ ] PySpark UDF wrapping the grading pipeline
- [ ] Distributed TF-IDF on Spark MLlib
- [ ] Auto-scale: Standard_DS3_v2, 2–10 nodes
- [ ] Azure Blob Storage for model artifacts
- [ ] **Missing previously**: workspace provisioning, cluster config, and Blob mount (see §Step-by-Step below)

### 5.2 Azure Container Registry
- [ ] Build and tag `api`, `worker`, `frontend` images
- [ ] Push to ACR (`asagregistry.azurecr.io`)

### 5.3 Managed Data Services
- [ ] Azure Database for PostgreSQL — Flexible Server (B2ms, 32 GB storage)
- [ ] Azure Cache for Redis (C1 Standard)
- [ ] Run `db/schema.sql` migrations against Flexible Server on first deploy

### 5.4 Secrets Management
- [ ] Azure Key Vault (`asag-kv`) storing: `DATABASE_URL`, `REDIS_URL`, `SECRET_KEY`, `ACR_PASSWORD`
- [ ] App Service identity granted `Key Vault Secrets User` role

### 5.5 Application Hosting
- [ ] App Service Plan (Linux, B2 SKU) hosting FastAPI container
- [ ] Second App Service slot hosting Nginx/React frontend
- [ ] Celery worker as a separate App Service (always-on)

### 5.6 Databricks Workspace
- [ ] Premium tier workspace in same region as Blob Storage
- [ ] Cluster policy: `Standard_DS3_v2`, autoscale 2–10 workers, auto-terminate 30 min
- [ ] DBFS mount to Azure Blob (`wasbs://models@asagstorage.blob.core.windows.net/`)
- [ ] `asag_spark.py` notebook uploaded; schedule via Databricks Job (daily or on-demand)

### 5.7 CI/CD Pipeline (`.github/workflows/deploy.yml`)
- [ ] Trigger: push to `main`
- [ ] Steps: test → build images → push to ACR → restart App Services
- [ ] Secrets stored in GitHub repo secrets (mirror of Key Vault values)

### 5.8 Monitoring
- [ ] Application Insights workspace linked to both App Services
- [ ] Alert rule: HTTP 5xx rate > 1% → email notification
- [ ] Log Analytics query for RMSE drift (log prediction metadata)

### 5.9 Local Development Only — Docker Compose
```yaml
services:
  api:      FastAPI (port 8000)
  worker:   Celery
  redis:    Redis 7
  db:       PostgreSQL 15
  frontend: Nginx serving React build
```
> This is for local integration testing only. Cloud uses managed services (§5.3, §5.5).

---

## Step-by-Step Azure Deployment Guide

### Stage 1 — Bootstrap Azure Infrastructure

```bash
# 1. Login and set subscription
az login
az account set --subscription "<YOUR_SUBSCRIPTION_ID>"

# 2. Create resource group (choose region close to users)
az group create --name asag-rg --location eastus

# 3. Create Azure Blob Storage account for model artifacts
az storage account create \
  --name asagstorage \
  --resource-group asag-rg \
  --sku Standard_LRS \
  --kind StorageV2

az storage container create \
  --name models \
  --account-name asagstorage

# 4. Upload trained model artifact
az storage blob upload \
  --account-name asagstorage \
  --container-name models \
  --name grade_mapper.joblib \
  --file ./grade_mapper.joblib
```

---

### Stage 2 — Provision Managed Data Services

```bash
# 5. Azure Database for PostgreSQL — Flexible Server
az postgres flexible-server create \
  --resource-group asag-rg \
  --name asag-postgres \
  --sku-name Standard_B2ms \
  --storage-size 32 \
  --admin-user asagadmin \
  --admin-password "<STRONG_PASSWORD>" \
  --version 15

# Allow Azure services to connect
az postgres flexible-server firewall-rule create \
  --resource-group asag-rg \
  --name asag-postgres \
  --rule-name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

# Run schema migrations (from local machine with server firewall open to your IP)
psql "host=asag-postgres.postgres.database.azure.com \
      user=asagadmin password=<STRONG_PASSWORD> \
      dbname=asagdb sslmode=require" \
  -f db/schema.sql

# 6. Azure Cache for Redis (C1 Standard tier)
az redis create \
  --resource-group asag-rg \
  --name asag-redis \
  --sku Standard \
  --vm-size c1 \
  --location eastus
```

---

### Stage 3 — Key Vault and Secrets

```bash
# 7. Create Key Vault
az keyvault create \
  --name asag-kv \
  --resource-group asag-rg \
  --location eastus

# 8. Store secrets (retrieve connection strings from above steps)
REDIS_KEY=$(az redis list-keys --name asag-redis \
  --resource-group asag-rg --query primaryKey -o tsv)

DB_URL="postgresql://asagadmin:<STRONG_PASSWORD>@asag-postgres.postgres.database.azure.com/asagdb?sslmode=require"
REDIS_URL="rediss://:${REDIS_KEY}@asag-redis.redis.cache.windows.net:6380/0"

az keyvault secret set --vault-name asag-kv --name DATABASE-URL   --value "$DB_URL"
az keyvault secret set --vault-name asag-kv --name REDIS-URL       --value "$REDIS_URL"
az keyvault secret set --vault-name asag-kv --name SECRET-KEY      --value "$(openssl rand -hex 32)"
```

---

### Stage 4 — Container Registry and Images

```bash
# 9. Create Azure Container Registry
az acr create \
  --resource-group asag-rg \
  --name asagregistry \
  --sku Basic \
  --admin-enabled true

ACR_PASSWORD=$(az acr credential show \
  --name asagregistry --query "passwords[0].value" -o tsv)

az keyvault secret set --vault-name asag-kv --name ACR-PASSWORD --value "$ACR_PASSWORD"

# 10. Build and push images (run from repo root)
az acr login --name asagregistry

docker build -t asagregistry.azurecr.io/asag-api:latest    -f docker/Dockerfile.api .
docker build -t asagregistry.azurecr.io/asag-worker:latest -f docker/Dockerfile.worker .
docker build -t asagregistry.azurecr.io/asag-frontend:latest \
  --build-arg VITE_API_URL=https://asag-api.azurewebsites.net \
  -f frontend/Dockerfile .

docker push asagregistry.azurecr.io/asag-api:latest
docker push asagregistry.azurecr.io/asag-worker:latest
docker push asagregistry.azurecr.io/asag-frontend:latest
```

---

### Stage 5 — App Service Hosting

```bash
# 11. Create App Service Plan (Linux, B2 SKU)
az appservice plan create \
  --name asag-plan \
  --resource-group asag-rg \
  --is-linux \
  --sku B2

# 12. Deploy FastAPI container
az webapp create \
  --resource-group asag-rg \
  --plan asag-plan \
  --name asag-api \
  --deployment-container-image-name asagregistry.azurecr.io/asag-api:latest

# Wire Key Vault secrets as App Settings (Key Vault references)
az webapp config appsettings set \
  --resource-group asag-rg \
  --name asag-api \
  --settings \
    DATABASE_URL="@Microsoft.KeyVault(VaultName=asag-kv;SecretName=DATABASE-URL)" \
    REDIS_URL="@Microsoft.KeyVault(VaultName=asag-kv;SecretName=REDIS-URL)" \
    SECRET_KEY="@Microsoft.KeyVault(VaultName=asag-kv;SecretName=SECRET-KEY)" \
    MODEL_PATH="wasbs://models@asagstorage.blob.core.windows.net/grade_mapper.joblib" \
    WEBSITES_PORT=8000

# Grant system identity access to Key Vault
API_PRINCIPAL=$(az webapp identity assign \
  --name asag-api --resource-group asag-rg --query principalId -o tsv)
az keyvault set-policy --name asag-kv \
  --object-id "$API_PRINCIPAL" --secret-permissions get list

# 13. Deploy Celery worker
az webapp create \
  --resource-group asag-rg \
  --plan asag-plan \
  --name asag-worker \
  --deployment-container-image-name asagregistry.azurecr.io/asag-worker:latest

az webapp config appsettings set \
  --resource-group asag-rg --name asag-worker \
  --settings \
    DATABASE_URL="@Microsoft.KeyVault(VaultName=asag-kv;SecretName=DATABASE-URL)" \
    REDIS_URL="@Microsoft.KeyVault(VaultName=asag-kv;SecretName=REDIS-URL)" \
    COMMAND_OVERRIDE="celery -A src.api.tasks worker --loglevel=info"

WORKER_PRINCIPAL=$(az webapp identity assign \
  --name asag-worker --resource-group asag-rg --query principalId -o tsv)
az keyvault set-policy --name asag-kv \
  --object-id "$WORKER_PRINCIPAL" --secret-permissions get list

# 14. Deploy React frontend
az webapp create \
  --resource-group asag-rg \
  --plan asag-plan \
  --name asag-frontend \
  --deployment-container-image-name asagregistry.azurecr.io/asag-frontend:latest
```

---

### Stage 6 — Azure Databricks (Batch Grading)

```bash
# 15. Create Databricks workspace (Premium tier for DBFS mounts)
az databricks workspace create \
  --resource-group asag-rg \
  --name asag-databricks \
  --location eastus \
  --sku premium

# 16. After workspace is ready, open the UI and:
#     a. Go to Compute → Create Cluster
#        - Runtime: 13.3 LTS (Spark 3.4, Python 3.10)
#        - Worker type: Standard_DS3_v2
#        - Min workers: 2, Max workers: 10
#        - Auto-terminate: 30 minutes
#     b. Install library: pip install scikit-learn nltk
#
# 17. Mount Blob Storage (run once in a Databricks notebook cell):
```

```python
# In Databricks notebook — mount Blob Storage
storage_account = "asagstorage"
container       = "models"
sas_key         = dbutils.secrets.get(scope="asag-kv", key="STORAGE-SAS")

dbutils.fs.mount(
    source=f"wasbs://{container}@{storage_account}.blob.core.windows.net/",
    mount_point="/mnt/asag-models",
    extra_configs={
        f"fs.azure.sas.{container}.{storage_account}.blob.core.windows.net": sas_key
    }
)

# 18. Upload asag_spark.py notebook via Databricks CLI
# pip install databricks-cli
# databricks configure --token
databricks workspace import notebooks/asag_spark.py \
  /Shared/asag_spark --language PYTHON --overwrite

# 19. Schedule as a Databricks Job (daily or on-demand via API)
# Use Databricks UI: Workflows → Create Job → select /Shared/asag_spark
```

---

### Stage 7 — CI/CD Pipeline

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]" && pytest tests/ --tb=short

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Login to ACR
        uses: docker/login-action@v3
        with:
          registry: asagregistry.azurecr.io
          username: asagregistry
          password: ${{ secrets.ACR_PASSWORD }}

      - name: Build and push images
        run: |
          docker build -t asagregistry.azurecr.io/asag-api:${{ github.sha }} \
            -t asagregistry.azurecr.io/asag-api:latest -f docker/Dockerfile.api .
          docker build -t asagregistry.azurecr.io/asag-worker:${{ github.sha }} \
            -t asagregistry.azurecr.io/asag-worker:latest -f docker/Dockerfile.worker .
          docker build -t asagregistry.azurecr.io/asag-frontend:${{ github.sha }} \
            -t asagregistry.azurecr.io/asag-frontend:latest \
            --build-arg VITE_API_URL=https://asag-api.azurewebsites.net \
            -f frontend/Dockerfile .
          docker push asagregistry.azurecr.io/asag-api --all-tags
          docker push asagregistry.azurecr.io/asag-worker --all-tags
          docker push asagregistry.azurecr.io/asag-frontend --all-tags

      - name: Restart App Services
        uses: azure/CLI@v2
        with:
          azcliversion: latest
          inlineScript: |
            az webapp restart --name asag-api      --resource-group asag-rg
            az webapp restart --name asag-worker   --resource-group asag-rg
            az webapp restart --name asag-frontend --resource-group asag-rg
        env:
          AZURE_CREDENTIALS: ${{ secrets.AZURE_CREDENTIALS }}
```

**Required GitHub Secrets:**
- `ACR_PASSWORD` — from `az acr credential show --name asagregistry`
- `AZURE_CREDENTIALS` — from `az ad sp create-for-rbac --sdk-auth`

---

### Stage 8 — Monitoring

```bash
# 20. Create Application Insights workspace
az monitor app-insights component create \
  --app asag-insights \
  --location eastus \
  --resource-group asag-rg \
  --application-type web

INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app asag-insights --resource-group asag-rg \
  --query instrumentationKey -o tsv)

# 21. Add instrumentation key to API App Settings
az webapp config appsettings set \
  --resource-group asag-rg --name asag-api \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY="$INSTRUMENTATION_KEY"

# 22. Create HTTP 5xx alert rule
az monitor metrics alert create \
  --name "asag-5xx-alert" \
  --resource-group asag-rg \
  --scopes $(az webapp show --name asag-api --resource-group asag-rg --query id -o tsv) \
  --condition "avg Http5xx > 1" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action $(az monitor action-group create \
    --name asag-alert-group \
    --resource-group asag-rg \
    --short-name asagalert \
    --email-receiver name=admin email=data-science03@siya.com \
    --query id -o tsv)
```

---

### Stage 9 — Smoke Test

```bash
# 23. Hit the live API health endpoint
curl https://asag-api.azurewebsites.net/api/v1/health

# 24. Run a single-grade smoke test
curl -X POST https://asag-api.azurewebsites.net/api/v1/grade/single \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is photosynthesis?",
    "reference": "Photosynthesis is the process plants use to convert light into glucose.",
    "response": "Plants use sunlight to make food from carbon dioxide and water.",
    "strictness": 20
  }'
# Expected: {"predicted_grade": <float 0-5>, "similarity_score": <float>}

# 25. Verify frontend loads
curl -I https://asag-frontend.azurewebsites.net
```

---

### Azure Resource Summary

| Resource | SKU / Tier | Est. Monthly Cost |
|---|---|---|
| App Service Plan (shared by 3 apps) | B2 Linux | ~$75 |
| Azure Database for PostgreSQL | B2ms Flexible | ~$55 |
| Azure Cache for Redis | C1 Standard | ~$55 |
| Azure Blob Storage | Standard LRS | ~$5 |
| Azure Container Registry | Basic | ~$5 |
| Azure Databricks | Premium (pay-per-DBU) | ~$50–200 |
| Application Insights | Pay-as-you-go | ~$5 |
| **Total estimate** | | **~$250–400 / month** |

---

### Rollback Procedure

```bash
# Revert API to previous image tag (replace <PREV_SHA> with last good commit SHA)
az webapp config container set \
  --name asag-api --resource-group asag-rg \
  --docker-custom-image-name asagregistry.azurecr.io/asag-api:<PREV_SHA>

az webapp restart --name asag-api --resource-group asag-rg
```

---

## Phase 6 — Optional: HuggingFace Embeddings (`src/embeddings.py`)

- [ ] Pluggable interface: `SimilarityBackend` abstract class
- [ ] `TFIDFBackend` (default, no GPU)
- [ ] `SentenceTransformerBackend` (optional, requires `sentence-transformers`)
  - Models: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `paraphrase-multilingual-MiniLM-L12-v2`
- [ ] Backend selected via config flag `EMBEDDING_BACKEND=tfidf|sentence_transformer`

---

## File Structure

```
Capstone_final/
├── src/
│   ├── preprocessing.py      # TextNormalizer, Stemmer, QuestionWordDemoter
│   ├── vectorizer.py         # TF-IDF wrapper
│   ├── key_builder.py        # Dynamic key construction (core algorithm)
│   ├── similarity.py         # Cosine similarity
│   ├── grade_mapper.py       # Polynomial regression grade mapping
│   ├── embeddings.py         # Optional HuggingFace backend
│   ├── models.py             # SQLAlchemy ORM
│   └── api/
│       ├── main.py           # FastAPI app
│       ├── routes/           # Endpoint handlers
│       ├── auth.py           # JWT + roles
│       └── tasks.py          # Celery tasks
├── frontend/                 # React + Vite app
├── notebooks/
│   └── asag_spark.py         # Azure Databricks notebook
├── scripts/
│   ├── evaluate.py           # Mohler benchmark runner
│   └── train_regressor.py    # Train + save polynomial model
├── tests/
│   ├── unit/                 # UT-PP-*, UT-VEC-*, UT-SIM-*
│   └── integration/          # IT-GP-*
├── db/
│   └── schema.sql
├── docker-compose.yml
├── main.py                   # CLI entry point / demo
└── pyproject.toml
```

---

## Execution Order

```
1. src/preprocessing.py        → unit test UT-PP-01..06
2. src/vectorizer.py           → unit test UT-VEC-01..04
3. src/key_builder.py          → manual validation on toy corpus
4. src/similarity.py           → unit test UT-SIM-01..04
5. src/grade_mapper.py         → train on Mohler, verify RMSE ≤ 0.81
6. scripts/evaluate.py         → full benchmark
7. src/models.py + db/         → DB layer
8. src/api/                    → API + auth + Celery
9. frontend/                   → React UI
10. docker-compose.yml         → integration test
11. notebooks/asag_spark.py    → cloud deployment
```

---

## Success Criteria

| Metric | Target | Source |
|---|---|---|
| RMSE (Mohler, LOQO CV) | ≤ 0.93 | ST-ACC-01 |
| Pearson correlation | ≥ 0.55 | ST-ACC-02 |
| Single response latency | < 100ms | ST-PERF-01 |
| Batch 1000 responses | < 30s | ST-PERF-03 |
| Peak memory (10k responses) | < 2GB | ST-PERF-05 |
| Unit test coverage | ≥ 80% | NFR-06.2 |
| Min hardware | 4GB RAM, no GPU | NFR-03 |

---

## Key Design Decisions

1. **Strictness=20 default** — preliminary R implementation validated this value for RMSE 0.81.
2. **Binary + TF-IDF fusion** for response vectors — improves recall on rare but relevant terms.
3. **Degree-7 polynomial** regression — captures non-linear similarity-to-grade mapping observed in Mohler data.
4. **Question-word demotion ÷1.8** — prevents students gaming the system by repeating question words.
5. **Pluggable embedding backend** — allows upgrade path to sentence transformers without API changes.
6. **Offline-first** — all core functionality works without internet; cloud is additive, not required.
