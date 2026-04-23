# Automated Short Answer Grading: A Peer-Aware Approach for Resource-Constrained Educational Environments

**Project Report**

**Authors:**
- Gokularajan R
- Prashitha J R

**Institution:** VIT University, School of Computer Science and Engineering

**Date:** April 2026

---

# Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
  - [1.1 Background](#11-background)
  - [1.2 Motivations](#12-motivations)
  - [1.3 Scope of the Project](#13-scope-of-the-project)
- [2. Project Description and Goals](#2-project-description-and-goals)
  - [2.1 Literature Review](#21-literature-review)
  - [2.2 Gaps Identified](#22-gaps-identified)
  - [2.3 Objectives](#23-objectives)
  - [2.4 Problem Statement](#24-problem-statement)
  - [2.5 Project Plan](#25-project-plan)
- [3. Technical Specification](#3-technical-specification)
  - [3.1 Requirements](#31-requirements)
  - [3.2 Feasibility Study](#32-feasibility-study)
  - [3.3 System Specification](#33-system-specification)
- [4. Design Approach and Details](#4-design-approach-and-details)
  - [4.1 System Architecture](#41-system-architecture)
  - [4.2 Design](#42-design)
- [5. Methodology and Testing](#5-methodology-and-testing)
  - [5.1 Module Description](#51-module-description)
  - [5.2 Testing](#52-testing)
- [6. References](#6-references)

---

## Abstract

Short-answer grading at scale is a genuine operational problem. A single MOOC course with 100,000 enrolled students and 10 short-answer questions per examination produces a million responses to evaluate—manually. This project developed and deployed a working automated grading system that addresses this problem without requiring expensive hardware or cloud subscriptions.

The core contribution is peer-aware grading. Rather than comparing each student response directly against a static reference answer, the system first analyses the entire submission batch to construct a dynamic answer key. Terms that many students mention receive partial credit even if absent from the reference; terms in the reference always carry full weight. This more closely mirrors how experienced human graders calibrate expectations across a cohort.

The grading model takes six features per response—two peer-aware similarity scores (with and without question-word demotion), length ratio, unigram Jaccard overlap, keyword coverage, and bigram Jaccard overlap—and feeds them into a trained Gradient Boosting Regressor. On the Mohler benchmark (2,264 responses), the system achieves an RMSE of 0.81, surpassing every traditional machine learning baseline reported in the literature.

The system is fully deployed end-to-end on Azure App Service. Teachers can create questions, export them as portable JSON files, and grade every student response in a single operation. Students sit examinations through a browser with auto-saved answers and free navigation between questions. The backend is FastAPI with JSON file storage mounted on Azure Files; the frontend is plain HTML/CSS/JS. Total hosting cost is approximately $20 per month. No GPU, no managed database, no specialized infrastructure.

**Keywords:** Automated Short Answer Grading, ASAG, TF-IDF, Peer-Aware Grading, Dynamic Key Construction, Gradient Boosting, Educational Technology, Natural Language Processing

---

## 1. Introduction

### 1.1 Background

India's education system operates at a scale that is difficult to fully appreciate: 1.5 million schools, 250 million students, more than 40,000 higher education institutions. Platforms such as NPTEL, SWAYAM, and DIKSHA have extended digital learning to remote parts of the country, adding millions more learners to an already strained assessment infrastructure.

The evaluation bottleneck is apparent. Multiple-choice questions can be graded instantaneously at any scale. Short-answer questions—the assessments that test genuine understanding rather than pattern recognition—require a human reader. A teacher managing five sections of 40 students faces 200 answer sheets per examination. At 10 to 15 short-answer questions per paper, that amounts to 2,000 to 3,000 individual responses per sitting.

Automated Short Answer Grading (ASAG) has been an active research area since the 1960s. The field has progressed through keyword matching, corpus-based methods, statistical machine learning, and now transformer architectures. The accuracy trajectory is real: fine-tuned RoBERTa models achieve RMSE around 0.70 on standard benchmarks.

The difficulty is that those models require GPU hardware to train and substantial compute to run at scale. A government school in Bihar does not have that. A rural college in Tamil Nadu does not have that. The research community has largely optimized for benchmark performance without examining whether the resulting systems can be deployed where the need is concentrated.

### 1.2 Motivations

The gap between state-of-the-art ASAG research and what most Indian educational institutions can practically deploy motivated this project.

**The infrastructure reality**

Most government and aided schools in India operate with limited or no high-performance computing hardware, intermittent internet connectivity in rural areas, minimal technical support staff, and operating budgets that do not accommodate ongoing cloud costs.

The National Education Policy 2020 mandates competency-based, continuous assessment. That is a sound policy position. Continuous assessment of open-ended responses, however, requires either perpetually available teaching staff or automated tools that function on the hardware institutions already possess.

**The research-practice gap**

The dominant approach in current ASAG research involves pre-trained transformer models with 100M+ parameters, fine-tuning on GPU clusters, and inference costs that scale with response volume. For a rural school or a government college operating under budget constraints, these systems are not realistically deployable.

The central claim this project tests is that a well-designed traditional NLP approach, enhanced with peer-awareness and a multi-feature ensemble model, can achieve accuracy close enough to transformer performance that the residual gap becomes acceptable in practice—and in exchange, the system runs on a standard laptop or a $13/month cloud instance with no internet dependency after setup.

**Why peer-awareness matters**

Experienced teachers do not grade each response in isolation. Reading through a batch, they develop calibrated expectations about what the cohort understood and where understanding broke down. A response that correctly identifies the core concept may receive partial credit in one context and not another, depending on how the rest of the class answered.

Standard ASAG systems do not model this. Every response is evaluated against a fixed reference answer with no information from the submission batch. The dynamic key mechanism in this system formalizes that calibration: the grading criteria adapt to the actual vocabulary and expression patterns the cohort produced.

### 1.3 Scope of the Project

This project covers the complete stack: algorithm, web application, and cloud deployment.

**Grading algorithm**
- Two-pass peer-aware grading methodology
- Six-feature extraction pipeline (peer-aware similarity, Jaccard overlap, length ratio, keyword coverage, bigrams)
- Gradient Boosting Regressor trained on labeled short-answer datasets
- Question-word demotion to prevent gaming through question paraphrasing
- Dynamic key construction with configurable strictness parameter

**Web application**
- Teacher interface: question bank management, JSON export and import, one-click batch grading
- Student interface: browser-based examination with auto-save and free question navigation
- Teacher dashboard: three views (class overview, per-question breakdown, per-student detail)
- CSV export at both class and individual student level

**Deployment**
- FastAPI backend served by uvicorn, containerized with Docker
- Azure App Service (Linux B1) for cloud hosting
- Azure Files mount for persistent JSON data and model file storage
- Azure Container Registry for Docker image management
- Standalone deployment on commodity hardware also supported, with offline operation

**Explicitly out of scope**
- Handwriting recognition (assumes typed text input)
- Plagiarism detection
- Essay-length response grading
- Examination proctoring or identity verification
- Languages other than English in this version

---

## 2. Project Description and Goals

### 2.1 Literature Review

#### 2.1.1 Survey Papers and Foundational Work

Burrows, Gurevych, and Stein (2014) surveyed over 80 ASAG papers and proposed a taxonomy covering five eras: concept mapping (1966–1999), information extraction (2000–2006), corpus-based methods (2006–2009), machine learning (2009–2013), and evaluation challenges. That taxonomy remains useful, with a sixth era of transformer dominance now clearly established.

A 2022 arXiv survey traced the trajectory from word embeddings to transformers, organizing approaches into word embedding methods, sequential models (LSTM, GRU), and attention-based architectures (BERT, RoBERTa). A 2018 systematic review of 44 papers found that statistical models remain widely used and that standardized evaluation datasets are scarce—a finding that still holds, since most published systems evaluate on different subsets across different corpora.

#### 2.1.2 TF-IDF and Traditional Methods

TF-IDF remains relevant because it is interpretable, computationally cheap, and requires no GPU. Mohler and Mihalcea (2009) explored unsupervised text similarity for short-answer grading, comparing knowledge-based and corpus-based approaches. Their work on integrating feedback from student answers was an early signal that peer information carries useful grading signal—other students' responses contain information about what constitutes a reasonable answer, not only the reference.

Albitar et al. (2014) proposed an improved TF-IDF similarity function demonstrating meaningful gains over standard cosine similarity on the Microsoft paraphrase corpus. Lan et al. (2022) acknowledged TF-IDF's fundamental limitation in handling synonymous expressions—a constraint this project partially addresses through the peer-aware key rather than expensive semantic embeddings.

TF-IDF baselines on the Mohler dataset consistently fall around Pearson 0.55 and RMSE near 1.0. That is the floor this project needed to surpass.

#### 2.1.3 Semantic Similarity Measures

Beyond TF-IDF, researchers have explored semantic similarity measures. The Universal Sentence Encoder, Word Mover's Distance, and edit similarity all have published proponents. Systems combining multiple measures through learned weighting generally outperform any single metric. This is the same motivation behind the six-feature approach here: no individual signal fully characterizes answer quality, but a learned combination does.

#### 2.1.4 Word Embeddings

Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) introduced dense semantic vector representations as an alternative to sparse bag-of-words methods. Gaddipati (2020) compared ELMo, BERT, GPT, and GPT-2 embeddings with cosine similarity on the Mohler dataset. ELMo outperformed all others—a result that illustrates that raw model scale does not automatically translate to better short-answer grading performance.

#### 2.1.5 Transformer-Based Models

Fine-tuned BERT models showed up to 10% absolute F1 improvement over earlier methods (Sung et al., 2019). Sentence-BERT enabled efficient similarity computation by precomputing sentence embeddings, reducing inference cost for large batches. Thakkar (2021) achieved RMSE 0.70 on the Mohler dataset by fine-tuning RoBERTa-Large—the best published result on that benchmark and the upper bound against which this project is compared.

#### 2.1.6 Large Language Models

Work at ACM LAK 2025 found that GPT-4 with prompt engineering achieves performance comparable to hand-engineered models without requiring any training data. Studies in medical education (BMC Medical Education, 2024) compared GPT-4 and Gemini across 2,288 multi-language responses, finding moderate agreement with human raters. A Nature Scientific Reports (2025) study found correlation up to 0.98 between GPT-4 rankings and human grader rankings in specific settings.

Zero-shot LLM frameworks (Yeung et al., 2025) demonstrate that grading without fine-tuning is now feasible. The practical constraints remain cost—GPT-4 API pricing at examination scale is not negligible—and the requirement for live internet access, which excludes institutions with unreliable connectivity.

#### 2.1.7 Neural Network Architectures

Bidirectional LSTM with attention (AC-BiLSTM) has shown competitive performance across multiple datasets. Siamese networks—architectures that learn pairwise comparison—have been adapted for ASAG, with the Manhattan LSTM variant producing grades from learned similarity functions over word embedding inputs. Kim (2014) demonstrated that shallow CNNs with pre-trained word vectors sometimes outperform deeper architectures, suggesting that the relationship between model complexity and grading quality is not monotonic.

#### 2.1.8 Benchmark Datasets

**Mohler Dataset (2009, 2011):** 79 questions, 2,273 student responses from a data structures course, graded 0–5 by two educators. The primary benchmark for this project. Baseline RMSE: 0.978.

**SemEval-2013 Task 7:** Beetle (3,941 responses on electricity/electronics) and SciEntsBank (~10,000 responses across 15 scientific domains). Three test scenarios: unseen answer, unseen question, unseen domain.

**ASAP Dataset (2012):** ~17,450 essays across eight prompts, evaluated with Quadratic Weighted Kappa. Used for automated essay scoring rather than short-answer grading, but methodologically adjacent.

### 2.2 Gaps Identified

**Gap 1: Computational accessibility**

State-of-the-art ASAG requires GPU hardware, substantial memory, and ongoing cloud compute costs. No prior published work has explicitly optimized for the intersection of competitive accuracy and minimal hardware requirements. This system operates on 4 GB RAM with no GPU, and deploys to the cloud for approximately $20 per month.

**Gap 2: Static grading keys**

All mainstream approaches—from TF-IDF baselines through transformer models—evaluate each response independently against a fixed reference. The submission corpus carries no information. The dynamic key construction in this system is the primary algorithmic contribution: grading criteria adapt to the vocabulary and expression patterns the actual student cohort used.

**Gap 3: The Indian educational context**

ASAG research has produced systems that perform well under laboratory conditions with reliable infrastructure. None has been designed with explicit consideration for Indian schools and colleges where connectivity is intermittent, IT support is minimal, and budget constraints are significant. This system supports both a $20/month cloud deployment and fully offline standalone operation.

**Gap 4: Interpretability**

Most high-performing ASAG systems are black boxes from the teacher's perspective. The reasons behind a particular grade are not visible. This system surfaces all six feature values alongside every grade in the per-question dashboard view—the teacher can inspect the similarity score and keyword coverage that drove each decision.

**Gap 5: End-to-end application**

ASAG papers typically evaluate an algorithm on a benchmark and treat the surrounding application as out of scope. This project built the complete application: the NLP core, the asynchronous grading pipeline, the browser-based student examination interface, and the three-view teacher dashboard.

### 2.3 Objectives

**Primary Objectives — achieved**

1. Develop a peer-aware ASAG algorithm with competitive accuracy on the Mohler benchmark, operating on minimal hardware. **Achieved: RMSE 0.81 on the Mohler dataset, running on consumer-grade hardware.**

2. Implement the algorithm as a production-ready application with teacher and student interfaces, batch grading, and result export. **Achieved.**

3. Design for resource-constrained deployment: standard hardware, no GPU, offline capability. **Achieved: the system runs fully offline after initial setup; no GPU required.**

4. Validate through rigorous benchmark evaluation with question-level cross-validation. **Achieved.**

**Secondary Objectives — achieved or noted**

5. Cloud deployment. **Achieved:** the system runs as a single Docker container on Azure App Service (Linux B1) with persistent storage on Azure Files. Estimated monthly cost: ~$20.

6. Active Moodle and Google Classroom integration. Not implemented in this version. The RESTful API is structured to support future LMS connectors.

7. HuggingFace sentence transformer plugin. Architecture supports it; the plugin was not built.

8. Documentation for Indian institutions. Covered in the deployment guide.

### 2.4 Problem Statement

**Formal statement:**

Given a set of student responses $R = \{r_1, r_2, ..., r_n\}$ to a short-answer question $Q$ with reference answer $A$, develop a grading function $G: R \rightarrow [0, \text{max\_marks}]$ that:

1. Assigns grades consistent with human evaluation (minimizing RMSE against human-graded scores)
2. Adapts evaluation criteria based on the collective characteristics of $R$ (peer-awareness)
3. Executes in $O(n \cdot m)$ time complexity where $m$ is average response length
4. Operates with $O(V)$ space complexity where $V$ is vocabulary size
5. Requires no GPU or specialized hardware

**Contextual statement:**

Government schools and colleges across India lack automated assessment tools that could reduce teacher workload and support the continuous competency-based evaluation that NEP 2020 requires. Existing ASAG solutions are either too computationally expensive for available infrastructure, insufficiently accurate to be trusted for assessment, or too complex to maintain without dedicated technical staff. This project built a system that navigates those constraints: accurate enough to be useful, inexpensive enough to run anywhere, and simple enough that a teacher with basic computer literacy can operate it without IT assistance.

### 2.5 Project Plan

The project was organized across six phases over 16 weeks.

**Phase 1: Research and Requirements (Weeks 1–2)**
- Literature review and dataset acquisition
- Requirements definition
- Development environment setup

**Phase 2: Core Algorithm Development (Weeks 3–5)**
- TF-IDF vectorization and dynamic key construction
- Six-feature extraction pipeline design and implementation
- Gradient Boosting Regressor training and validation on Mohler dataset
- Unit testing of core components

**Phase 3: Application Development (Weeks 6–9)**
- FastAPI backend, storage layer, and asynchronous job management
- Teacher and student web interfaces
- Batch grading pipeline with polling
- Integration testing

**Phase 4: Deployment (Weeks 10–12)**
- Docker containerization
- Azure App Service configuration
- Azure Files mount setup
- Performance benchmarking and load testing

**Phase 5: Evaluation and Validation (Weeks 13–14)**
- Mohler benchmark evaluation
- Question-level cross-validation
- Comparative analysis against published baselines
- User acceptance testing

**Phase 6: Documentation and Final Report (Weeks 15–16)**
- Technical documentation and deployment guide
- Final report preparation and project presentation

```
Gantt Chart:

Week:    1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
Phase 1: ████
Phase 2:       ██████████
Phase 3:                   ████████████████
Phase 4:                                     ██████████
Phase 5:                                              ████████
Phase 6:                                                    ████████
```

---

## 3. Technical Specification

### 3.1 Requirements

#### 3.1.1 Functional Requirements

**FR-01: Question Management**
- FR-01.1: Teachers can create questions with text, reference answer, subject tag, and maximum marks (any value, not fixed to 5).
- FR-01.2: Teachers can delete questions from the question bank.
- FR-01.3: Teachers can export the entire question bank as a portable JSON file.
- FR-01.4: Teachers can import a JSON question paper, accepting both `{ "questions": [...] }` format and bare arrays, with field aliases for externally generated files.

**FR-02: Student Examination Interface**
- FR-02.1: Students enter name and roll number to begin. No password is required; roll number serves as the student identifier throughout the system.
- FR-02.2: Students see questions one at a time with a progress bar indicating position in the paper.
- FR-02.3: Students navigate freely between questions in any order using numbered dot navigation (green = answered, indigo = current, grey = unanswered).
- FR-02.4: Answers auto-save to browser storage after every keystroke. Navigating away or refreshing the page does not lose work.
- FR-02.5: Students submit all answers in a single operation; a warning is displayed if any questions are unanswered before the submission is confirmed.
- FR-02.6: A confirmation screen is shown after submission; the saved draft is cleared from browser storage.

**FR-03: Grading**
- FR-03.1: The system grades individual responses through the six-feature GBM pipeline.
- FR-03.2: "Grade All Questions" retrieves every question, fetches all submissions, and runs batch grading for each question in a single operation.
- FR-03.3: Batch grading runs asynchronously as a background task. The client polls for completion.
- FR-03.4: Grades are clipped to [0, max\_marks] where max\_marks is set per question by the teacher.
- FR-03.5: The strictness parameter for dynamic key construction is configurable (default 20).

**FR-04: Dashboard and Results**
- FR-04.1: Class overview displays total students graded, mean and median total marks, pass rate (≥60% of total possible marks), grade distribution chart (five score bands, 0–100%), and a ranked student table.
- FR-04.2: Per-question view displays response count, mean, median, pass rate, grade distribution, and a per-student table with answer text, grade, percentage, peer-aware similarity score, and keyword coverage.
- FR-04.3: Per-student view displays that student's total marks, percentage, rank among all graded students, a horizontal bar chart by question (colour-coded green/yellow/red), and a complete answer breakdown table.
- FR-04.4: The grade summary caches in localStorage after one "Grade All" run. All three dashboard views read from cache without additional API calls.
- FR-04.5: Class results are exportable as CSV with one row per student and per-question grade columns.
- FR-04.6: Individual student results are exportable as CSV.

**FR-05: Authentication**
- FR-05.1: Teacher access requires an API key validated live against the server. All write operations and sensitive read operations require this key.
- FR-05.2: Student access requires name and roll number. No server-side validation is performed; roll number is the student identifier in all grading records.
- FR-05.3: Public endpoints (list questions, submit answer) require no authentication, so the student examination works without credentials.

**FR-06: API**
- FR-06.1: A RESTful API exposes all grading and question management functions.
- FR-06.2: Teacher operations require the API key in request headers.
- FR-06.3: A CLI entry point (`main.py`) supports single-response grading for debugging, printing all six feature values and the predicted grade.

#### 3.1.2 Non-Functional Requirements

**NFR-01: Performance**
- Single response grading: under 100 ms
- Batch of 1,000 responses: under 30 seconds
- System startup (model load): under 5 seconds

**NFR-02: Accuracy**
- RMSE < 0.85 on Mohler benchmark. *Achieved: 0.81.*
- Pearson correlation > 0.75 with human grades.

**NFR-03: Usability**
- Common tasks (upload questions, grade all, export) completable in three clicks or fewer.
- Student examination functions on any modern browser without plugins or client-side installation.
- Error messages include remediation guidance.

**NFR-04: Reliability**
- Atomic file writes prevent data corruption if the container restarts mid-write.
- Batch job state is persisted to Azure Files; recovery is possible after a container restart.

**NFR-05: Security**
- All user inputs are sanitized before processing.
- Teacher API key required for all write and sensitive read operations.

**NFR-06: Portability**
- Runs on Windows, Linux, and macOS in standalone mode.
- Containerized with Docker for consistent deployment to Azure App Service.
- No internet access required after initial deployment.

**NFR-07: Maintainability**
- Modular architecture with clear separation between the NLP core, API layer, and storage layer.
- Minimum 80% code coverage in unit tests.
- Storage layer is replaceable with MongoDB or PostgreSQL without modifying the NLP or API layers.

### 3.2 Feasibility Study

#### 3.2.1 Technical Feasibility

The system is built on mature, widely-used components. The only novel element—dynamic key construction—is algorithmically straightforward:

| Component | Technology | Maturity | Risk |
|-----------|------------|----------|------|
| Text preprocessing | NLTK | 15+ years | Low |
| TF-IDF vectorization | scikit-learn | 10+ years | Low |
| Cosine/Jaccard similarity | NumPy | 20+ years | Low |
| Gradient Boosting | scikit-learn | 10+ years | Low |
| Dynamic key construction | Custom | Novel | Medium |
| Web API | FastAPI | 5+ years | Low |
| Atomic file storage | Python stdlib | Mature | Low |
| Containerization | Docker | 10+ years | Low |
| Cloud hosting | Azure App Service | 10+ years | Low |

A preliminary R implementation validated the dynamic key approach before the Python production system was built.

**Infrastructure feasibility**

Two deployment modes are supported:

*Standalone:* Runs on any machine with Python installed. No cloud account, no internet after setup. Suitable for schools and colleges with unreliable connectivity.

*Cloud (Azure App Service):* One container on a B1 Linux plan, with Azure Files for persistent storage. Total cost approximately $20/month. Managed container restarts, HTTPS by default, and accessible from any internet-connected browser.

**Technical risk assessment:**

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Algorithm underperforms on new datasets | Medium | High | Question-level cross-validation, configurable strictness |
| Azure Files latency under concurrent writes | Low | Medium | Atomic writes; file-per-collection limits contention |
| Browser compatibility issues | Low | Low | Vanilla JS with no framework dependencies |

**Conclusion:** Technically feasible with low to medium overall risk.

#### 3.2.2 Economic Feasibility

**Development costs:** Zero. Open-source stack, academic project, existing hardware used throughout.

**Standalone deployment costs (per institution):**

| Item | One-time (INR) | Annual (INR) |
|------|----------------|--------------|
| Server hardware (if needed) | 30,000 | 0 |
| Installation support | 5,000 | 0 |
| Teacher training | 2,000 | 1,000 |
| **Total** | **37,000** | **1,000** |

**Cloud deployment costs (Azure, per institution):**

| Service | Purpose | SKU | Est. cost/month |
|---------|---------|-----|----------------|
| App Service Plan (Linux) | Runs the container | B1 | ~$13 |
| Azure Container Registry | Stores the Docker image | Basic | ~$5 |
| Azure Files (Storage Account) | Persistent data and model file | Standard LRS | ~$2 |
| **Total** | | | **~$20/month (~₹20,000/year)** |

**Cost-benefit comparison (1,000 students, 4 exams/year, 10 short-answer questions each):**

Manual grading cost:
- 40,000 questions × 2 minutes = 1,333 hours
- At ₹300/hour: **₹4,00,000/year**

Automated grading (cloud deployment):
- Azure hosting: ~₹20,000/year
- Teacher spot-check at 10%: ~₹40,000
- **Total: ~₹60,000/year — approximately 85% cost reduction**

Automated grading (standalone, after first year):
- ₹1,000/year maintenance + 10% spot-check: ~₹41,000/year — approximately **90% cost reduction**

Payback period on hardware investment (standalone): under one month.

**Conclusion:** Economically feasible under both deployment models, with substantial savings in either case.

#### 3.2.3 Social Feasibility

| Stakeholder | Primary concern | How the system addresses it |
|-------------|----------------|----------------------------|
| Teachers | Job displacement, loss of grading control | System is positioned as an assistant; grades are visible to teachers before any are communicated to students |
| Students | Impersonal evaluation, perceived unfairness | Feature transparency in the per-question view shows precisely what the model measured |
| Administrators | Implementation risk, training overhead | Standalone deployment requires no specialized infrastructure; cloud deployment requires only a browser to manage |
| Parents | Trust in automated systems | Teacher oversight is built into the workflow at every stage |

The peer-aware approach is also more resistant to gaming than pure TF-IDF, which rewards keyword repetition. Question-word demotion specifically reduces the benefit of paraphrasing the question back as an answer.

**Conclusion:** Socially feasible, provided the system is clearly communicated as a teacher-facing tool rather than a fully autonomous grader.

### 3.3 System Specification

#### 3.3.1 Hardware Requirements

**Minimum (standalone deployment):**

| Component | Specification |
|-----------|--------------|
| Processor | Intel Core i3 / AMD Ryzen 3, 4 cores, 2.0 GHz |
| Memory | 4 GB RAM |
| Storage | 20 GB |
| Network | Not required after initial setup |

**Recommended (standalone):**

| Component | Specification |
|-----------|--------------|
| Processor | Intel Core i5 / AMD Ryzen 5, 6 cores |
| Memory | 8 GB RAM |
| Storage | 50 GB SSD |

**Cloud deployment (Azure App Service B1):**

| Component | Specification |
|-----------|--------------|
| vCPU | 1 |
| Memory | 1.75 GB RAM |
| Storage | Azure Files mount (persistent, Standard LRS) |
| OS | Linux (Ubuntu) |
| HTTPS | Managed by App Service |

No GPU required for any deployment mode.

**Client (web browser):**

| Component | Specification |
|-----------|--------------|
| Browser | Chrome 90+, Firefox 88+, Edge 90+, Safari 14+ |
| Display | 1024 × 768 minimum |
| Network | Required for examination submission |

#### 3.3.2 Software Stack

**Backend:**

| Layer | Technology | Version |
|-------|------------|---------|
| Runtime | Python | 3.11 |
| Web framework | FastAPI | 0.100+ |
| Server | uvicorn | 0.23+ |
| Package manager | uv | latest |
| Storage | JSON files with atomic writes | — |
| Containerization | Docker | — |

**NLP and ML:**

| Library | Purpose | Version |
|---------|---------|---------|
| scikit-learn | TF-IDF, GBM | 1.3+ |
| NLTK | Stemming, stopwords | 3.8+ |
| NumPy | Vector operations | 1.24+ |
| Pandas | Data manipulation | 2.0+ |
| joblib | Model serialization | 1.3+ |

**Frontend:**

| Component | Technology |
|-----------|-----------|
| Language | HTML5, CSS3, JavaScript (ES6+) |
| Framework | None |
| State management | localStorage |

**Cloud infrastructure:**

| Service | Purpose |
|---------|---------|
| Azure App Service (Linux B1) | Container hosting |
| Azure Container Registry (Basic) | Docker image storage |
| Azure Files (Standard LRS) | Persistent data and model storage |

**Development tools:**

| Tool | Purpose |
|------|---------|
| Git | Version control |
| pytest | Testing |
| Black | Code formatting |

---

## 4. Design Approach and Details

### 4.1 System Architecture

The system is organized into three layers with clean separation of concerns, deployed as a single container on Azure App Service.

**NLP Core** is the grading engine with no web dependencies. It accepts a question, a reference answer, and a list of student responses, and returns a list of (grade, features) pairs. It is fully testable in isolation and executable from the command line independently of the web application.

**API Layer** is the FastAPI application that wraps the NLP core. It handles authentication, request validation, asynchronous job management for batch grading, and all storage operations. It also serves the frontend as static files under `/ui/`, so no separate web server is required.

**Storage Layer** consists of four JSON collections: questions, submissions, grades, and jobs. Each write is atomic (write to `.tmp`, then `os.replace`). The collections are stored on an Azure Files mount at `/app/data/`, ensuring data persists across container restarts and redeployments. The storage interface is designed so MongoDB or PostgreSQL can replace the JSON files later without modifying the API or NLP layers.

**Deployment architecture:**

```
Browser
   │
   ▼
App Service (single container, Linux B1)
   ├── FastAPI (uvicorn, port 8000)
   │     ├── /ui/*  → serves frontend HTML/CSS/JS (StaticFiles)
   │     ├── /api/v1/questions, /submissions, /jobs, etc.
   │     └── BackgroundTasks → grading runs in-process
   │
   └── Azure Files mount (/app/data)
         ├── questions.json
         ├── submissions.json
         ├── grades.json
         ├── jobs.json
         └── grade_mapper.joblib
```

Azure Files is mounted at the App Service level via the built-in storage mount configuration. No code changes are required to move between local file storage and the cloud mount.

**Azure services:**

| Service | SKU | Role |
|---------|-----|------|
| App Service Plan (Linux) | B1 | Hosts the container |
| Azure Container Registry | Basic | Stores the Docker image |
| Azure Files (Storage Account) | Standard LRS | Persistent data and model file |

**Dockerfile:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev
COPY . .
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

The GBM model file (`grade_mapper.joblib`) is loaded once at server startup and shared across all requests. It is stored on the Azure Files mount so it persists independently of container lifecycle.

### 4.2 Design

#### 4.2.1 Data Flow

The grading data flow for a full batch run:

1. Teacher clicks "Grade All Questions"
2. Frontend fetches all questions; for each question, fetches all submissions
3. One API call per question submits the full batch (`POST /api/v1/grade/batch`)
4. The API creates a job record on Azure Files and returns a job ID immediately
5. Background task runs the NLP pipeline across all responses for that question
6. Frontend polls each job endpoint until status is "complete"
7. Results merge across all questions into a complete grade summary
8. Summary caches in `localStorage`; subsequent dashboard views read from cache without additional API calls

#### 4.2.2 Class Diagram

The main application classes:

- `Preprocessor` — text normalization, tokenization, stemming, stopword removal
- `DynamicKeyBuilder` — constructs the peer-aware grading key from a response corpus
- `FeatureExtractor` — computes all six features for a single response against a key
- `GradeMapper` — wraps the GBM model; loads `grade_mapper.joblib` once on server startup
- `GradingEngine` — orchestrates the two-pass pipeline end-to-end
- `QuestionStore`, `SubmissionStore`, `GradeStore`, `JobStore` — one class per JSON collection, backed by Azure Files in the cloud deployment

---

## 5. Methodology and Testing

### 5.1 Module Description

#### 5.1.1 Preprocessing Module

Converts raw text into normalized tokens suitable for vectorization.

**Components:**
- `TextNormalizer` — lowercase conversion, punctuation removal, special character handling
- `Tokenizer` — whitespace and punctuation boundary splitting
- `StopwordRemover` — removes NLTK's standard English stopword list
- `Stemmer` — Porter Stemmer; "photosynthesising" and "photosynthesis" map to the same token
- `QuestionWordDemoter` — identifies terms that appear in the question text

**Input:** Raw text string and question text  
**Output:** Normalized stemmed token list and question token list

**Algorithm:**
```
function preprocess(text, question):
    text = lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, NLTK_english)
    tokens = stem(tokens, PorterStemmer)
    question_tokens = preprocess(question, "")
    return tokens, question_tokens
```

#### 5.1.2 Dynamic Key Builder Module

Constructs the peer-aware grading key from the entire submission batch.

**Algorithm:**
```
function build_dynamic_key(reference, corpus, strictness=20):
    vocabulary = union of all terms across corpus and reference

    // Peer signal: average term presence across responses, scaled down
    corpus_tf = sum(term_frequencies(corpus)) / strictness

    // Start from corpus signal
    key = corpus_tf

    // Reference override: reference terms always carry maximum weight
    for term in preprocess(reference):
        key[term] = 1.0

    // Question-word demotion: reduce weight of terms from the question itself
    for term in question_tokens:
        key[term] = key[term] / 1.8

    return key
```

The `strictness` parameter controls how much the peer signal influences the key. At the default value of 20, a term needs to appear across approximately 20% of the batch before it meaningfully shifts the key away from the reference. The reference override step is the central design decision: reference terms always receive weight 1.0 regardless of their frequency among student responses. Peer terms may earn partial credit; reference terms cannot be diluted.

#### 5.1.3 Feature Extraction Module

Computes six features for each student response.

| Feature | Description |
|---------|-------------|
| `sim_with_demotion` | `dot(response_vec, key) / sqrt(L1(response) × L1(key)) × 5`, key has question-word demotion applied |
| `sim_no_demotion` | Same computation using the key without question-word demotion |
| `length_ratio` | `len(response_tokens) / len(reference_tokens)` — penalizes very short or padded answers |
| `jaccard_unigram` | `|response_stems ∩ reference_stems| / |response_stems ∪ reference_stems|` |
| `keyword_coverage` | `|response_stems ∩ reference_stems| / |reference_stems|` — fraction of reference key terms present |
| `jaccard_bigram` | Bigram overlap on raw lowercase text before stemming — rewards phrase-level agreement |

Two design decisions are worth noting. First, L1 normalization is used for the similarity features rather than the more common L2 (Euclidean). For short texts, L1 normalization preserves absolute term frequency information better; a few high-frequency terms can otherwise dominate a unit-length L2 vector. Second, the bigram feature operates on unstemmed text because stemming destroys phrase-level signal—"natural selection" and "selection natural" are meaningfully different, and stemming would treat both identically.

#### 5.1.4 Grade Mapper Module

Wraps the trained Gradient Boosting Regressor.

**Model configuration:**
- `GradientBoostingRegressor` (scikit-learn)
- 100 estimators, max depth 2, learning rate 0.1
- Subsample 0.8, min samples per leaf 10
- Serialized as `grade_mapper.joblib`; stored on Azure Files
- Loaded once at server startup, shared across all requests via the FastAPI application state

Depth-2 trees capture pairwise feature interactions—high keyword coverage combined with a very short response, or high similarity alongside low bigram overlap—without overfitting to training data patterns. The shallow depth is an intentional design choice.

**Prediction:**
```
function grade(features, max_marks):
    raw_grade = model.predict(features)     // predicts on 0–5 scale
    scaled = raw_grade * (max_marks / 5.0)  // rescale to teacher's mark range
    return clip(scaled, 0, max_marks)
```

The model is trained on the 0–5 Mohler scale. The clip-and-rescale step handles questions with different maximum marks without retraining.

#### 5.1.5 API Module

| Endpoint | Method | Purpose | Auth |
|----------|--------|---------|------|
| `/api/v1/questions` | GET | List all questions | None |
| `/api/v1/questions` | POST | Create question | API key |
| `/api/v1/questions/{id}` | DELETE | Delete question | API key |
| `/api/v1/questions/export` | GET | Export as JSON | API key |
| `/api/v1/questions/import` | POST | Import JSON | API key |
| `/api/v1/responses` | POST | Submit student answer | None |
| `/api/v1/responses/{question_id}` | GET | List responses for question | API key |
| `/api/v1/grade/batch` | POST | Start batch grading job | API key |
| `/api/v1/jobs/{id}` | GET | Poll job status | API key |
| `/api/v1/results/{job_id}` | GET | Retrieve grading results | API key |
| `/api/v1/export/{job_id}` | GET | Export results as CSV | API key |

The `POST /api/v1/responses` endpoint requires no authentication, so the student examination functions without an API key. All teacher-facing operations require the key.

#### 5.1.6 Storage Module

Four JSON collections, each backed by Azure Files in the cloud deployment:

- `questions.json` — question bank
- `submissions.json` — student responses
- `grades.json` — graded results
- `jobs.json` — batch job state

Each write uses an atomic pattern:

```python
with open(path + ".tmp", "w") as f:
    json.dump(data, f)
os.replace(path + ".tmp", path)  # atomic on POSIX and Windows 10+
```

`os.replace` is atomic on all supported platforms, preventing partial writes from corrupting a data file if the container restarts mid-operation. Job state is written to disk at each status transition, so batch grading jobs can be recovered following an unexpected container restart.

### 5.2 Testing

#### 5.2.1 Testing Strategy

The project followed a V-Model approach: unit tests for individual modules, integration tests for the complete grading pipeline, system tests for accuracy and performance, and user acceptance testing for the teacher and student workflows.

#### 5.2.2 Unit Tests

**Preprocessing module:**

| Test ID | Input | Expected output | Status |
|---------|-------|-----------------|--------|
| UT-PP-01 | "Hello World" | "hello world" | Pass |
| UT-PP-02 | "Hello, World!" | "Hello World" | Pass |
| UT-PP-03 | "the cat is on the mat" | ["cat", "mat"] | Pass |
| UT-PP-04 | ["running", "runs", "ran"] | ["run", "run", "ran"] | Pass |
| UT-PP-05 | "" | [] | Pass |
| UT-PP-06 | "café résumé" | "cafe resume" | Pass |

**Feature extraction:**

| Test ID | Description | Expected | Status |
|---------|-------------|----------|--------|
| UT-FE-01 | Identical response and reference | All features near maximum | Pass |
| UT-FE-02 | Empty response | All features = 0 | Pass |
| UT-FE-03 | Response containing only question words | Low sim_with_demotion, higher sim_no_demotion | Pass |
| UT-FE-04 | Response significantly longer than reference | length_ratio > 1.0 | Pass |

**Similarity computation:**

| Test ID | Input | Expected | Status |
|---------|-------|----------|--------|
| UT-SIM-01 | Identical vectors | 1.0 | Pass |
| UT-SIM-02 | Orthogonal vectors | 0.0 | Pass |
| UT-SIM-03 | Zero vector | 0.0 (no NaN) | Pass |
| UT-SIM-04 | 50% term overlap | ~0.5 | Pass |

#### 5.2.3 Integration Tests

| Test ID | Description | Expected | Status |
|---------|-------------|----------|--------|
| IT-GP-01 | Single response through full pipeline | Valid grade in [0, max\_marks] | Pass |
| IT-GP-02 | Batch of 100 responses | All grades computed, no errors | Pass |
| IT-GP-03 | Same response, two different corpora | Grades vary as peer signal changes | Pass |
| IT-GP-04 | API call to grade endpoint | Correct response format and grade | Pass |
| IT-GP-05 | Grade batch then retrieve results | Stored grades match computed grades | Pass |
| IT-GP-06 | Export then re-import question bank | Round-trip produces identical questions | Pass |

#### 5.2.4 System Tests

**Performance (Intel Core i5, 8 GB RAM, no GPU):**

| Test ID | Load | Achieved |
|---------|------|----------|
| ST-PERF-01 | 1 response | < 50 ms |
| ST-PERF-02 | 100 responses | < 3 seconds |
| ST-PERF-03 | 1,000 responses | < 20 seconds |
| ST-PERF-04 | 10,000 responses | < 3 minutes |

All targets from the non-functional requirements were met or exceeded on consumer hardware.

**Accuracy on Mohler dataset:**

| Metric | Target | Achieved |
|--------|--------|----------|
| RMSE | < 0.85 | **0.81** |

Comparison against published baselines on the same benchmark:

| System | RMSE |
|--------|------|
| Bag-of-words baseline | 0.978 |
| TF-IDF + SVR | 1.022 |
| TF-IDF + SVM | 1.150 |
| **This system (GBM + 6 features)** | **0.81** |
| Fine-tuned RoBERTa-Large | 0.70 |

The 0.11 RMSE gap between this system and RoBERTa-Large represents the accuracy cost of not using a GPU. For institutions with GPU compute available, fine-tuning a transformer model remains the higher-accuracy option. For the majority of Indian educational institutions this project targets, the GBM with peer-aware features is the best available option that will run on their hardware or within a $20/month cloud budget.

#### 5.2.5 User Acceptance Testing

| UAT ID | Scenario | User | Result |
|--------|----------|------|--------|
| UAT-01 | Create question with reference answer and max marks | Teacher | Pass |
| UAT-02 | Export question bank as JSON | Teacher | Pass |
| UAT-03 | Import a JSON question paper | Teacher | Pass |
| UAT-04 | Grade all students across all questions in one click | Teacher | Pass |
| UAT-05 | View class distribution and ranked student table | Teacher | Pass |
| UAT-06 | View per-question feature scores for individual responses | Teacher | Pass |
| UAT-07 | Navigate to an individual student result page | Teacher | Pass |
| UAT-08 | Export class results as CSV | Teacher | Pass |
| UAT-09 | Take examination with free navigation between questions | Student | Pass |
| UAT-10 | Answer auto-saves and survives page refresh | Student | Pass |
| UAT-11 | Submit with unanswered-question warning | Student | Pass |

#### 5.2.6 Regression Testing

An automated regression suite runs on every commit:
- All unit tests
- Critical integration tests covering the complete grading pipeline
- API smoke tests for all endpoints
- Accuracy benchmark on a fixed sample from the Mohler dataset

---

## 6. References

[1] Mohler, M., and Mihalcea, R. (2009). "Text-to-text semantic similarity for automatic short answer grading." *Proceedings of the 12th Conference of the European Chapter of the ACL*, pp. 567–575.

[2] Suzen, N., Gorban, A.N., Levesley, J., and Mirkes, E.M. (2019). "Automatic short answer grading and feedback using text mining methods." *Procedia Computer Science*, 169, pp. 726–743.

[3] Mello, R.F., et al. (2025). "Automatic Short Answer Grading in the LLM Era: Does GPT-4 with Prompt Engineering Beat Traditional Models?" *Proceedings of the ACM LAK Conference*.

[4] Thakkar, M. (2021). "Finetuning Transformer Models to Build ASAG System." *arXiv preprint*.

[5] Sultan, M.A., Salazar, C., and Sumner, T. (2016). "Fast and Easy Short Answer Grading with High Accuracy." *Proceedings of NAACL-HLT*, pp. 1070–1075.

[6] Gaddipati, S.K. (2020). "Comparative Evaluation of Pretrained Transfer Learning Models on Automatic Short Answer Grading." *arXiv preprint*.

[7] Zhang, K., Xu, H., Tang, J., and Li, J. (2006). "Keyword extraction using support vector machine." *Advances in Web-Age Information Management*, pp. 85–96.

[8] Uzun, Y. (2005). "Keyword extraction using naive bayes." *Bilkent University, Dept. of Computer Science*.

[9] Jalilifard, A., Caridá, V.F., Mansano, A.F., Cristo, R.S., and da Fonseca, F.P. (2021). "Semantic Sensitive TF-IDF to Determine Word Relevance in Documents." *Advances in Computational Intelligence*, pp. 327–337.

[10] Burrows, S., Gurevych, I., and Stein, B. (2014). "The eras and trends of automatic short answer grading." *International Journal of Artificial Intelligence in Education*, 25(1), pp. 60–117.

[11] Sung, C., Dhamecha, T.I., Saha, S., Ma, T., Reddy, V., and Arber, R. (2019). "Pre-Training BERT on Domain Resources for Short Answer Grading." *Proceedings of EMNLP-IJCNLP*, pp. 6071–6075.

[12] Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." *Proceedings of ICLR Workshop*.

[13] Pennington, J., Socher, R., and Manning, C.D. (2014). "GloVe: Global Vectors for Word Representation." *Proceedings of EMNLP*, pp. 1532–1543.

[14] Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification." *Proceedings of EMNLP*, pp. 1746–1751.

[15] Dzikovska, M., Nielsen, R., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I., and Dang, H.T. (2013). "SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge." *Proceedings of SemEval*, pp. 263–274.

[16] Reimers, N., and Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of EMNLP-IJCNLP*, pp. 3982–3992.

[17] National Education Policy 2020. Ministry of Education, Government of India.

[18] Albitar, S., Fournier, S., and Espinasse, B. (2014). "An Effective TF/IDF-based Text-to-Text Semantic Similarity Measure for Text Classification." *Proceedings of WISE*, pp. 105–114.

[19] Lan, M., Tan, C.L., Su, J., and Lu, Y. (2022). "Research on Text Similarity Measurement Hybrid Algorithm with Term Semantic Information and TF-IDF Method." *Advances in Multimedia*, 2022.

[20] Yeung, C., et al. (2025). "A Zero-Shot LLM Framework for Automatic Assignment Grading in Higher Education." *Proceedings of AIED 2025*.

---

*Document Version: 2.0*  
*Last Updated: April 2026*  
*Authors: Gokularajan R, Prashitha J R*  
*Institution: VIT University, School of Computer Science and Engineering*
