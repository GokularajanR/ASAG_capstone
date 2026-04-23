# Automated Short Answer Grading: A Peer-Aware Approach for Resource-Constrained Educational Environments

**Project Report**

**Authors:**
- Gokularajan R
- Prashitha J R

**Institution:** VIT University, School of Computer Science and Engineering

**Date:** February 2nd 2026

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

The rapid proliferation of digital learning platforms and the increasing scale of educational assessments have created an urgent need for automated evaluation systems capable of grading open-ended student responses. While multiple-choice questions lend themselves to straightforward automated scoring, short-answer questions pose significant challenges due to their linguistic variability and the nuanced reasoning they require. Traditional Automated Short Answer Grading (ASAG) systems evaluate each student response independently against a static, predefined answer key—an approach that fails to capture valid alternative phrasings, common misconceptions, and collective patterns across the student cohort.

This project presents a novel, lightweight ASAG system that employs a peer-aware, two-pass grading methodology. In the first pass, the system constructs a dynamic grading key by analysing the entire corpus of student responses, weighted by a configurable "strictness" parameter. In the second pass, individual responses are evaluated against this adaptive key using TF-IDF vectorization and cosine similarity, with the resulting similarity scores mapped to grades through polynomial regression. This approach more closely mirrors human grading behaviour, where evaluators naturally calibrate their expectations based on the overall quality and patterns observed across submissions.

Evaluated on the benchmark Mohler dataset comprising 2,264 student responses, the proposed system achieves a Root Mean Square Error (RMSE) of 0.81—outperforming all traditional machine learning baselines including TF-IDF with Support Vector Machines (RMSE 1.15), TF-IDF with Support Vector Regression (RMSE 1.022), and bag-of-words approaches (RMSE 0.978). Notably, this performance approaches that of computationally expensive transformer-based models such as fine-tuned RoBERTa-Large (RMSE 0.70), while requiring only seconds to train on standard hardware without GPU acceleration.

The system is designed as a complete end-to-end application deployable on cloud infrastructure such as Azure Databricks, with particular emphasis on serving resource-constrained educational institutions in India. With over 1.5 million schools in India—many lacking access to high-performance computing resources—this lightweight, open-source solution offers a practical pathway to automated assessment that can operate on minimal hardware while maintaining reasonable accuracy. The architecture supports future extensibility through optional integration of open-source sentence embedding models from HuggingFace, enabling institutions to upgrade their grading capabilities as resources permit.

**Keywords:** Automated Short Answer Grading, ASAG, TF-IDF, Peer-Aware Grading, Dynamic Key Construction, Educational Technology, Natural Language Processing, Machine Learning

---

## 1. Introduction

### 1.1 Background

The landscape of education has undergone a fundamental transformation over the past decade. The proliferation of Massive Open Online Courses (MOOCs), Learning Management Systems (LMS), and digital assessment platforms has enabled educational content to reach unprecedented numbers of learners. Platforms such as NPTEL, SWAYAM, Coursera, and edX serve millions of students annually, while school-level initiatives like DIKSHA and various state education portals have brought digital learning to primary and secondary education across India.

This digital transformation, while democratizing access to education, has simultaneously created a critical bottleneck: the evaluation of student learning. While objective assessments—multiple-choice questions, true/false items, and matching exercises—can be graded instantaneously and at infinite scale, they represent only a fraction of meaningful educational assessment. Short-answer questions, which require students to articulate understanding in their own words, demonstrate reasoning, and synthesize knowledge, remain essential for evaluating higher-order thinking skills as defined in Bloom's Taxonomy.

The challenge of grading short-answer responses at scale is not merely logistical but pedagogical. When a single MOOC course may have 100,000 enrolled students, and a typical examination contains 10-20 short-answer questions, the manual grading burden becomes insurmountable. Even in traditional classroom settings, a teacher with 40 students per section and 5 sections faces the prospect of evaluating 200 response sheets per examination—a task that consumes countless hours and inevitably suffers from grader fatigue and inconsistency.

Automated Short Answer Grading (ASAG) systems have emerged as a potential solution to this challenge. These systems leverage Natural Language Processing (NLP) and Machine Learning (ML) techniques to evaluate student responses against reference answers or learned patterns. The field has evolved considerably since its inception, progressing from simple keyword-matching approaches through statistical methods to contemporary deep learning architectures.

However, the evolution of ASAG has largely followed a trajectory that prioritizes accuracy improvements through increasingly complex models—transformer architectures with hundreds of millions of parameters, requiring GPU clusters for training and inference. While such approaches achieve impressive benchmark performance, they remain impractical for the vast majority of educational institutions worldwide, particularly in developing nations where computational resources are scarce and internet connectivity may be unreliable.

### 1.2 Motivations

The motivation for this project emerges from a critical examination of the gap between state-of-the-art ASAG research and the practical needs of educational institutions, particularly in the Indian context.

**The Indian Educational Landscape**

India's education system is among the largest in the world, encompassing:
- Approximately 1.5 million schools serving over 250 million students
- Over 40,000 higher education institutions
- A student-to-teacher ratio that often exceeds 30:1 in government schools
- Significant infrastructure disparities between urban and rural institutions

The National Education Policy (NEP) 2020 emphasizes competency-based assessment and continuous evaluation—approaches that inherently require frequent assessment of open-ended responses. Yet the infrastructure to support such assessment at scale remains woefully inadequate. Most government and aided schools operate with:
- Limited or no access to high-performance computing hardware
- Intermittent internet connectivity in rural areas
- Minimal technical support staff
- Constrained budgets that preclude cloud computing costs

**The Research-Practice Gap**

Contemporary ASAG research, while academically rigorous, has largely failed to address these practical constraints. The dominant paradigm involves:
- Pre-trained transformer models (BERT, RoBERTa, GPT) with 100M+ parameters
- Fine-tuning procedures requiring GPU acceleration
- Inference costs that scale linearly with response volume
- Dependency on cloud APIs with per-request pricing

For a rural school in Tamil Nadu or a government college in Bihar, deploying such systems is simply not feasible. The result is a widening gap between institutions that can afford sophisticated assessment technology and those that cannot—exacerbating existing educational inequities.

**The Case for Lightweight, Peer-Aware Grading**

This project is motivated by the conviction that meaningful automated assessment need not require massive computational resources. By returning to fundamental NLP techniques—TF-IDF vectorization, cosine similarity—and enhancing them with a novel peer-aware methodology, we demonstrate that competitive accuracy can be achieved with minimal computational overhead.

The peer-aware approach is further motivated by observations of human grading behaviour. Experienced educators do not grade each response in isolation; rather, they develop calibrated expectations based on the collective performance of the cohort. A response that might receive a mediocre grade in a high-performing class may be considered above average in a struggling one. Our dynamic key construction mechanism formalizes this intuition, allowing the system to adapt its evaluation criteria based on the actual distribution of student responses.

### 1.3 Scope of the Project

This project encompasses the design, implementation, and evaluation of a complete end-to-end ASAG system with the following scope:

**Core Algorithm Development**
- Implementation of the two-pass peer-aware grading methodology
- TF-IDF vectorization with dynamic vocabulary construction
- Configurable strictness parameter for key adaptation
- Polynomial regression for similarity-to-score mapping
- Question word demotion to prevent gaming

**Application Development**
- Web-based user interface for teachers and administrators
- RESTful API for integration with existing LMS platforms
- Batch processing capability for examination-scale grading
- Real-time single-response grading for formative assessment
- Export functionality for grade reports and analytics

**Deployment Architecture**
- Cloud deployment on Azure Databricks for scalable processing
- Containerized deployment option for on-premises installation
- Lightweight standalone mode for offline operation
- Database integration for response and grade persistence

**Extensibility Framework**
- Plugin architecture for alternative similarity measures
- Optional integration with HuggingFace sentence transformers
- Configurable preprocessing pipelines
- Support for multiple languages (initial focus on English)

**Evaluation and Validation**
- Benchmark evaluation on the Mohler dataset
- Cross-validation methodology ensuring question-level separation
- Comparative analysis with existing ASAG approaches
- Performance profiling for computational efficiency

**Exclusions**
The following are explicitly outside the scope of this project:
- Handwriting recognition (assumes digitized text input)
- Plagiarism detection
- Essay-length response grading
- Real-time proctoring or examination security
- Student authentication and identity verification

---

## 2. Project Description and Goals

### 2.1 Literature Review

The field of Automated Short Answer Grading has evolved significantly over the past two decades, progressing through several distinct paradigms. This section provides a comprehensive review of relevant literature, organized by methodological approach.

#### 2.1.1 Survey Papers and Foundational Work

Burrows, Gurevych, and Stein (2014) provided a seminal survey titled "The Eras and Trends of Automatic Short Answer Grading," reviewing over 80 papers and establishing a unified taxonomy for the field. They identified five historical eras: concept mapping (1966-1999), information extraction (2000-2006), corpus-based methods (2006-2009), machine learning (2009-2013), and the emerging era of evaluation challenges.

More recently, comprehensive surveys have documented the shift toward deep learning approaches. A 2022 survey on arXiv examined the transition "from Word Embeddings to Transformers," analyzing how advances in NLP have influenced ASAG system design. The survey categorized approaches into three groups: word embedding methods, sequential models (LSTM, GRU), and attention-based architectures (BERT, RoBERTa).

A systematic review by researchers in 2018 analyzed 44 papers employing machine learning for ASAG, concluding that statistical models remain commonly used, vector-based similarity measures are most popular, and standardized datasets for evaluation are lacking.

#### 2.1.2 TF-IDF and Traditional Text Similarity Methods

Term Frequency-Inverse Document Frequency (TF-IDF) remains a foundational technique in ASAG due to its interpretability and computational efficiency. Mohler and Mihalcea (2009) explored unsupervised techniques comparing knowledge-based and corpus-based text similarity measures, introducing a technique to improve performance by integrating automatic feedback from student answers—an early precursor to peer-aware approaches.

Albitar et al. (2014) proposed an improved TF-IDF function for text-to-text similarity, demonstrating high accuracy when applied to short answer grading and significant improvements over classical cosine similarity on the Microsoft paraphrase corpus.

Lan et al. (2022) addressed TF-IDF's limitation of ignoring semantic information by proposing a hybrid method combining TF-IDF with semantic representations. Their work acknowledged that pure TF-IDF cannot accurately reflect similarity between texts with synonymous expressions.

Comparative studies have consistently shown that while TF-IDF-based methods achieve moderate performance (Pearson correlation ~0.55), they are significantly outperformed by contextual embedding approaches (Pearson correlation ~0.85). However, this performance gap must be weighed against the substantial computational overhead of embedding-based methods.

#### 2.1.3 Semantic Similarity Measures

Semantic similarity measurement has been extensively studied in the NLP community. A comprehensive review of Short-Text Semantic Similarity (STSS) techniques published in MDPI Applied Sciences (2023) found that most recent studies rely on pre-trained transformer models for text vectorization.

The Universal Sentence Encoder, available through TensorFlow Hub, has been applied to automated grading with promising results. Systems combining multiple similarity measures—edit similarity, cosine similarity, Jaccard similarity, normalized word count, and semantic similarity—have shown improved performance through weighted combination of metrics.

Word Mover's Distance (WMD), which measures the minimum distance word embeddings need to "travel" to transform one document to another, has been shown to outperform cosine similarity in certain grading applications. Research indicates WMD achieved error rate reductions of 1.3% when combined with Multinomial Naive Bayes classification.

#### 2.1.4 Word Embeddings

Word embeddings represented a paradigm shift in NLP, enabling dense vector representations that capture semantic relationships. Word2Vec (Mikolov et al., 2013) introduced two architectures—Continuous Bag of Words (CBOW) and Skip-gram—that learn embeddings from large corpora. GloVe (Pennington, Socher, and Manning, 2014) complemented this with matrix factorization on word-context co-occurrence matrices.

Comparative evaluation of pretrained transfer learning models for ASAG (Sasikiran et al., 2020) compared ELMo, BERT, GPT, and GPT-2 embeddings with cosine similarity on the Mohler dataset. Surprisingly, ELMo outperformed the other three models on RMSE scores and correlation measurements, suggesting that more complex models do not always yield better results for this task.

#### 2.1.5 Transformer-Based Models

The introduction of BERT (Bidirectional Encoder Representations from Transformers) and its variants has dominated recent ASAG research. Sung et al. (2019) demonstrated that fine-tuning BERT for short answer grading achieved up to 10% absolute improvement in macro-average F1 compared to state-of-the-art methods, with additional gains through MNLI transfer learning.

Sentence-BERT (SBERT) has emerged as a particularly effective approach for ASAG, enabling efficient sentence embeddings that can be computed once and reused. Studies have shown that pre-trained SBERT models can be fine-tuned with minimal reference answers for effective grading.

Thakkar (2021) achieved an RMSE of 0.70 on the Mohler dataset by fine-tuning RoBERTa-Large—currently the best reported performance on this benchmark. However, this approach requires significant computational resources for both training and inference.

#### 2.1.6 Large Language Models

The emergence of Large Language Models (LLMs) has opened new possibilities for ASAG. Research published in 2025 at the ACM LAK Conference examined whether GPT-4 with prompt engineering could beat traditional models, finding that pre-trained GPT-4 performance was comparable to hand-engineered models without requiring any training data.

Studies on LLM-based grading in medical education (BMC Medical Education, 2024) compared GPT-4 and Gemini across 2,288 student answers in multiple languages. Results showed moderate agreement with human grades, with GPT-4 producing lower grades but fewer false positives.

Research published in Nature Scientific Reports (2025) found that GPT-4 shows comparable performance to human examiners in ranking tasks, with no consistent evidence of bias toward AI-generated or lengthy answers. The highest correlation achieved was 0.98 with human graders, suggesting near-human performance in certain contexts.

Zero-shot frameworks (Yeung et al., 2025) have demonstrated that LLM-based grading can be achieved without any training or fine-tuning, providing personalized feedback that significantly improves student motivation and understanding.

#### 2.1.7 Neural Network Architectures

Various neural network architectures have been applied to ASAG beyond transformers. Bidirectional LSTM with attention mechanisms and convolutional layers (AC-BiLSTM) has shown competitive performance across multiple datasets.

Siamese neural networks, which learn to compare pairs of inputs, have been adapted for ASAG. The Siamese Manhattan LSTM approach uses word embedding vectors to create matrices fed to LSTM networks and similarity functions, producing grades based on semantic similarity between student and reference answers.

Convolutional Neural Networks for sentence classification (Kim, 2014) demonstrated that shallow CNNs with pre-trained word vectors sometimes outperform deeper models, suggesting that architectural complexity does not always correlate with performance.

#### 2.1.8 Automated Essay Scoring

While distinct from short answer grading, Automated Essay Scoring (AES) research provides relevant insights. Comprehensive surveys (Artificial Intelligence Review, 2024) have documented the evolution from statistical ML with manual features to neural network approaches.

The ASAP (Automated Student Assessment Prize) dataset, comprising 17,450 essays from 7th-10th grade students, has become the de facto benchmark for essay scoring, with 90% of essay grading systems using this dataset. Recent methods achieve 79.3% Quadratic Weighted Kappa (QWK), indicating substantial agreement with human graders.

Hybrid approaches combining deep learning embeddings with handcrafted linguistic features (grammar errors, readability, sentence length) have shown improved accuracy over pure neural methods, suggesting that domain knowledge remains valuable even in the deep learning era.

#### 2.1.9 Benchmark Datasets

Several benchmark datasets have emerged for ASAG evaluation:

**Mohler Dataset (2009, 2011):** 79 questions, 2,273 student answers from a data structures course, graded 0-5 by two educators. This dataset provides individual and average scores, enabling inter-rater reliability analysis. Baseline performance: Pearson 0.464, RMSE 0.978.

**SemEval-2013 Task 7:** Comprises Beetle (3,941 responses to 56 questions on electricity/electronics) and SciEntsBank (~10,000 responses to 197 questions across 15 scientific domains). Provides 5-way, 3-way, and 2-way label schemes with unseen answer, unseen question, and unseen domain test scenarios.

**ASAP Dataset (2012):** ~17,450 essays across 8 prompts from 7th-10th grade students. Evaluation uses Quadratic Weighted Kappa (QWK).

### 2.2 Gaps Identified

Analysis of the existing literature reveals several critical gaps that this project aims to address:

**Gap 1: Computational Accessibility**

The dominant trend in ASAG research prioritizes benchmark performance through increasingly complex models. Fine-tuned transformer models (BERT, RoBERTa) achieve state-of-the-art results but require:
- GPU hardware for training (often multiple GPUs for reasonable training times)
- Significant memory (16GB+ for large models)
- Cloud computing budgets for deployment at scale
- Technical expertise for model fine-tuning and maintenance

This creates a de facto barrier excluding resource-constrained institutions from adopting modern ASAG technology. No existing work has explicitly optimized for the intersection of competitive accuracy and minimal computational requirements.

**Gap 2: Static Key Limitation**

Nearly all existing approaches—from traditional TF-IDF to transformer-based methods—compare each student response independently against a static reference answer. This approach:
- Fails to recognize valid alternative phrasings not present in the key
- Cannot adapt to cohort-specific vocabulary or expression patterns
- Treats outliers and typical responses identically
- Does not leverage collective patterns that human graders naturally observe

While some work has explored clustering approaches (Suzen et al., 2019) to group similar responses, no prior work has proposed a systematic method for constructing a dynamic grading key that adapts based on the corpus of student responses.

**Gap 3: Indian Educational Context**

Despite India's position as a major developing market for educational technology, existing ASAG research has largely ignored the specific constraints and requirements of Indian educational institutions:
- Intermittent internet connectivity in rural areas
- Limited IT infrastructure in government schools
- Budget constraints precluding cloud computing costs
- Need for offline operation capability
- Teacher training and technical support limitations

No existing ASAG system has been designed with explicit consideration for deployment in such environments.

**Gap 4: Interpretability and Transparency**

While explainability has received attention in recent research (RATAS, LLM-Rubric), most high-performing ASAG systems operate as black boxes. Teachers cannot understand why a particular grade was assigned, limiting their ability to:
- Verify grading decisions
- Provide meaningful feedback to students
- Identify potential system errors
- Build trust in automated assessment

The peer-aware approach proposed in this project offers inherent interpretability—grades are based on measurable similarity to a dynamic key constructed from the actual response corpus, with the strictness parameter providing explicit control over evaluation criteria.

**Gap 5: End-to-End Application**

Academic ASAG research typically focuses on the core grading algorithm, presenting results on benchmark datasets without addressing practical deployment concerns:
- User interface for teachers
- Integration with existing LMS platforms
- Batch processing for examination-scale grading
- Database management for response persistence
- Deployment and scaling considerations

This project addresses this gap by developing a complete end-to-end application suitable for real-world educational use.

### 2.3 Objectives

The objectives of this project are organized into primary and secondary categories:

**Primary Objectives**

1. **Develop a peer-aware ASAG algorithm** that constructs dynamic grading keys from student response corpora, achieving competitive accuracy (RMSE < 0.85 on Mohler dataset) while maintaining minimal computational requirements.

2. **Implement the algorithm as a production-ready application** with web-based user interface, RESTful API, batch processing capability, and database integration.

3. **Design for resource-constrained deployment** ensuring the system operates effectively on standard hardware (4GB RAM, no GPU) with optional cloud scaling for larger institutions.

4. **Validate the approach** through rigorous evaluation on benchmark datasets with appropriate cross-validation methodology.

**Secondary Objectives**

5. **Create deployment templates** for Azure Databricks enabling institutions with cloud access to leverage distributed processing for large-scale grading.

6. **Develop integration mechanisms** allowing the system to interface with common LMS platforms (Moodle, Google Classroom) via standard APIs.

7. **Implement extensibility framework** supporting optional integration of HuggingFace sentence transformers for institutions desiring enhanced accuracy at higher computational cost.

8. **Document best practices** for deploying ASAG systems in Indian educational institutions, including guidance for teacher training and system administration.

### 2.4 Problem Statement

**Formal Problem Statement:**

Given a set of student responses $R = \{r_1, r_2, ..., r_n\}$ to a short-answer question $Q$ with reference answer $A$, develop an automated grading function $G: R \rightarrow [0, 5]$ that:

1. Assigns grades consistent with human evaluation (minimizing RMSE against human-graded scores)
2. Adapts evaluation criteria based on the collective characteristics of $R$ (peer-awareness)
3. Executes in $O(n \cdot m)$ time complexity where $m$ is average response length (computational efficiency)
4. Operates with $O(V)$ space complexity where $V$ is vocabulary size (memory efficiency)
5. Requires no GPU acceleration or specialized hardware (accessibility)

**Contextual Problem Statement:**

Educational institutions across India, particularly government schools and colleges in rural areas, lack access to automated assessment tools that could alleviate teacher workload and enable competency-based continuous evaluation as mandated by NEP 2020. Existing ASAG solutions are either:
- Too computationally expensive for available infrastructure
- Too inaccurate to be practically useful
- Too complex to deploy and maintain without specialized technical staff

This project addresses this problem by developing an ASAG system that achieves the optimal trade-off between accuracy, computational efficiency, and deployment simplicity—making automated short answer grading accessible to institutions that have been excluded from this technology.

### 2.5 Project Plan

The project is organized into six phases spanning 16 weeks:

**Phase 1: Research and Requirements (Weeks 1-2)**
- Comprehensive literature review
- Requirements gathering from educational stakeholders
- Dataset acquisition and preprocessing
- Development environment setup

**Phase 2: Core Algorithm Development (Weeks 3-5)**
- TF-IDF vectorization implementation
- Dynamic key construction mechanism
- Similarity calculation module
- Polynomial regression training
- Unit testing of core components

**Phase 3: Application Development (Weeks 6-9)**
- Database schema design and implementation
- RESTful API development
- Web-based user interface
- Batch processing pipeline
- Integration testing

**Phase 4: Cloud Deployment (Weeks 10-12)**
- Azure Databricks workspace setup
- Spark-based distributed processing implementation
- Container orchestration configuration
- Performance optimization
- Load testing

**Phase 5: Evaluation and Validation (Weeks 13-14)**
- Benchmark evaluation on Mohler dataset
- Cross-validation with question-level separation
- Comparative analysis with baseline methods
- Computational efficiency profiling
- User acceptance testing

**Phase 6: Documentation and Deployment (Weeks 15-16)**
- Technical documentation
- User manuals and training materials
- Deployment guides for various environments
- Final report preparation
- Project presentation

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

**FR-01: Question and Answer Management**
- FR-01.1: The system shall allow teachers to input questions with corresponding reference answers.
- FR-01.2: The system shall support bulk import of questions via CSV/Excel upload.
- FR-01.3: The system shall maintain a question bank organized by subject and topic.
- FR-01.4: The system shall allow editing and versioning of questions and reference answers.

**FR-02: Student Response Processing**
- FR-02.1: The system shall accept individual student responses through web interface.
- FR-02.2: The system shall support batch upload of responses via CSV/Excel files.
- FR-02.3: The system shall preprocess responses (lowercase, punctuation removal, stopword removal, stemming).
- FR-02.4: The system shall handle responses in English language.

**FR-03: Grading Functionality**
- FR-03.1: The system shall construct dynamic grading keys using the peer-aware two-pass methodology.
- FR-03.2: The system shall compute TF-IDF vectors for all responses in a grading batch.
- FR-03.3: The system shall calculate cosine similarity between responses and dynamic keys.
- FR-03.4: The system shall map similarity scores to grades using trained polynomial regression.
- FR-03.5: The system shall assign grades on a configurable scale (default 0-5).
- FR-03.6: The system shall support configurable strictness parameter (default 20).

**FR-04: Results and Reporting**
- FR-04.1: The system shall display individual grades with similarity scores.
- FR-04.2: The system shall generate class-level grade distribution reports.
- FR-04.3: The system shall export grades to CSV/Excel/PDF formats.
- FR-04.4: The system shall provide grade analytics (mean, median, standard deviation).

**FR-05: User Management**
- FR-05.1: The system shall support user authentication with role-based access.
- FR-05.2: The system shall define three roles: Administrator, Teacher, and Student.
- FR-05.3: The system shall log user actions for audit purposes.

**FR-06: API Integration**
- FR-06.1: The system shall expose RESTful API endpoints for all grading functions.
- FR-06.2: The system shall support API authentication via JWT tokens.
- FR-06.3: The system shall provide webhooks for LMS integration.

**FR-07: Model Management**
- FR-07.1: The system shall allow training of polynomial regression models on custom datasets.
- FR-07.2: The system shall persist trained models for reuse.
- FR-07.3: The system shall support model versioning and rollback.

#### 3.1.2 Non-Functional Requirements

**NFR-01: Performance**
- NFR-01.1: The system shall grade a single response within 100 milliseconds.
- NFR-01.2: The system shall grade a batch of 1000 responses within 30 seconds.
- NFR-01.3: The system shall support concurrent grading of up to 100 batches on cloud deployment.
- NFR-01.4: The system shall initialize and load models within 5 seconds of startup.

**NFR-02: Scalability**
- NFR-02.1: The system shall scale horizontally on Azure Databricks for large workloads.
- NFR-02.2: The system shall support up to 1 million responses per grading session on cloud deployment.
- NFR-02.3: The standalone deployment shall support up to 10,000 responses per session.

**NFR-03: Reliability**
- NFR-03.1: The system shall achieve 99.5% uptime for cloud deployment.
- NFR-03.2: The system shall implement automatic retry for transient failures.
- NFR-03.3: The system shall checkpoint batch processing progress for recovery.

**NFR-04: Usability**
- NFR-04.1: The system shall provide responsive web interface supporting desktop and tablet devices.
- NFR-04.2: The system shall complete common tasks (upload, grade, export) within 3 clicks.
- NFR-04.3: The system shall display clear error messages with remediation guidance.
- NFR-04.4: The system shall support English interface language.

**NFR-05: Security**
- NFR-05.1: The system shall encrypt all data in transit using TLS 1.3.
- NFR-05.2: The system shall encrypt sensitive data at rest using AES-256.
- NFR-05.3: The system shall implement rate limiting to prevent abuse.
- NFR-05.4: The system shall sanitize all user inputs to prevent injection attacks.

**NFR-06: Maintainability**
- NFR-06.1: The system shall follow modular architecture with clear separation of concerns.
- NFR-06.2: The system shall achieve minimum 80% code coverage in unit tests.
- NFR-06.3: The system shall use semantic versioning for releases.

**NFR-07: Portability**
- NFR-07.1: The system shall deploy on Windows, Linux, and macOS for standalone mode.
- NFR-07.2: The system shall containerize using Docker for consistent deployment.
- NFR-07.3: The system shall operate without internet connectivity in offline mode.

**NFR-08: Accuracy**
- NFR-08.1: The system shall achieve RMSE < 0.85 on the Mohler benchmark dataset.
- NFR-08.2: The system shall achieve Pearson correlation > 0.75 with human grades.

### 3.2 Feasibility Study

#### 3.2.1 Technical Feasibility

**Algorithm Feasibility**

The proposed peer-aware TF-IDF approach builds upon well-established NLP techniques with proven implementations:

| Component | Technology | Maturity | Risk Level |
|-----------|------------|----------|------------|
| Text Preprocessing | NLTK, tm (R) | Mature (15+ years) | Low |
| TF-IDF Vectorization | scikit-learn, text2vec | Mature (10+ years) | Low |
| Cosine Similarity | NumPy, base R | Mature (20+ years) | Low |
| Polynomial Regression | scikit-learn, lm (R) | Mature (30+ years) | Low |
| Dynamic Key Construction | Custom implementation | Novel | Medium |

The only novel component—dynamic key construction—is algorithmically straightforward, involving weighted aggregation of term frequencies. Preliminary implementation in R has demonstrated feasibility with RMSE of 0.81.

**Infrastructure Feasibility**

The system targets two deployment modes with distinct infrastructure requirements:

**Standalone Mode:**
- Commodity hardware available at any educational institution
- No specialized dependencies beyond Python/R runtime
- Offline operation capability ensures functionality without internet

**Cloud Mode (Azure Databricks):**
- Azure provides academic pricing and credits for educational institutions
- Databricks offers managed Spark clusters with auto-scaling
- Pay-per-use model eliminates upfront infrastructure investment

**Integration Feasibility**

Standard protocols ensure compatibility with existing educational technology:
- REST API compatibility with LMS platforms (Moodle REST API, Google Classroom API)
- CSV/Excel import/export for manual workflows
- JWT authentication compatible with institutional identity providers

**Technical Risk Assessment:**

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Algorithm underperforms on new datasets | Medium | High | Cross-validation, parameter tuning |
| Scalability bottlenecks | Low | Medium | Azure auto-scaling, performance testing |
| Integration complexity | Medium | Low | Standard APIs, documentation |

**Conclusion:** The project is technically feasible with low to medium risk.

#### 3.2.2 Economic Feasibility

**Development Costs**

| Item | Cost (INR) | Notes |
|------|------------|-------|
| Development Hardware | 0 | Existing institutional resources |
| Cloud Development Credits | 0 | Azure for Students program |
| Software Licenses | 0 | Open-source stack (Python, R, PostgreSQL) |
| Developer Time | 0 | Academic project |
| **Total Development Cost** | **0** | |

**Deployment Costs (Per Institution)**

**Standalone Deployment:**
| Item | One-Time (INR) | Annual (INR) |
|------|----------------|--------------|
| Server Hardware | 30,000 | 0 |
| Operating System | 0 | 0 (Linux) |
| Installation Support | 5,000 | 0 |
| Training | 2,000 | 1,000 |
| **Total** | **37,000** | **1,000** |

**Cloud Deployment (Small Institution - 1000 students):**
| Item | Annual (INR) |
|------|--------------|
| Azure Databricks (pay-per-use) | 12,000 |
| Storage (10 GB) | 1,200 |
| Network Transfer | 600 |
| **Total** | **13,800** |

**Cost-Benefit Analysis**

Manual grading costs for comparison (1000 students, 4 exams/year, 10 short-answer questions each):
- Grading time: 40,000 questions × 2 minutes = 1,333 hours
- Teacher cost: 1,333 hours × ₹300/hour = ₹4,00,000

Automated grading with proposed system:
- Annual cost: ₹13,800 (cloud) or ₹1,000 (standalone after first year)
- Teacher review time (10% spot-check): ₹40,000
- **Total: ₹53,800** (cloud) or **₹41,000** (standalone)

**Savings: ₹3,47,000 - ₹3,59,000 per year (87-90% reduction)**

**Return on Investment:**
- Standalone: Payback period < 1 month
- Cloud: Payback period < 1 month

**Conclusion:** The project is economically feasible with significant cost savings.

#### 3.2.3 Social Feasibility

**Stakeholder Analysis**

| Stakeholder | Interest | Concerns | Mitigation |
|-------------|----------|----------|------------|
| Teachers | Reduced workload, more time for teaching | Job displacement, loss of control | Position as assistant not replacement, teacher oversight |
| Students | Faster feedback, consistent grading | Impersonal evaluation, accuracy | Transparency in grading criteria, appeal process |
| Administrators | Cost savings, efficiency | Implementation risk, training | Phased rollout, comprehensive training |
| Parents | Quality education | Trust in automated systems | Communication about benefits and safeguards |

**Social Impact Assessment**

**Positive Impacts:**
1. **Educational Equity:** Enables schools with limited resources to offer sophisticated assessment
2. **Teacher Empowerment:** Frees teachers to focus on pedagogy rather than clerical grading
3. **Timely Feedback:** Students receive grades faster, enabling rapid learning iteration
4. **Consistency:** Eliminates grader fatigue and inter-rater variability

**Potential Negative Impacts:**
1. **Depersonalization:** Risk of students feeling evaluated by machines
2. **Over-reliance:** Teachers may lose grading skills if fully automated
3. **Gaming:** Students may learn to exploit system weaknesses

**Mitigation Strategies:**
1. Design system as teacher assistant, not replacement
2. Require teacher review and approval of grades
3. Implement adversarial testing and continuous improvement
4. Maintain transparency about system capabilities and limitations

**Acceptance Factors for Indian Schools:**

| Factor | Status | Notes |
|--------|--------|-------|
| Language Support | English initially | Hindi, regional languages in future |
| Cultural Appropriateness | Neutral | Domain-agnostic approach |
| Infrastructure Compatibility | High | Designed for low-resource environments |
| Teacher Digital Literacy | Variable | Comprehensive training provided |
| Institutional Readiness | Variable | Phased adoption recommended |

**Conclusion:** The project is socially feasible with appropriate change management.

### 3.3 System Specification

#### 3.3.1 Hardware Specification

**Minimum Requirements (Standalone Deployment)**

| Component | Specification | Notes |
|-----------|--------------|-------|
| Processor | Intel Core i3 / AMD Ryzen 3 (4 cores) | 2.0 GHz minimum |
| Memory | 4 GB RAM | 8 GB recommended |
| Storage | 20 GB HDD | SSD preferred for performance |
| Display | 1366 × 768 resolution | For administrative interface |
| Network | Optional | Required only for updates |

**Recommended Requirements (Standalone Deployment)**

| Component | Specification |
|-----------|--------------|
| Processor | Intel Core i5 / AMD Ryzen 5 (6 cores) |
| Memory | 8 GB RAM |
| Storage | 50 GB SSD |
| Display | 1920 × 1080 resolution |
| Network | Broadband internet |

**Cloud Deployment (Azure Databricks)**

| Component | Specification | Scaling |
|-----------|--------------|---------|
| Cluster Type | Standard_DS3_v2 | Auto-scale 2-10 nodes |
| Driver Memory | 14 GB | Fixed |
| Worker Memory | 14 GB per node | Scales with cluster |
| Storage | Azure Blob Storage | Pay-per-use |
| Network | Azure Virtual Network | Managed |

**Client Requirements (Web Interface)**

| Component | Specification |
|-----------|--------------|
| Browser | Chrome 90+, Firefox 88+, Edge 90+, Safari 14+ |
| Display | 1024 × 768 minimum |
| Network | 1 Mbps minimum |

#### 3.3.2 Software Specification

**Server-Side Stack**

| Layer | Technology | Version | License |
|-------|------------|---------|---------|
| Operating System | Ubuntu Server | 22.04 LTS | Open Source |
| Runtime | Python | 3.10+ | PSF License |
| Web Framework | FastAPI | 0.100+ | MIT |
| Task Queue | Celery | 5.3+ | BSD |
| Message Broker | Redis | 7.0+ | BSD |
| Database | PostgreSQL | 15+ | PostgreSQL License |
| ORM | SQLAlchemy | 2.0+ | MIT |

**NLP and ML Libraries**

| Library | Purpose | Version |
|---------|---------|---------|
| scikit-learn | ML algorithms, TF-IDF | 1.3+ |
| NLTK | Text preprocessing | 3.8+ |
| NumPy | Numerical operations | 1.24+ |
| Pandas | Data manipulation | 2.0+ |
| sentence-transformers | Optional embeddings | 2.2+ |

**Cloud-Specific (Azure Databricks)**

| Component | Technology |
|-----------|------------|
| Compute | Apache Spark 3.4 |
| Notebook | Databricks Notebooks |
| Storage | Azure Blob Storage |
| Orchestration | Azure Data Factory |
| Monitoring | Azure Monitor |

**Client-Side Stack**

| Layer | Technology | Version |
|-------|------------|---------|
| Framework | React | 18+ |
| State Management | Redux Toolkit | 1.9+ |
| UI Components | Material-UI | 5.14+ |
| HTTP Client | Axios | 1.4+ |
| Build Tool | Vite | 4.4+ |

**Development and DevOps**

| Tool | Purpose |
|------|---------|
| Git | Version control |
| GitHub Actions | CI/CD |
| Docker | Containerization |
| pytest | Testing |
| Black, isort | Code formatting |
| mypy | Type checking |

---

## 4. Design Approach and Details

### 4.1 System Architecture

The system follows a layered architecture with clear separation of concerns, designed to support both standalone and cloud deployment modes.




### 4.2 Design

#### 4.2.1 Data Flow Diagram

![DFD](DFD.jpg)

#### 4.2.2 Class Diagram

![Class Diagram](Class.png)

## 5. Methodology and Testing

### 5.1 Module Description

The system is organized into the following functional modules:

#### 5.1.1 Preprocessing Module

**Purpose:** Transform raw text responses into normalized form suitable for vectorization.

**Components:**
- **TextNormalizer:** Converts text to lowercase, removes punctuation, handles special characters
- **Tokenizer:** Splits text into word tokens using whitespace and punctuation boundaries
- **StopwordRemover:** Eliminates common English stopwords using NLTK's standard list
- **Stemmer:** Reduces words to root form using Porter Stemming algorithm
- **QuestionWordDemoter:** Identifies and reduces weight of words appearing in question

**Input:** Raw text string
**Output:** List of normalized, stemmed tokens

**Algorithm:**
```
function preprocess(text, question):
    text = lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    question_words = preprocess(question, "")
    tokens = demote_question_words(tokens, question_words)
    return tokens
```

#### 5.1.2 Vectorization Module

**Purpose:** Convert preprocessed text into numerical vectors using TF-IDF weighting.

**Components:**
- **VocabularyBuilder:** Constructs vocabulary from corpus of all responses
- **TermFrequencyCalculator:** Computes TF for each term in each document
- **InverseDocumentFrequencyCalculator:** Computes IDF for each term across corpus
- **TFIDFTransformer:** Combines TF and IDF into final vectors

**Input:** List of preprocessed token lists
**Output:** Sparse matrix of TF-IDF vectors

**Algorithm:**
```
function compute_tfidf(corpus):
    vocabulary = build_vocabulary(corpus)
    N = len(corpus)
    tfidf_matrix = empty_matrix(len(corpus), len(vocabulary))

    for doc_idx, document in enumerate(corpus):
        for term in vocabulary:
            tf = count(term, document) / len(document)
            df = count_documents_containing(term, corpus)
            idf = log(N / df)
            tfidf_matrix[doc_idx, term_idx] = tf * idf

    return tfidf_matrix, vocabulary
```

#### 5.1.3 Dynamic Key Builder Module

**Purpose:** Construct adaptive grading key incorporating information from student responses.

**Components:**
- **ReferenceKeyInitializer:** Creates initial key from reference answer
- **CorpusAnalyzer:** Extracts term frequency statistics from all responses
- **KeyAdapter:** Blends reference key with corpus statistics using strictness parameter
- **QuestionWordDemoter:** Reduces influence of question-repeated terms

**Input:** Reference answer, corpus of responses, strictness parameter
**Output:** Dynamic key vector

**Algorithm:**
```
function build_dynamic_key(reference, corpus, strictness=20):
    vocabulary = get_vocabulary(corpus)

    // Initialize key from reference
    key = binary_vector(reference, vocabulary)

    // Compute corpus term frequencies
    corpus_tf = sum(term_frequencies(corpus)) / strictness

    // Blend reference with corpus
    key = key + corpus_tf

    // Demote question words
    question_words = extract_question_words()
    key[question_words] = key[question_words] / 1.8

    return key
```

#### 5.1.4 Similarity Calculator Module

**Purpose:** Compute similarity between response vectors and dynamic key.

**Components:**
- **CosineSimilarity:** Implements cosine similarity metric
- **VectorNormalizer:** L2 normalizes vectors for similarity computation
- **BatchProcessor:** Efficiently computes similarities for multiple responses

**Input:** Response vector, key vector
**Output:** Similarity score in [0, 1]

**Algorithm:**
```
function cosine_similarity(response_vec, key_vec):
    // Add TF-IDF to binary presence
    response_vec = binary_vector(response) + tfidf_vector(response)

    dot_product = sum(response_vec * key_vec)
    norm_response = sqrt(sum(response_vec))
    norm_key = sqrt(sum(key_vec))

    similarity = dot_product / (norm_response * norm_key)

    // Handle edge cases
    if is_nan(similarity):
        similarity = 0

    return similarity
```

#### 5.1.5 Grade Mapper Module

**Purpose:** Map similarity scores to grades using trained polynomial regression.

**Components:**
- **PolynomialFeatureGenerator:** Creates polynomial features from similarity scores
- **RegressionModel:** Polynomial regression model (degree 7)
- **GradeClipper:** Ensures grades fall within valid range [0, max_score]

**Input:** Similarity score, trained model
**Output:** Predicted grade

**Algorithm:**
```
function map_to_grade(similarity, model):
    // Scale similarity to [0, 5] range
    scaled_similarity = similarity * 5

    // Generate polynomial features
    features = [scaled_similarity^i for i in 1..7]

    // Predict using trained model
    grade = model.predict(features)

    // Clip to valid range
    grade = clip(grade, 0, max_score)

    return grade
```

#### 5.1.6 API Module

**Purpose:** Expose grading functionality through RESTful endpoints.

**Endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/questions` | POST | Create new question with reference answer |
| `/api/v1/questions/{id}` | GET | Retrieve question details |
| `/api/v1/responses` | POST | Submit student response |
| `/api/v1/grade/single` | POST | Grade individual response |
| `/api/v1/grade/batch` | POST | Grade batch of responses |
| `/api/v1/jobs/{id}` | GET | Check batch grading job status |
| `/api/v1/results/{job_id}` | GET | Retrieve grading results |
| `/api/v1/export/{job_id}` | GET | Export results to CSV/Excel |

#### 5.1.7 Optional: HuggingFace Embedding Module

**Purpose:** Provide alternative similarity computation using pre-trained sentence transformers.

**Components:**
- **ModelLoader:** Loads HuggingFace sentence-transformers models
- **EmbeddingGenerator:** Generates dense embeddings for text
- **SemanticSimilarity:** Computes embedding-based similarity

**Supported Models:**
- `all-MiniLM-L6-v2` (384 dimensions, fast)
- `all-mpnet-base-v2` (768 dimensions, accurate)
- `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, multilingual)

**Integration:**
```
function grade_with_embeddings(response, reference, model_name):
    model = load_model(model_name)

    response_emb = model.encode(response)
    reference_emb = model.encode(reference)

    similarity = cosine_similarity(response_emb, reference_emb)
    grade = map_to_grade(similarity)

    return grade
```

### 5.2 Testing

#### 5.2.1 Testing Strategy

The testing strategy follows the V-Model approach with testing activities corresponding to each development phase:

```
Requirements ────────────────────────────────────── Acceptance Testing
     │                                                      │
     ▼                                                      ▼
High-Level Design ────────────────────────────── System Testing
     │                                                      │
     ▼                                                      ▼
Low-Level Design ─────────────────────────────── Integration Testing
     │                                                      │
     ▼                                                      ▼
Implementation ───────────────────────────────── Unit Testing
```

#### 5.2.2 Unit Testing

**Coverage Target:** 80% code coverage

**Test Cases for Preprocessing Module:**

| Test ID | Test Case | Input | Expected Output | Status |
|---------|-----------|-------|-----------------|--------|
| UT-PP-01 | Lowercase conversion | "Hello World" | "hello world" | Pass |
| UT-PP-02 | Punctuation removal | "Hello, World!" | "Hello World" | Pass |
| UT-PP-03 | Stopword removal | "the cat is on the mat" | ["cat", "mat"] | Pass |
| UT-PP-04 | Stemming | ["running", "runs", "ran"] | ["run", "run", "ran"] | Pass |
| UT-PP-05 | Empty input handling | "" | [] | Pass |
| UT-PP-06 | Special characters | "café résumé" | "cafe resume" | Pass |

**Test Cases for Vectorization Module:**

| Test ID | Test Case | Input | Expected Output | Status |
|---------|-----------|-------|-----------------|--------|
| UT-VEC-01 | Single document TF | ["word word other"] | TF(word)=0.67, TF(other)=0.33 | Pass |
| UT-VEC-02 | IDF calculation | Corpus with varying freq | Higher IDF for rare terms | Pass |
| UT-VEC-03 | Sparse matrix format | Large vocabulary | CSR matrix with <5% density | Pass |
| UT-VEC-04 | Vocabulary size | 100 documents | Reasonable vocabulary size | Pass |

**Test Cases for Similarity Module:**

| Test ID | Test Case | Input | Expected Output | Status |
|---------|-----------|-------|-----------------|--------|
| UT-SIM-01 | Identical vectors | v1 = v2 | similarity = 1.0 | Pass |
| UT-SIM-02 | Orthogonal vectors | v1 ⊥ v2 | similarity = 0.0 | Pass |
| UT-SIM-03 | Zero vector handling | v1 = [0,0,0] | similarity = 0.0 (no NaN) | Pass |
| UT-SIM-04 | Partial overlap | 50% common terms | similarity ≈ 0.5 | Pass |

#### 5.2.3 Integration Testing

**Test Cases for Grading Pipeline:**

| Test ID | Test Case | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| IT-GP-01 | End-to-end single grade | Process one response through entire pipeline | Valid grade in [0,5] |
| IT-GP-02 | Batch processing | Process 100 responses | All grades computed, no errors |
| IT-GP-03 | Dynamic key adaptation | Grade same response with different corpora | Grades vary appropriately |
| IT-GP-04 | API to engine integration | REST call to grade endpoint | Correct response format |
| IT-GP-05 | Database persistence | Grade batch and retrieve | Stored grades match computed |

#### 5.2.4 System Testing

**Performance Test Cases:**

| Test ID | Scenario | Load | Expected Performance |
|---------|----------|------|----------------------|
| ST-PERF-01 | Single response | 1 request | < 100ms response time |
| ST-PERF-02 | Small batch | 100 responses | < 5 seconds total |
| ST-PERF-03 | Large batch | 1000 responses | < 30 seconds total |
| ST-PERF-04 | Concurrent batches | 10 × 100 responses | < 60 seconds total |
| ST-PERF-05 | Memory usage | 10000 responses | < 2GB peak memory |

**Accuracy Test Cases:**

| Test ID | Dataset | Metric | Target | Achieved |
|---------|---------|--------|--------|----------|
| ST-ACC-01 | Mohler (full) | RMSE | < 0.85 | 0.81 |
| ST-ACC-02 | Mohler (full) | Pearson | > 0.75 | TBD |
| ST-ACC-03 | Mohler (unseen Q) | RMSE | < 1.0 | TBD |
| ST-ACC-04 | Cross-validation | Mean RMSE | < 0.90 | TBD |

#### 5.2.5 User Acceptance Testing

**UAT Scenarios:**

| UAT ID | Scenario | User | Success Criteria |
|--------|----------|------|------------------|
| UAT-01 | Teacher uploads questions | Teacher | Questions stored correctly |
| UAT-02 | Teacher grades batch | Teacher | Grades displayed, exportable |
| UAT-03 | Teacher reviews grades | Teacher | Can modify grades, save |
| UAT-04 | Student views grade | Student | Grade displayed with similarity |
| UAT-05 | Admin manages users | Admin | CRUD operations work |
| UAT-06 | LMS integration | Teacher | Grades sync to Moodle |

#### 5.2.6 Regression Testing

Automated regression suite runs on every commit:
- All unit tests
- Critical integration tests
- Smoke tests for API endpoints
- Accuracy benchmark on sample dataset

---

## 6. References

[1] Mohler, M., and Mihalcea, R. (2009). "Text-to-text semantic similarity for automatic short answer grading." *Proceedings of the 12th Conference of the European Chapter of the ACL*, pp. 567-575.

[2] Suzen, N., Gorban, A.N., Levesley, J., and Mirkes, E.M. (2019). "Automatic short answer grading and feedback using text mining methods." *Procedia Computer Science*, 169, pp. 726-743.

[3] Mello, R.F., et al. (2025). "Automatic Short Answer Grading in the LLM Era: Does GPT-4 with Prompt Engineering Beat Traditional Models?" *Proceedings of the ACM LAK Conference*.

[4] Thakkar, M. (2021). "Finetuning Transformer Models to Build ASAG System." *arXiv preprint*.

[5] Sultan, M.A., Salazar, C., and Sumner, T. (2016). "Fast and Easy Short Answer Grading with High Accuracy." *Proceedings of NAACL-HLT*, pp. 1070-1075.

[6] Gaddipati, S.K. (2020). "Comparative Evaluation of Pretrained Transfer Learning Models on Automatic Short Answer Grading." *arXiv preprint*.

[7] Zhang, K., Xu, H., Tang, J., and Li, J. (2006). "Keyword extraction using support vector machine." *Advances in Web-Age Information Management*, pp. 85-96.

[8] Uzun, Y. (2005). "Keyword extraction using naive bayes." *Bilkent University, Dept. of Computer Science*.

[9] Jalilifard, A., Caridá, V.F., Mansano, A.F., Cristo, R.S., and da Fonseca, F.P. (2021). "Semantic Sensitive TF-IDF to Determine Word Relevance in Documents." *Advances in Computational Intelligence*, pp. 327-337.

[10] Burrows, S., Gurevych, I., and Stein, B. (2014). "The eras and trends of automatic short answer grading." *International Journal of Artificial Intelligence in Education*, 25(1), pp. 60-117.

[11] Sung, C., Dhamecha, T.I., Saha, S., Ma, T., Reddy, V., and Arber, R. (2019). "Pre-Training BERT on Domain Resources for Short Answer Grading." *Proceedings of EMNLP-IJCNLP*, pp. 6071-6075.

[12] Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). "Efficient Estimation of Word Representations in Vector Space." *Proceedings of ICLR Workshop*.

[13] Pennington, J., Socher, R., and Manning, C.D. (2014). "GloVe: Global Vectors for Word Representation." *Proceedings of EMNLP*, pp. 1532-1543.

[14] Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification." *Proceedings of EMNLP*, pp. 1746-1751.

[15] Dzikovska, M., Nielsen, R., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I., and Dang, H.T. (2013). "SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge." *Proceedings of SemEval*, pp. 263-274.

[16] Reimers, N., and Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of EMNLP-IJCNLP*, pp. 3982-3992.

[17] National Education Policy 2020. Ministry of Education, Government of India.

[18] Albitar, S., Fournier, S., and Espinasse, B. (2014). "An Effective TF/IDF-based Text-to-Text Semantic Similarity Measure for Text Classification." *Proceedings of WISE*, pp. 105-114.

[19] Lan, M., Tan, C.L., Su, J., and Lu, Y. (2022). "Research on Text Similarity Measurement Hybrid Algorithm with Term Semantic Information and TF-IDF Method." *Advances in Multimedia*, 2022.

[20] Yeung, C., et al. (2025). "A Zero-Shot LLM Framework for Automatic Assignment Grading in Higher Education." *Proceedings of AIED 2025*.

---

*Document Version: 1.0*
*Last Updated: February 2026*
*Authors: Gokularajan R, Yuva Yashvin S, Atharva Chinchane*
*Institution: VIT University, School of Computer Science and Engineering*
