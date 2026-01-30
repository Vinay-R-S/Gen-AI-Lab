# Prompt Engineering Patterns & Personalization

1. Persona Prompting
2. Cognitive Verifier Pattern
3. Question Refinement Pattern
4. Provide New Information and Ask Questions
5. Root Prompt

## Prompt to learn LangGraph using AI

1. Persona Prompting
```
You are a senior machine-learning researcher with 10+ years of experience in knowledge graphs and representation learning. I want you to: explain the core ideas behind "Learning Lang graph" (knowledge graph construction from textual sources), list three practical architectures to build such graphs for a moderately-sized research dataset (50k documents), and provide a prioritized implementation checklist (data pipeline, entity extraction, relation schema, deduplication, storage, evaluation). For each architecture, include pros/cons, required compute, and a public baseline or library to start with.
```

2. Cognitive Verifier Pattern
```
You are an ML systems architect. Before answering the main question—"How do I build a robust Learning Lang graph from mixed technical docs and forum posts?"—list the sub-questions you need to verify (e.g., text languages, domain-specific jargon, scale, update frequency, ground-truth availability). For each sub-question, say how you would verify the answer (tool, test, or data sample) and how the answer changes your architecture choices. After listing/verifying, produce a high-level plan.
```

3. Question Refinement Pattern
```
Original prompt: "How to make a knowledge graph from text?"
Task: Suggest a better version of this question to get a more accurate and actionable response for someone building a production pipeline. Provide the improved prompt, explain why each change improves the result, and list three clarifying details the model should ask about next.
```

4. Provide New Information and Ask Questions
```
Paste the following text (example):
"We have 52,000 technical documents (English), average length 1,200 words. We need nightly updates, labels exist for 6k documents, and entities include Products, APIs, Vulnerabilities. Storage target: Neo4j for fast traversal. Budget: single 16‑core server + 64GB RAM."
Instruction: Based strictly on this information, ask me three questions that test my understanding and three follow-up design constraints you think are missing.
```

5. Root Prompt
```
Primary goal: Build and maintain a high‑quality knowledge graph from mixed technical text to support semantic search and link discovery with near‑real‑time nightly updates and explainability of extracted relations.
Start the session with this root constraint, then propose: (a) an end‑to‑end architecture, (b) a prioritized milestone plan for the next 12 weeks, and (c) an evaluation suite (metrics and tests) to validate correctness and freshness.
```

<details>
<Summary> Combined Prompt </Summary>

```
You are a senior machine-learning researcher with 10+ years of experience in knowledge graphs and representation learning. I want you to: explain the core ideas behind "Learning Lang graph" (knowledge graph construction from textual sources), list three practical architectures to build such graphs for a moderately-sized research dataset (50k documents), and provide a prioritized implementation checklist (data pipeline, entity extraction, relation schema, deduplication, storage, evaluation). For each architecture, include pros/cons, required compute, and a public baseline or library to start with.

You are an ML systems architect. Before answering the main question—"How do I build a robust Learning Lang graph from mixed technical docs and forum posts?"—list the sub-questions you need to verify (e.g., text languages, domain-specific jargon, scale, update frequency, ground-truth availability). For each sub-question, say how you would verify the answer (tool, test, or data sample) and how the answer changes your architecture choices. After listing/verifying, produce a high-level plan.

Original prompt: "How to make a knowledge graph from text?"
Task: Suggest a better version of this question to get a more accurate and actionable response for someone building a production pipeline. Provide the improved prompt, explain why each change improves the result, and list three clarifying details the model should ask about next.

Paste the following text (example):
"We have 52,000 technical documents (English), average length 1,200 words. We need nightly updates, labels exist for 6k documents, and entities include Products, APIs, Vulnerabilities. Storage target: Neo4j for fast traversal. Budget: single 16‑core server + 64GB RAM."
Instruction: Based strictly on this information, ask me three questions that test my understanding and three follow-up design constraints you think are missing.

Primary goal: Build and maintain a high‑quality knowledge graph from mixed technical text to support semantic search and link discovery with near‑real‑time nightly updates and explainability of extracted relations.
Start the session with this root constraint, then propose: (a) an end‑to‑end architecture, (b) a prioritized milestone plan for the next 12 weeks, and (c) an evaluation suite (metrics and tests) to validate correctness and freshness.
```

</details>

## Prompt to learn LangChain using AI

1. Persona Prompting
```
You are an expert developer evangelist for LangChain with deep practical knowledge of LLM toolkits and vector DB integrations. Explain how to design a LangChain-based assistant that: ingests PDFs, uses a vector store for retrieval, and supports multi-step tool execution. Provide sample code snippets (pseudo or TypeScript) for ingestion, chain assembly, and a minimal test harness.
```

2. Cognitive Verifier Pattern
```
You are a senior system designer. Before giving the LangChain implementation plan, enumerate the detailed sub-questions needed to choose the right vector DB, embedding model, and caching strategy (e.g., expected QPS, document size distribution, cost constraints). For each sub-question, provide a small verification step (one-liner test or quick metric) and state which LangChain components would be impacted.
```

3. Question Refinement Pattern
```
Original prompt: "Help me build LangChain app"
Task: Improve this prompt so the LLM returns runnable starter code, a list of dependencies, and a simple test. Provide the improved prompt and explain how it forces the assistant to return concrete artifacts.
```

4. Provide New Information and Ask Questions
```
Paste the following text: "Ingestion progress: 12k PDFs converted to text (avg 4000 tokens), vector store: Pinecone (single project), embeddings: OpenAI ada-002, current 200 MB index size, queries per minute: ~8 peak."
Instruction: Based only on this text, ask me three questions to test whether my LangChain setup will scale and suggest two immediate engineering changes.
```

5. Root Prompt
```
Primary goal: Deliver a production-ready LangChain assistant that handles document ingestion, semantic retrieval, and safe tool execution under a 10 QPS constraint and a $200/month embedding budget. Start by proposing an architecture with cost estimates, failure modes, and monitoring signals.
```

<details>
<Summary> Combined Prompt </Summary>

```
You are an expert developer evangelist for LangChain with deep practical knowledge of LLM toolkits and vector DB integrations. Explain how to design a LangChain-based assistant that: ingests PDFs, uses a vector store for retrieval, and supports multi-step tool execution. Provide sample code snippets (pseudo or TypeScript) for ingestion, chain assembly, and a minimal test harness.

You are a senior system designer. Before giving the LangChain implementation plan, enumerate the detailed sub-questions needed to choose the right vector DB, embedding model, and caching strategy (e.g., expected QPS, document size distribution, cost constraints). For each sub-question, provide a small verification step (one-liner test or quick metric) and state which LangChain components would be impacted.

Original prompt: "Help me build LangChain app"
Task: Improve this prompt so the LLM returns runnable starter code, a list of dependencies, and a simple test. Provide the improved prompt and explain how it forces the assistant to return concrete artifacts.

Paste the following text: "Ingestion progress: 12k PDFs converted to text (avg 4000 tokens), vector store: Pinecone (single project), embeddings: OpenAI ada-002, current 200 MB index size, queries per minute: ~8 peak."
Instruction: Based only on this text, ask me three questions to test whether my LangChain setup will scale and suggest two immediate engineering changes.

Primary goal: Deliver a production-ready LangChain assistant that handles document ingestion, semantic retrieval, and safe tool execution under a 10 QPS constraint and a $200/month embedding budget. Start by proposing an architecture with cost estimates, failure modes, and monitoring signals.
```

</details>

## Prompt to learn MCP using AI

1. Persona Prompting
```
You are a senior game-theory researcher and systems engineer specializing in multi-agent competitive environments. Describe how to design an MCP where autonomous agents compete to optimize a shared resource. Include agent interface definitions, sandboxing strategies, scoring rules, and anti-abuse mechanisms. Provide a sample YAML or JSON spec for an agent submission.
```

2. Cognitive Verifier Pattern
```
You are a security-focused platform architect. Before suggesting MCP rules and architecture, list the verification sub-questions (e.g., determinism of environment, seed control, allowed libraries, runtime limits). For each one, provide a short test (unit or fuzz test idea) that verifies the property. Then produce a minimal safe baseline environment spec.
```

3. Question Refinement Pattern
```
Original prompt: "Build an MCP competition platform"
Task: Create a clearer, testable version of this prompt that asks for competition rules, grading scripts, and a CI pipeline for validating new agent submissions. Explain the improvements and produce the refined prompt.
```

4. Provide New Information and Ask Questions
```
Paste the following: "Competition needs: 200 simultaneous agents, deterministic physics sim, 30s per turn CPU, submissions as Docker images, results logged to S3. Judges require reproducible replay."
Instruction: Based strictly on this data, ask me three questions that check my assumptions and suggest two essential mitigations for cheating or non-reproducibility.
```

5. Root Prompt
```
Primary goal: Create a robust, fair, and reproducible MCP environment for competitive research with safety controls. Begin the session by designing the submission API, runtime sandbox, scoring engine, and a monitoring/forensics plan.
```

<details>
<Summary> Combined Prompt </Summary>

```
You are a senior game-theory researcher and systems engineer specializing in multi-agent competitive environments. Describe how to design an MCP where autonomous agents compete to optimize a shared resource. Include agent interface definitions, sandboxing strategies, scoring rules, and anti-abuse mechanisms. Provide a sample YAML or JSON spec for an agent submission.

You are a security-focused platform architect. Before suggesting MCP rules and architecture, list the verification sub-questions (e.g., determinism of environment, seed control, allowed libraries, runtime limits). For each one, provide a short test (unit or fuzz test idea) that verifies the property. Then produce a minimal safe baseline environment spec.

Original prompt: "Build an MCP competition platform"
Task: Create a clearer, testable version of this prompt that asks for competition rules, grading scripts, and a CI pipeline for validating new agent submissions. Explain the improvements and produce the refined prompt.

Paste the following: "Competition needs: 200 simultaneous agents, deterministic physics sim, 30s per turn CPU, submissions as Docker images, results logged to S3. Judges require reproducible replay."
Instruction: Based strictly on this data, ask me three questions that check my assumptions and suggest two essential mitigations for cheating or non-reproducibility.

Primary goal: Create a robust, fair, and reproducible MCP environment for competitive research with safety controls. Begin the session by designing the submission API, runtime sandbox, scoring engine, and a monitoring/forensics plan.
```

</details>

## Prompt to learn Docker using AI

1. Persona Prompting
```
You are a senior DevOps engineer experienced with container orchestration and secure CI/CD. Explain how to containerize a Python web service (FastAPI) and produce: a well-documented Dockerfile, a multi-stage build for minimal image size, a docker-compose file for local dev (Postgres + Redis + app), and security best practices (secrets, user privileges, image scanning).
```

2. Cognitive Verifier Pattern
```
You are a build reliability engineer. Before producing Docker artifacts, list verification sub-questions (e.g., base image policy, required OS packages, expected memory/cpu limits, runtime user). For each, say how you'd validate via CI (unit test, smoke test, image scan) and how answers change the Dockerfile or orchestration config.
```

3. Question Refinement Pattern
```
Original prompt: "Write a Dockerfile for my app"
Task: Provide an improved prompt that ensures the response includes a secure, multi-stage Dockerfile, docker-compose for local testing, instructions to build and run, and a sample healthcheck. Explain how the improved wording ensures actionable output.
```

4. Provide New Information and Ask Questions
```
Paste: "App: FastAPI, Python 3.10, dependencies in poetry.lock, uses Postgres and Redis, dev requires local hot-reload. CI runner: Ubuntu-latest, target deploy: Kubernetes."
Instruction: Based only on that text, ask me three questions to verify deployment constraints and list two Dockerfile optimizations you would apply.
```

5. Root Prompt
```
Primary goal: Produce production-ready container images and local dev setup that support safe CI/CD and easy troubleshooting. Start by outlining the image build pipeline, tagging scheme, scanning/registry policy, and rollback strategy.
```

<details>
<Summary> Combined Prompt </Summary>

```
You are a senior DevOps engineer experienced with container orchestration and secure CI/CD. Explain how to containerize a Python web service (FastAPI) and produce: a well-documented Dockerfile, a multi-stage build for minimal image size, a docker-compose file for local dev (Postgres + Redis + app), and security best practices (secrets, user privileges, image scanning).

You are a build reliability engineer. Before producing Docker artifacts, list verification sub-questions (e.g., base image policy, required OS packages, expected memory/cpu limits, runtime user). For each, say how you'd validate via CI (unit test, smoke test, image scan) and how answers change the Dockerfile or orchestration config.

Original prompt: "Write a Dockerfile for my app"
Task: Provide an improved prompt that ensures the response includes a secure, multi-stage Dockerfile, docker-compose for local testing, instructions to build and run, and a sample healthcheck. Explain how the improved wording ensures actionable output.

Paste: "App: FastAPI, Python 3.10, dependencies in poetry.lock, uses Postgres and Redis, dev requires local hot-reload. CI runner: Ubuntu-latest, target deploy: Kubernetes."
Instruction: Based only on that text, ask me three questions to verify deployment constraints and list two Dockerfile optimizations you would apply.

Primary goal: Produce production-ready container images and local dev setup that support safe CI/CD and easy troubleshooting. Start by outlining the image build pipeline, tagging scheme, scanning/registry policy, and rollback strategy.
```

</details>

## Prompt to learn Docker using NLP

1. Persona Prompting
```
You are a research NLP engineer with experience deploying transformer models. Explain a practical workflow to fine-tune a transformer for domain-specific text classification, including data preparation, class imbalance handling, training schedule, evaluation metrics, and model deployment tips. Include sample training hyperparameters for a 4‑GPU setup (or describe scaled-down single‑GPU alternatives).
```

2. Cognitive Verifier Pattern
```
You are a validation engineer. Before proposing an NLP pipeline, list the sub-questions to confirm (e.g., dataset size, label noise, average tokens per sample, expected latency), explain how to measure each quickly, and then provide two alternative pipelines based on the answers (fast/cheap vs accurate/expensive).
```

3. Question Refinement Pattern
```
Original prompt: "Help me do NLP"
Task: Suggest a more precise version that will return a reproducible experiment plan: training script, preprocessing steps, and a small eval dataset. Provide the refined prompt and explain which ambiguities were removed.
```

4. Provide New Information and Ask Questions
```
Paste: "Dataset: 52,500 grayscale images across 7 emotion classes (7,500 each). Model: ResNet-50 pretrained on ImageNet; batch size 32; Adam lr=1e-4; splits: 70/15/15. Training environment: TensorFlow 2.15 on RTX 4060 (8GB)."
Instruction: Based strictly on this text, ask me three questions to test my setup and list two immediate optimizations or risks to watch for.
```

5. Root Prompt
```
Primary goal: Build an efficient, reproducible NLP/vision training and deployment pipeline that maximizes accuracy under given compute limits and supports easy experimentation. Start with a prioritized checklist for the first four weeks and include a minimal monitoring plan for mod
```

<details>
<Summary> Combined Prompt </Summary>

```
You are a research NLP engineer with experience deploying transformer models. Explain a practical workflow to fine-tune a transformer for domain-specific text classification, including data preparation, class imbalance handling, training schedule, evaluation metrics, and model deployment tips. Include sample training hyperparameters for a 4‑GPU setup (or describe scaled-down single‑GPU alternatives).

You are a validation engineer. Before proposing an NLP pipeline, list the sub-questions to confirm (e.g., dataset size, label noise, average tokens per sample, expected latency), explain how to measure each quickly, and then provide two alternative pipelines based on the answers (fast/cheap vs accurate/expensive).

Original prompt: "Help me do NLP"
Task: Suggest a more precise version that will return a reproducible experiment plan: training script, preprocessing steps, and a small eval dataset. Provide the refined prompt and explain which ambiguities were removed.

Paste: "Dataset: 52,500 grayscale images across 7 emotion classes (7,500 each). Model: ResNet-50 pretrained on ImageNet; batch size 32; Adam lr=1e-4; splits: 70/15/15. Training environment: TensorFlow 2.15 on RTX 4060 (8GB)."
Instruction: Based strictly on this text, ask me three questions to test my setup and list two immediate optimizations or risks to watch for.

Primary goal: Build an efficient, reproducible NLP/vision training and deployment pipeline that maximizes accuracy under given compute limits and supports easy experimentation. Start with a prioritized checklist for the first four weeks and include a minimal monitoring plan for mod
```

</details>

