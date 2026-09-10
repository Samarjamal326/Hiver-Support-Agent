# Decision Log

## Stage 2: Data Pipeline & Sampling Decisions

### 1. Brand Selection: SpotifyCares
- **Quantitative Profile**:
  - **43,265** brand replies (`author_id == 'SpotifyCares'`).
  - **41,697** matched direct parent customer messages.
  - **62.5%** thread openers (26,068 conversations originating at root).
- **Strategic Rationale**:
  - **Avoided Overused Brands**: Bypassed `sprintcare` and `AmazonHelp`, which represent the most generic and over-represented domains in customer support benchmarks.
  - **Why over AppleSupport**: `SpotifyCares` conversations feature richer conversational nuance and less strictly templated responses compared to `AppleSupport`'s ubiquitous redirect macros.
  - **Domain Alignment with Hiver**: Spotify's consumer-facing SaaS and subscription troubleshooting profile (billing disputes, device synchronizations, playback faults, playlist recovery) maps directly onto the multi-intent SaaS customer support workflows targeted by Hiver.

### 2. Subsampling Method: Thread-Preserving Graph Traversal
- **Methodology**:
  - Rather than applying uniform random row subsampling (which fragments dialogues and produces isolated one-off utterances), we implemented a bidirectional tweet graph traversal:
    - **Backward Traversal**: Trace `in_response_to_tweet_id` iteratively up to `max_thread_traverse_depth=20` to recover all antecedent customer inquiries and conversational context back to the root `thread_id`.
    - **Forward Traversal**: Trace `response_tweet_id` iteratively (handling comma-separated branch responses) up to `max_thread_traverse_depth=20` to capture customer follow-up questions, confirmations, and brand resolutions.
- **Outcome & Rationale**:
  - Preserves end-to-end multi-turn dialogic context, which is strictly required for grounded retrieval-augmented generation (RAG) and realistic conversational escalation evaluation in later stages.

### 3. Non-English Filtering: ASCII-Ratio Heuristic
- **Methodology**:
  - Used an ASCII alphabetic ratio heuristic (`ratio of [a-zA-Z] characters to total characters >= 0.6`) implemented using Python stdlib without introducing heavy third-party NLP packages (`langdetect`, `fasttext`, or `pycld2`).
- **Audit & Trade-Offs**:
  - **Benefits**: Zero runtime overhead, 100% deterministic, adds no complex binary dependencies.
  - **Approximation & Limitations**:
    - *False Negatives*: Legitimate English tweets with high densities of emojis, symbols, or numeric IDs can fall below the 0.6 ratio and be dropped.
    - *False Positives*: Short Romance language phrases (e.g. Spanish, French, or Italian phrases lacking diacritics) contain predominantly ASCII letters and will pass the filter.
  - Deemed an acceptable engineering compromise for this stage since English tweets represent the vast majority of SpotifyCares interactions and retention remains above 96%.
