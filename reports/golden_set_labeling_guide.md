# Golden Set Human Labeling Guide

## Overview

This guide provides instructions for human annotators creating the ground-truth benchmark for Hiver's customer support agent. Annotators will label customer messages in `reports/golden_set_candidates.csv` across three core dimensions:
1. **Primary Intent** (`human_intent`)
2. **Annotation Difficulty** (`human_difficulty`)
3. **Escalation Decision** (`human_escalate`)

### Note on Candidate Sampling Heuristic
Candidate messages were sampled from historical customer inquiries. To ensure representative coverage across diverse customer issues rather than skewing heavily toward the most frequent topics, an **approximate regex/keyword heuristic (`rough_intent_bucket`)** was used internally solely to balance sampling proportions. 
This heuristic was used exclusively to ensure minority categories are represented in the candidate pool; it is **never** shown to annotators, and is **never** treated as an accurate classifier. All ground-truth labels are determined strictly by the human labeler.

---

## Allowed Field Values

### 1. `human_intent` (Mandatory)
Every message must be assigned exactly one of the **10 canonical taxonomy labels** or marked as **`N/A - not a genuine support request`**.

| Label | Scope & Description | Typical Triggers & Examples |
|---|---|---|
| **`Account Access/Login`** | Inability to log in, password resets, verification code failures, third-party authentication errors (e.g. Facebook login), or compromised/hacked accounts. | *"Can't log into my account, says password invalid"*, *"Someone changed my email address without permission"* |
| **`Billing & Subscription`** | Charges, renewal questions, refunds, student/promotional discount verification, failed payments, payment method changes, or receipt requests. | *"Charged $9.99 twice this month"*, *"Why isn't my student discount applying at checkout?"* |
| **`Advertisement Complaints`** | Audio/video ads playing on paid Premium subscriptions, excessive/intrusive advertisements, or broken ad-skip functionality. | *"Paying for premium yet audio ads keep interrupting my playlist"*, *"Halloween ads are way too loud"* |
| **`Playback & App Bugs`** | Audio buffering, unexpected pauses, queue failures, songs skipping unexpectedly, missing offline downloads, or playback crashes. | *"Songs pause every 30 seconds"*, *"My downloaded playlist suddenly disappeared"* |
| **`Platform-Specific Technical`** | Device-specific incompatibilities, OS version issues (iOS, Android, Windows, macOS), smart speaker integration (Alexa, Google Home), CarPlay, desktop client lag, or web player bugs. | *"Desktop app freezes on Windows 10 startup"*, *"CarPlay won't sync recent tracks"* |
| **`Content & Feature Requests`** | Requests to add specific songs, albums, or artists to the catalog, feature suggestions (lyrics, UI folder organization), or restoring discontinued features. | *"Please put Taylor Swift's new album on Spotify"*, *"Can you bring back synced lyrics?"* |
| **`Family Plan Management`** | Adding or removing members from a Family plan, address/location verification disputes, member invitations, or switching between individual and family accounts. | *"My brother got kicked from the family plan"*, *"How do I invite another person to our family plan?"* |
| **`Regional/Country Restriction`** | Content unavailability due to regional licensing, country mismatch errors while traveling abroad, or account location configuration. | *"Account locked because country doesn't match location"*, *"Can people in Russia access this track?"* |
| **`General Complaint/Frustration`** | Expressions of dissatisfaction, venting about service or support experience, or churn threats without an actionable bug or technical issue. | *"Your service has become completely unusable lately"*, *"Customer support was useless, ditching Spotify"* |
| **`Praise/Off-topic`** | Compliments, gratitude, social banter, casual acknowledgments, or conversational confirmations (e.g. "DM sent"). | *"Thanks so much, that fixed it!"*, *"Love the new UI update, you guys rock"*, *"DM sent"* |
| **`N/A - not a genuine support request`** | Non-support content swept into the dataset (e.g. brand promotional broadcasts, spam, gibberish, or external marketing tweets). | *"3 months of Premium is just $0.99! Check it out here: http://..."* |

---

### 2. `human_difficulty` (Mandatory)
Indicates how challenging the inquiry is to categorize and understand.

- **`easy`**: The customer's request is explicit, unambiguous, and maps directly onto a single clear category.
- **`ambiguous`**: The message touches on multiple overlapping issues, contains vague or conflicting phrasing, or could legitimately fit more than one intent.
- **`needs_context`**: The message cannot be reliably classified in isolation without reading the antecedent conversational history in `thread_context` (frequent in brief follow-up replies like *"Yes, done that"* or *"Still not working"*).

---

### 3. `human_escalate` (Mandatory)
Indicates whether the inquiry requires routing to a human specialist rather than autonomous AI resolution.

- **`yes`**: Must be escalated to a human agent if:
  - **Account Compromise & Security**: Any indication that an account was hacked, unauthorized credentials were changed, or sensitive data is at risk.
  - **Severe Billing Disputes**: Unauthorized repeated charges, demands for immediate monetary refunds, or disputed bank transactions.
  - **High Frustration & Churn**: Intense anger, abusive language, or explicit threats to cancel/ditch the service where empathetic human de-escalation is necessary.
  - **AI Confidence Boundary**: Any inquiry where the problem description is too complex, contradictory, or high-stakes for an automated system to safely resolve without human judgment.
- **`no`**: Can be handled autonomously:
  - Routine informational inquiries, known troubleshooting workflows (e.g. cache clearing, password reset self-serve links), catalog availability explanations, or polite praise/banter.

---

### 4. `notes` (Optional)
Annotator comments explaining unusual edge cases, multi-intent overlaps, or reasoning behind an escalation decision.

---

## Worked Examples

### Example 1: Clear Opener (Billing Dispute)
- **Position**: `opener`
- **Thread Context**: *(blank)*
- **Message Text**: *"Hi @SpotifyCares I got charged $9.99 yesterday and another $9.99 today for my subscription. Can someone please refund the extra charge?"*
- **Annotation**:
  - `human_intent`: `Billing & Subscription`
  - `human_difficulty`: `easy`
  - `human_escalate`: `yes`
  - `notes`: *"Clear double-charge billing inquiry; requires human agent with billing refund authorization."*

### Example 2: Follow-up Requiring Context
- **Position**: `follow_up`
- **Thread Context**:
  ```text
  Customer: My app keeps freezing on startup on my Galaxy S21.
  Agent: Hey there! Could you try clearing your app cache in Android Settings > Apps > Spotify?
  ```
- **Message Text**: *"Did that, but it still crashes immediately when opening playlists."*
- **Annotation**:
  - `human_intent`: `Playback & App Bugs`
  - `human_difficulty`: `needs_context`
  - `human_escalate`: `no`
  - `notes`: *"Message text alone lacks subject, but thread context clarifies this is an app crash bug on Android following troubleshooting."*

### Example 3: Security & Compromise (Urgent Escalation)
- **Position**: `opener`
- **Thread Context**: *(blank)*
- **Message Text**: *"Someone from another country logged into my account, changed the email, and deleted all my playlists! I'm furious. Fix this now or I'm calling my credit card company!"*
- **Annotation**:
  - `human_intent`: `Account Access/Login`
  - `human_difficulty`: `ambiguous`
  - `human_escalate`: `yes`
  - `notes`: *"Covers account takeover, deleted playlists, and high frustration. Intent categorized under root cause (Account Access/Login). Must escalate immediately due to security risk and churn threat."*

### Example 4: Non-Support Promotional Broadcast
- **Position**: `opener`
- **Thread Context**: *(blank)*
- **Message Text**: *"Holiday savings are here! Get 3 months of Spotify Premium for $0.99. Offer ends Dec 31."*
- **Annotation**:
  - `human_intent`: `N/A - not a genuine support request`
  - `human_difficulty`: `easy`
  - `human_escalate`: `no`
  - `notes`: *"Marketing announcement tweet swept into thread opener dataset, not a customer inquiry."*
