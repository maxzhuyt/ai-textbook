# AI Historiography: Data Collection Methodology
*Iteration 2 — Concrete schemas, APIs, and actionable steps | February 3, 2026*

---

## 1. Executive Summary

After exploring OpenAlex, NSF APIs, and bibliometric schemas, here's what I found:

**Good news:**
- **OpenAlex** provides free, comprehensive bibliometric data with ~3.5M+ works classified under "Artificial Intelligence" subfield
- Funder acknowledgments are tracked (DARPA: 59K works, NSF: 1.5M works)
- Co-authorship, citations, and institutional affiliations are all accessible
- Historical coverage back to 1950s (though quality degrades)

**Challenges:**
- Retroactive classification creates noise (pre-1956 "AI" papers are misclassified)
- DARPA funding data sparse before 1990 (acknowledgments weren't systematic)
- No single definition of "AI field" — OpenAlex uses ML-based topic assignment

---

## 2. OpenAlex: Primary Data Source

### 2.1 Schema Overview

**Entities available:**
| Entity | Count | Relevant Fields |
|--------|-------|-----------------|
| Works | 240M+ | title, abstract, authors, citations, topics, funders, publication_year |
| Authors | 90M+ | name, affiliations, works_count, cited_by_count, orcid |
| Institutions | 100K+ | name, type, country, works_count |
| Topics | 4,500 | display_name, keywords, subfield, field, domain |
| Funders | 30K+ | name, works_count (DARPA, NSF included) |

### 2.2 Defining "AI" via OpenAlex

**Option A: Subfield-based (recommended for starting)**
```
filter=primary_topic.subfield.id:1702
```
- Subfield 1702 = "Artificial Intelligence" under Computer Science
- ~3.5M works total, growing ~200K/year

**Option B: Topic-based (more granular)**
Core AI topics I identified:
- T13904: Artificial Intelligence Applications (15K works)
- T10906: AI-based Problem Solving and Planning (50K works)
- T10456: Multi-Agent Systems (54K works)
- T10181: Natural Language Processing (large)
- T10320: Neural Networks and Applications
- T11975: Evolutionary Algorithms
- T12026: Explainable AI (39K works)
- T11574: AI in Games (44K works)

**Option C: Keyword + venue hybrid**
Search for "artificial intelligence" OR "machine learning" OR "neural network" etc., filtered to CS venues.

### 2.3 Historical Coverage (1950-1990)

| Decade | Works (Subfield 1702) | Notes |
|--------|----------------------|-------|
| 1950s | ~12K | Pre-field formation; includes cybernetics |
| 1960s | ~33K | Post-Dartmouth growth |
| 1970s | ~65K | First AI winter (1974-80) not visible in counts |
| 1980s | ~117K | Expert systems boom (1986-1990 spike) |

**Key foundational works I verified:**
- John McCarthy "Recursive functions..." (1960) ✓
- John Holland "Adaptive Systems" (1962) ✓
- Fogel/Owens/Walsh "AI through Simulated Evolution" (1966) ✓
- W. Ross Ashby "Introduction to Cybernetics" (1956) ✓
- Chomsky "Three models for description of language" (1956) ✓

### 2.4 API Access

**Base URL:** `https://api.openalex.org/`

**Sample queries:**

```bash
# Get AI works by year with counts
curl "https://api.openalex.org/works?filter=primary_topic.subfield.id:1702&group_by=publication_year"

# Get highly-cited AI works from 1980s
curl "https://api.openalex.org/works?filter=primary_topic.subfield.id:1702,publication_year:1980-1989&sort=cited_by_count:desc&per_page=50"

# Get DARPA-funded AI works
curl "https://api.openalex.org/works?filter=funders.id:F4320332180,primary_topic.subfield.id:1702&per_page=50"

# Get co-authors for a specific author
curl "https://api.openalex.org/authors/A5085687241" # John McCarthy
```

**Rate limits:** 100K requests/day with free API key

**Bulk data:** Monthly snapshots available (recommended for large-scale analysis)

---

## 3. Funding Data

### 3.1 DARPA via OpenAlex

**Funder ID:** F4320332180 (Defense Advanced Research Projects Agency)
**Total works:** 59,199
**AI-classified works:** ~7,000

**Historical pattern I found:**
| Period | DARPA AI Works | Context |
|--------|----------------|---------|
| 1974-1979 | ~10 | Pre-Strategic Computing |
| 1980-1987 | ~67 | Early Strategic Computing |
| 1988-1993 | ~308 | Peak Strategic Computing |
| 1994-1999 | ~231 | Post-SC decline |
| 2000-2009 | ~1,500 | Recovery |
| 2010-2019 | ~2,600 | Modern growth |
| 2020-2025 | ~2,833 | AI boom |

**Limitation:** Funding acknowledgments in papers weren't standardized before 1990s. This underestimates historical DARPA influence.

### 3.2 NSF Awards API

**Base URL:** `https://api.nsf.gov/services/v1/awards.json`

**Sample query:**
```bash
curl "https://api.nsf.gov/services/v1/awards.json?keyword=artificial+intelligence&printFields=id,title,startDate,expDate,estimatedTotalAmt,piFirstName,piLastName,awardeeName,fundProgramName"
```

**Available fields:**
- Award ID, title, abstract
- PI name, institution
- Start/end dates
- Funding amount
- Program name (useful for tracking AI-specific programs)

**Limitations:**
- Keyword search has false positives ("artificial pinning centers" matches "artificial")
- Need to filter by program codes for precision
- Historical coverage varies

### 3.3 Recommended Approach for Funding Data

1. **Primary:** Use OpenAlex funder filters for post-1990 papers
2. **Supplement:** NSF Awards API for award-level data (amounts, PIs)
3. **Historical:** Need archival research for pre-1990 DARPA/NSF data
   - Congressional budget documents
   - DARPA program histories
   - Strategic Computing Initiative reports

---

## 4. Operationalizing "AI Research"

### 4.1 The Definitional Problem

OpenAlex's topic classification is **retroactive and ML-based**. This means:
- Modern definition of "AI" applied to historical papers
- Papers that didn't self-identify as "AI" may be included
- Papers that *did* self-identify as "AI" may be classified elsewhere

**Example issues I found:**
- Kissinger's "Variation of peak temperature" (1956) — thermal analysis paper classified as "Neural Networks" (misclassification)
- Hebb's "Organization of Behavior" (1950) — foundational neuroscience, not self-identified as AI

### 4.2 Multi-Method Approach

I recommend triangulating definitions:

**Method 1: OpenAlex subfield (broad)**
- Filter: `primary_topic.subfield.id:1702`
- Pros: Consistent, comprehensive
- Cons: Includes related work that may not self-identify as AI

**Method 2: Venue-based (institutional)**
- Define "AI venues": AAAI, IJCAI, NeurIPS, ICML, JAIR, Artificial Intelligence journal
- Query: `filter=primary_location.source.id:S###`
- Pros: Captures self-identified AI community
- Cons: Misses interdisciplinary work, venues change over time

**Method 3: Citation seed expansion**
- Start with canonical AI papers (Turing 1950, Dartmouth proposal, McCarthy LISP, etc.)
- Follow citations forward/backward
- Pros: Captures researchers' own sense of lineage
- Cons: May miss parallel traditions

**Method 4: Keyword + author intersection**
- Search for AI keywords in papers by known AI researchers
- Pros: Balances terminology shifts with community membership
- Cons: Requires pre-defining "AI researchers"

### 4.3 Validating Definitions

For your project on "constructed continuity," you could **compare** these definitions:
- Do papers classified as AI by OpenAlex cite each other more than expected?
- Do self-identified AI venues overlap with OpenAlex classification?
- When did terminology shift (cybernetics → AI → ML → deep learning)?

---

## 5. Concrete Data Collection Steps

### Phase 1: Build Core Dataset (Week 1-2)

1. **Download OpenAlex bulk data** for AI subfield
   ```
   # Filter for subfield 1702, years 1950-2025
   # ~3.5M works, will need bulk download
   ```

2. **Extract key fields:**
   - Work ID, title, abstract, publication_year
   - Author IDs, names, affiliations
   - Citation counts, referenced_works (for citation network)
   - Topic IDs (for tracking definitional shifts)
   - Funder IDs (for DARPA/NSF analysis)

3. **Build author network:**
   - Co-authorship edges from works
   - Institutional affiliations over time
   - Career trajectories (first AI paper → last)

### Phase 2: Funding Analysis (Week 2-3)

4. **Cross-reference with NSF Awards:**
   - Match author names to PI names
   - Match institutions
   - Build funding timeline per researcher

5. **DARPA analysis:**
   - Use OpenAlex funder filter for post-1990
   - Compile list of major DARPA AI programs from secondary sources
   - Map programs to publication patterns

### Phase 3: Definitional Analysis (Week 3-4)

6. **Track topic evolution:**
   - How do OpenAlex topic assignments shift over time?
   - Which topics gain/lose papers?
   - When do new AI topics emerge?

7. **Venue analysis:**
   - When were key AI venues founded?
   - How does venue-based definition differ from topic-based?

8. **Citation lineage:**
   - Do AI papers cite foundational works (Turing, McCarthy, etc.)?
   - Is there a "canon" that papers consistently reference?

---

## 6. Potential Analyses for "Constructed Continuity"

Based on Koch's framework and your angle, here are specific analyses:

### 6.1 Rhetorical Continuity
- **Method:** Track citations to "canonical" AI works over time
- **Hypothesis:** If continuity is constructed, papers should cite foundational works even when methods diverge
- **Data:** Citation networks from OpenAlex `referenced_works` field

### 6.2 Institutional Continuity
- **Method:** Map researcher movements between institutions
- **Hypothesis:** Continuity maintained through PhD lineages and lab migrations
- **Data:** Author-institution affiliations over time

### 6.3 Definitional Boundaries
- **Method:** Compare papers classified as "AI" at different time periods
- **Hypothesis:** What counts as "AI" expands/contracts with funding and hype cycles
- **Data:** Topic assignments, keyword frequencies

### 6.4 Funding-Driven Continuity
- **Method:** Correlate funding levels with publication patterns
- **Hypothesis:** "AI winters" visible in funding but not publications (researchers relabel work)
- **Data:** NSF/DARPA funding + publication counts

---

## 7. Technical Notes

### 7.1 Tools Recommended
- **Python:** `pyalex` library for OpenAlex API
- **R:** `bibliometrix` package for analysis
- **Neo4j or NetworkX:** For citation/co-authorship networks
- **Jupyter notebooks:** For iterative exploration

### 7.2 Data Volume Estimates
| Dataset | Size | Notes |
|---------|------|-------|
| AI works (1950-2025) | ~3.5M records | Bulk download recommended |
| Authors in AI subfield | ~1M unique | Extract from works |
| Citation edges | ~50M+ | Can sample if needed |
| DARPA-AI works | ~7K | API query sufficient |
| NSF awards (AI keyword) | ~20K | API query sufficient |

### 7.3 Storage Requirements
- Raw OpenAlex JSON: ~50-100GB
- Processed tabular data: ~5-10GB
- Citation network: ~10GB

---

## 8. Open Questions for Max

1. **Temporal focus:** Want to cover all 1950-2025, or focus on specific periods (e.g., Koch's three eras)?

2. **Validation strategy:** How do we know if our operationalization captures what *researchers* considered AI? Interviews? Archival documents?

3. **Comparative baseline:** Would it help to compare AI to another field (e.g., molecular biology, economics) to see if AI's continuity claims are unusual?

4. **Primary vs secondary:** Are you planning to collect primary data (interviews, archives) or primarily analyze bibliometric data?

5. **Causality vs description:** Are you aiming to *explain* how continuity was constructed, or *describe* that it was constructed?

---

## 9. Next Steps (After Your Review)

1. **If you want to proceed with OpenAlex:**
   - I can set up bulk data download
   - Build initial dataset of AI papers 1950-2025
   - Create co-authorship network

2. **If you want more definitional exploration:**
   - Compare venue-based vs topic-based definitions
   - Track term frequency shifts over time

3. **If you want funding focus:**
   - Deep dive into NSF Awards API
   - Compile DARPA program list from secondary sources

Let me know which direction to prioritize!

---

*Self-critique applied: A sociology reviewer would ask "How do you know OpenAlex's classification reflects social reality?" — valid concern. Recommend triangulating with venue-based and self-identification methods.*
