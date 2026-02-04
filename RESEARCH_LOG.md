# AI Historiography Research Log
*Started: February 3, 2026*
*Researcher: Tia (for Max)*

---

## Session 1: February 3, 2026 (Evening)

### Objectives
1. Compare OpenAlex vs Web of Science vs Scopus for this research
2. Collect NSF funding data for AI-related research
3. Collect DARPA funding data
4. Document methodology thoroughly

---

## Data Collection Log

### 1. NSF Awards API

**Source:** https://api.nsf.gov/services/v1/awards.json

**Access:** Public, no authentication required

**Rate limits:** Not documented, but ~1 request/second seems safe

**Query parameters used:**
- `keyword` — phrase search (must URL-encode quotes for exact phrase)
- `dateStart`, `dateEnd` — filter by award start date (MM/DD/YYYY format)
- `printFields` — select which fields to return
- `rpp` — results per page (max 100)
- `offset` — pagination offset

**Fields collected:**
- `id` — NSF award number
- `title` — Award title
- `startDate`, `expDate` — Start and expiration dates
- `estimatedTotalAmt` — Funding amount (USD, not inflation-adjusted)
- `piFirstName`, `piLastName` — Principal investigator name
- `awardeeName` — Institution receiving award
- `fundProgramName` — NSF program name

**Searches conducted:**

| Search Term | Date Range | Records | Files |
|-------------|------------|---------|-------|
| "artificial intelligence" | All | ~8,400 | ai_batch1.json, ai_offset_*.json |
| "artificial intelligence" | 1960-1985 | 44 | ai_historical_1960_1985.json |
| "artificial intelligence" | 1985-1995 | 300 | ai_historical_1985_1995_*.json |
| "artificial intelligence" | 1995-2005 | 300 | ai_historical_1995_2005_*.json |
| "artificial intelligence" | 2005-2015 | 300 | ai_historical_2005_2015_*.json |
| "machine learning" | All | ~1,600 | ml_*.json |
| "expert system" | All | 264 | expert_systems_*.json |
| "neural network" | All | ~1,100 | neural_networks_*.json |
| "connectionist" | All | 86 | connectionist.json |
| "deep learning" | All | 100 | deep_learning.json |
| "natural language" | All | 100+ | natural_language.json |
| "computer vision" | All | 100+ | computer_vision.json |
| "pattern recognition" | 1970-2000 | 100 | pattern_recognition_1970_2000.json |
| "knowledge-based" | 1980-2000 | 100 | knowledge_based_1980_2000.json |

**Known limitations:**
1. API returns most recent awards first; historical coverage requires date filters
2. Phrase search may miss relevant awards using different terminology
3. Pre-1978 data appears sparse or missing
4. Award amounts not inflation-adjusted
5. Some false positives (e.g., "artificial" matches "artificial pinning centers")

**Data quality notes:**
- Earliest "artificial intelligence" exact phrase match: 1978
- Notable early awards found: McCarthy (Stanford, 1981), Feigenbaum (Stanford, 1983), Minker (Maryland, 1983)
- Clear terminology shifts visible in data (see Terminology Analysis section)

---

### 2. OpenAlex API (DARPA-funded papers)

**Source:** https://api.openalex.org/

**Access:** Public, optional API key for higher rate limits

**Rate limits:** 100,000 requests/day with API key, 10/second without

**Query for DARPA AI papers:**
```
/works?filter=funders.id:F4320332180,primary_topic.subfield.id:1702&per_page=200&page=N
```

**Key IDs:**
- DARPA funder ID: `F4320332180`
- AI subfield ID: `1702` (under Computer Science)

**Records collected:** 7,584 papers (38 pages × 200 per page)

**Fields available per paper:**
- `id` — OpenAlex work ID
- `title` — Paper title
- `abstract` — Abstract text (when available)
- `publication_year` — Year published
- `cited_by_count` — Citation count
- `authorships` — Array of authors with:
  - `author.id`, `author.display_name`
  - `institutions` — Array of affiliated institutions
- `primary_topic` — OpenAlex topic classification
- `referenced_works` — Array of cited work IDs (for citation network)
- `primary_location.source` — Journal/conference info

**Temporal coverage:**
- Earliest DARPA-AI paper: 1974 (Herbert Simon)
- Sparse before 1990 (funding acknowledgments not standardized)
- Dense coverage 2000+

**Known limitations:**
1. Funder data depends on acknowledgment parsing — underestimates historical DARPA influence
2. Topic classification is retroactive (ML-based, trained on modern data)
3. Some papers may be classified as AI that didn't self-identify as such

---

### 3. OpenAlex vs Web of Science vs Scopus Comparison

**Research consulted:**
- Alperin et al. (2024) "An analysis of the suitability of OpenAlex for bibliometric analyses" (arXiv:2404.17663)
- Culbert et al. (2025) "Reference coverage analysis of OpenAlex compared to Web of Science and Scopus" (Scientometrics)
- Simard et al. (2024) "The open access coverage of OpenAlex, Scopus and Web of Science" (arXiv:2404.01985)

**Key findings from literature:**

1. **Coverage:** OpenAlex > Scopus ≈ WoS
   - OpenAlex: 240M+ works
   - Scopus: ~90M works
   - WoS: ~90M works
   - Alperin et al.: "OpenAlex is a superset of Scopus"

2. **Cost:**
   - OpenAlex: Free, CC0 license
   - Scopus: Institutional subscription required ($$$)
   - WoS: Institutional subscription required ($$$)

3. **Funder tracking:**
   - OpenAlex: Yes, parsed from acknowledgments
   - Scopus: Limited
   - WoS: Limited

4. **Historical depth:**
   - All three have reasonable coverage back to 1950s
   - WoS may have better curation for pre-1990 literature

5. **Citation networks:**
   - OpenAlex: Open, accessible via API
   - Scopus/WoS: Behind paywall

6. **Author disambiguation:**
   - OpenAlex: ML-based, generally good
   - Scopus: ORCID-based
   - WoS: Manual curation

**Recommendation for this project:**
- Primary: OpenAlex (free, good coverage, funder data, citation networks)
- Validation: WoS if institutional access available (better historical curation)

---

## Terminology Analysis

### Methodology
Searched NSF Awards API for different AI-related terms to track how researchers relabeled work during "AI winters."

### Findings

**Expert Systems (264 awards)**
- First appearance: 1983
- Peak: 1985-1992 (20-24 awards/year)
- Decline: Sharp drop after 1992 (7 awards in 1993)
- Context: Primary "escape term" during first AI winter

**Connectionist (86 awards)**
- First appearance: 1984
- Peak: 1987-1993
- Context: Alternative to "neural network" during period when both "AI" and "neural" were stigmatized
- Associated with Rumelhart, McClelland, PDP Research Group

**Neural Networks (~1,100 awards)**
- Two distinct waves:
  - Wave 1: 1991-1995 (peak ~47/year in 1993)
  - Gap: 1996-2009 (sparse)
  - Wave 2: 2010+ (accelerating)
- Context: Term revived with deep learning success

**Knowledge-based (100+ awards)**
- Peak: 1993-1997
- Context: Filled vacuum after expert systems declined

**Machine Learning (1,600+ awards)**
- Gradual rise from mid-1990s
- Explosion post-2015
- Context: Became acceptable umbrella term

**Deep Learning (100+ awards)**
- First appearance: ~2013
- Rapid growth: 2015+
- Context: Specific technical term, less stigmatized

**Artificial Intelligence (8,400+ awards)**
- Sparse: Pre-2015 (~50-100/year)
- Explosion: 2018+ (1,000+/year)
- Context: Term reclaimed after AlphaGo, GPT, etc.

### Interpretation
The data strongly supports the "terminology avoidance" thesis:
1. Researchers relabeled the same type of work depending on funding climate
2. Clear succession: Expert Systems → Connectionist → Knowledge-based → Machine Learning → Deep Learning → AI
3. The "AI winter" is visible not as absence of research but as absence of the *term* "AI"

---

## Files Created

### Data files
```
research/data/
├── nsf_awards/
│   ├── consolidated_ai_awards.csv      # 8,373 unique AI awards
│   ├── combined_ai_terminology.csv     # 9,505 awards across all terms
│   ├── ai_batch1.json                  # Raw API response
│   ├── ai_offset_*.json               # Paginated AI awards
│   ├── ai_historical_*.json           # Historical periods
│   ├── ml_*.json                      # Machine learning awards
│   ├── expert_systems_*.json          # Expert systems awards
│   ├── neural_networks_*.json         # Neural network awards
│   ├── connectionist.json             # Connectionist awards
│   ├── deep_learning.json             # Deep learning awards
│   ├── natural_language.json          # NLP awards
│   ├── computer_vision.json           # Computer vision awards
│   ├── pattern_recognition_*.json     # Pattern recognition awards
│   └── knowledge_based_*.json         # Knowledge-based awards
├── darpa/
│   └── openalex_darpa_ai_page*.json   # 7,584 DARPA-AI papers
└── README.md                          # Data documentation
```

### Documentation files
```
research/
├── ai-historiography-exploration.md    # Initial literature review
├── ai-historiography-data-collection.md # Methodology document
└── RESEARCH_LOG.md                     # This file
```

---

## Questions for Further Investigation

1. **Cross-referencing:** Can we match NSF PI names to OpenAlex author IDs to track career trajectories?

2. **Institutional patterns:** Which institutions led each terminological wave?

3. **Citation networks:** Do papers using different terms cite each other? (Tests whether it's truly the "same" field)

4. **Funding amounts:** How did total funding levels change across terminology shifts?

5. **International comparison:** Is the terminology shift pattern US-specific or global?

6. **Validation:** How well does NSF phrase search capture actual AI research? What's the false positive/negative rate?

---

## Next Session TODO

- [ ] Continue paginating remaining searches (computer vision, NLP need more pages)
- [ ] Cross-reference NSF PIs with OpenAlex authors
- [ ] Extract citation network from DARPA papers
- [ ] Calculate total funding by year/decade
- [ ] Look for pre-1978 AI funding using alternative terms (cybernetics, automata theory)
- [ ] Check for ARPA (pre-DARPA) funding in OpenAlex

---

*Log updated: 2026-02-03 19:30 CST*

---

## Session 1 Continued: Deep Analysis

### NSF Funding Analysis by Decade

| Decade | Awards | Total Funding | Avg Grant |
|--------|--------|---------------|-----------|
| 1970s | 7 | $0.77M | $111K |
| 1980s | 88 | $17.33M | $197K |
| 1990s | 493 | $122.05M | $248K |
| 2000s | 256 | $65.72M | $257K |
| 2010s | 1,585 | $842.52M | $532K |
| 2020s | 6,815 | $6,703M | $984K |

**Key observation:** The 2000s shows *fewer* awards than the 1990s — this may reflect post-dot-com AI winter.

### Top Institutions (AI Awards)

1. Carnegie Mellon University (198)
2. Georgia Tech Research Corporation (160)
3. University of Michigan (146)
4. University of Texas at Austin (136)
5. University of Illinois Urbana-Champaign (135)
6. Purdue University (129)
7. Massachusetts Institute of Technology (129)
8. Penn State (128)
9. University of Washington (124)
10. Arizona State University (124)
11. Stanford University (120)

### DARPA AI Papers Analysis

**Top Institutions in DARPA-funded AI:**
1. Carnegie Mellon University (1,469 papers)
2. University of Southern California (1,127)
3. University of Illinois Urbana-Champaign (871)
4. University of Washington (818)
5. Stanford University (795)
6. MIT (789)

**Most Cited DARPA-funded AI Papers:**
1. XGBoost (2016) - 43,298 citations
2. Transfer Learning Survey (2009) - 22,322 citations
3. BLEU (2001) - 20,720 citations
4. LIME "Why Should I Trust You?" (2016) - 13,780 citations

**Citations by Decade:**
- 1970s: 10 papers, 364 cites (avg 36.4)
- 1980s: 190 papers, 17,332 cites (avg 91.2)
- 1990s: 473 papers, 58,932 cites (avg 124.6)
- 2000s: 1,487 papers, 162,866 cites (avg 109.5)
- 2010s: 2,589 papers, 255,916 cites (avg 98.8)
- 2020s: 2,833 papers, 65,090 cites (avg 23.0)

### John McCarthy Funding History

John McCarthy (Stanford) received $4.43M total from NSF:
- 1974: $989,800 - Computer Integrated Assembly Systems (earliest)
- 1977: $274,400 - Dialnet, Automatic Programming, OS Verification
- 1979: $297,350 - Mechanization of Formal Reasoning
- 1981: $295,367 - Basic Research in AI
- 1982: $398,964 - Mechanical Theorem Proving
- 1984: $407,532 - Basic Research in AI
- 1985: $37,168 - US-Japan AI Cooperation
- 1988: $350,351 - Mechanical Theorem Proving
- 1989: $389,277 - Basic Research in AI
- 1990s: Various grants on formal reasoning

### Pre-AI Terminology Findings

Found awards using pre-AI terminology:
- **Cybernetics**: 82 awards, earliest 1970
- **Automata**: 100+ awards, earliest 1969
- **Connectionist**: 86 awards, peaked 1987-1993
- **Heuristic**: 93+ awards in 1960-1990 period

Key finding: Automata theory awards exist back to 1969, predating explicit "AI" terminology in NSF records.

### CMU AI History

CMU received 24 "artificial intelligence" awards pre-2000:
- 1986: $6.7M Engineering Research Center
- 1987: Tom Mitchell Presidential Young Investigator
- 1992: Jack Mostow speech recognition project
- 1995: Manuela Veloso CAREER award
- 1996: Andrew Moore CAREER award
- 1999: Sebastian Thrun CAREER award

---


### Institutional Leadership by Decade

**1980s (88 awards total):**
1. Stanford University (6)
2. University of Maryland (6)
3. University of Washington (5)
4. MIT (3)
5. Carnegie Mellon (3)

**1990s (493 awards total):**
1. Stanford University (16)
2. Carnegie Mellon University (14)
3. University of Maryland (9)
4. University of Illinois (9)
5. MIT (7)

**2010s (1,585 awards total):**
1. Carnegie Mellon University (44)
2. Georgia Tech (33)
3. USC (32)
4. MIT (30)
5. Arizona State (25)

**2020s (6,815 awards total):**
1. Purdue University (112)
2. Carnegie Mellon University (108)
3. Georgia Tech (103)
4. Penn State (102)
5. University of Michigan (97)

**Key observations:**
- Stanford and MIT dominated early AI funding but CMU overtook them by 2000s
- Georgia Tech emerged as a powerhouse in 2010s
- 2020s shows major diversification; Purdue now leads
- AAAI (Association for Advancement of AI) receives significant funding in 2010s (24 awards)

### Files Summary

**Final datasets created:**
- `nsf_ai_awards_dedup.csv` — 8,373 unique NSF AI awards
- `ai_terminology_dedup.csv` — 9,505 unique awards (all AI-related terms)
- `ai_funding_timeline.csv` — Combined timeline (NSF, DARPA, ARPA, ONR)
- `terminology_timeline.csv` — Term usage over time
- DARPA papers: 7,584 in OpenAlex
- ARPA papers: 2,399 in OpenAlex
- ONR papers: 6,618 in OpenAlex

### Data Collection Complete

**Total unique records collected:**
- NSF Awards: ~10,000 (across all AI-related terms)
- OpenAlex Papers: ~16,600 (DARPA + ARPA + ONR funded AI)
- Total: ~26,000+ unique funding/publication records

**Temporal coverage:**
- NSF: 1978-2026
- OpenAlex/ARPA: 1962-2025
- OpenAlex/ONR: 1951-2025 (though early years noisy)

**Storage:** ~231MB total

---

*Session 1 complete. Ready for next iteration with Max.*
*Log closed: 2026-02-03 ~20:00 CST*
