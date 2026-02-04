# AI Historiography Research Crosswalk
*Created: February 3, 2026*
*Central Argument: The continuity of "AI" as a field is constructed through funding categories, rhetorical lineage claims, and institutional persistence—not through stable epistemic criteria.*

---

## Phase 1: Axes Definition

### Rows: Research Questions

**TA1: Patronage Without Field**
| ID | Question | Operational Specification |
|----|----------|---------------------------|
| 1.1a | How did DARPA/NSF categorize AI-related funding programs over time? | Track program names/codes containing AI-related terms 1960-2025 |
| 1.1b | Did funding categories precede or follow researcher self-identification? | Compare timing: funding program labels vs. conference/journal naming |
| 1.1c | What was the relationship between funding program labels and venue naming? | Cross-reference new funding programs with founding dates of venues |
| 1.2a | Did total funding for "AI-related work" decline during winters? | Time series analysis of funding across multiple term variants |
| 1.2b | Did funding continue under alternative labels (ML, pattern recognition, etc.)? | Compare term-specific funding during 1974-80, 1988-93 periods |
| 1.2c | Under what labels did specific researchers (Hinton, LeCun) receive funding during winters? | Trace individual funding histories across terminology shifts |
| 1.3a | Did AI's legitimacy differ across audiences (CS vs. applied math vs. social science)? | Compare citation patterns to AI research from different fields |
| 1.3b | Which audience's approval mattered most for survival? | Correlate funder preferences with publication patterns |
| 1.3c | Did AI researchers seek publication in general-science venues as legitimation? | Track AI researcher publications in Science/Nature/PNAS over time |

**TA2: Reinvention (Constructed Continuity)**
| ID | Question | Operational Specification |
|----|----------|---------------------------|
| 2.1a | Did 1970s-80s researchers (symbolic AI) remain central after 2010? | Author overlap analysis: AAAI/IJCAI (1980-95) vs NeurIPS/ICML (2010-20) |
| 2.1b | Did 1980s neural network researchers become mainstream post-2012? | Track career trajectories of connectionist researchers |
| 2.1c | What is the citation relationship between eras? | Do post-2012 papers cite pre-1990 AI literature? |
| 2.1d | What are the PhD genealogies of 2010s leaders? | Were their advisors "AI researchers" or from adjacent fields? |
| 2.2a | When did "AI" become a publicly recognizable media term? | Google Ngrams + news article counts over time |
| 2.2b | What events/figures made AI publicly legible? | Identify first major media coverage moments |
| 2.2c | Did researchers relabel work to avoid "AI" during winters? | Track self-labeling changes in abstracts/titles |
| 2.3a | Can we detect temporal changes in how researchers narrate AI history? | Analyze keynotes, Turing lectures for retrospective lineage claims |
| 2.3b | Did researchers' self-described field labels change over careers? | Track individual researchers' terminology use 1990-2020 |
| 2.4a | Was deep learning a conceptual or hardware/scaling breakthrough? | Trace citations to 1980s connectionist work in post-2012 papers |
| 2.4b | Do 2012+ papers frame themselves as "reviving" old ideas or as "breakthroughs"? | Content analysis of framing in abstracts |

---

### Columns: Data Sources

| ID | Source | Access | Coverage | Status | Notes |
|----|--------|--------|----------|--------|-------|
| **S1** | **NSF Awards API** | Public | 1978-2026 | ✅ Collected | 8,373 AI awards + 9,505 related terms |
| **S2** | **OpenAlex API** | Public | 1950-2025 | ✅ Partial | 3.5M AI subfield works; bulk download available |
| **S3** | **OpenAlex-DARPA** | Public | 1974-2025 | ✅ Collected | 7,584 papers with DARPA funding acknowledgment |
| **S4** | **OpenAlex-ARPA** | Public | 1962-2025 | ✅ Collected | 2,399 papers (pre-DARPA) |
| **S5** | **OpenAlex-ONR** | Public | 1951-2025 | ✅ Collected | 6,618 papers (Office of Naval Research) |
| **S6** | **Google Ngrams** | Public | 1800-2019 | ⬜ Not started | Term frequency in published books |
| **S7** | **ProQuest/LexisNexis** | Institutional | 1980-2025 | ⬜ Not started | News article database |
| **S8** | **Semantic Scholar** | Public API | 1950-2025 | ⬜ Not started | Alternative to OpenAlex, different coverage |
| **S9** | **AAAI Proceedings** | Partial | 1980-2025 | ⬜ Not started | Conference proceedings (venue-based definition) |
| **S10** | **NeurIPS Proceedings** | Public | 1987-2025 | ⬜ Not started | Conference proceedings |
| **S11** | **IJCAI Proceedings** | Partial | 1969-2025 | ⬜ Not started | Conference proceedings |
| **S12** | **ICML Proceedings** | Public | 1984-2025 | ⬜ Not started | Conference proceedings |
| **S13** | **Mathematics Genealogy Project** | Public | Historical | ⬜ Not started | PhD advisor-advisee relationships |
| **S14** | **AI Magazine Archive** | AAAI | 1980-2025 | ⬜ Not started | AAAI's magazine, good for self-narratives |
| **S15** | **Turing Award Lectures** | ACM | 1966-2025 | ⬜ Not started | Retrospective framing by field leaders |
| **S16** | **DARPA Program Histories** | Mixed | 1958-2025 | ⬜ Not started | Official program documentation |
| **S17** | **Congressional Budget Docs** | Public | 1960-2025 | ⬜ Not started | Appropriations for AI-related programs |
| **S18** | **Strategic Computing Initiative** | Archive | 1983-1993 | ⬜ Not started | Major DARPA AI program documentation |
| **S19** | **Science/Nature/PNAS** | Institutional | 1950-2025 | ⬜ Not started | General science venues |
| **S20** | **Hardware Timeline** | Secondary | 1950-2025 | ⬜ Not started | GPU advances, Moore's Law milestones |

---

## Phase 2: The Matrix

### TA1: Patronage Without Field

| Question | S1 NSF | S2 OpenAlex | S3 DARPA | S4 ARPA | S5 ONR | S6 Ngrams | S7 News | S15 Turing | S16 DARPA Hist | S17 Congress |
|----------|--------|-------------|----------|---------|--------|-----------|---------|------------|----------------|--------------|
| **1.1a** Funder categorization | ⚫ program codes | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚫ program names | ⚫ line items |
| **1.1b** Funding vs self-ID timing | ⚫ earliest term use | ⚫ venue founding | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚫ | ⚪ |
| **1.1c** Funding-venue relationship | ⚫ program→venue | ⚫ venue metadata | ⚫ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚫ | ⚪ |
| **1.2a** Total funding decline? | ⚫ amount time series | ⚫ paper counts | ⚫ | ⚫ | ⚫ | ⚪ | ⚪ | ⚪ | ⚫ | ⚫ |
| **1.2b** Alternative label funding | ⚫ multi-term search | ⚫ topic shifts | ⚫ | ⚫ | ⚫ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| **1.2c** Individual funding traces | ⚫ PI name search | ⚫ author funding | ⚫ | ⚫ | ⚫ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| **1.3a** Audience-specific legitimacy | ⚪ | ⚫ cross-field cites | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| **1.3b** Which audience mattered? | ⚫ funding correlation | ⚫ publication patterns | ⚫ | ⚫ | ⚫ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| **1.3c** General-science publication | ⚪ | ⚫ venue filtering | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |

**Legend:** ⚫ = Primary source | ◐ = Supporting source | ⚪ = Not applicable

### TA2: Reinvention

| Question | S2 OpenAlex | S9 AAAI | S10 NeurIPS | S11 IJCAI | S12 ICML | S6 Ngrams | S7 News | S13 Genealogy | S14 AI Mag | S15 Turing | S20 Hardware |
|----------|-------------|---------|-------------|-----------|----------|-----------|---------|---------------|------------|------------|--------------|
| **2.1a** Personnel continuity | ⚫ author overlap | ⚫ | ◐ | ⚫ | ◐ | ⚪ | ⚪ | ◐ | ⚪ | ⚪ | ⚪ |
| **2.1b** Connectionist → mainstream | ⚫ career tracks | ◐ | ⚫ | ◐ | ⚫ | ⚪ | ⚪ | ⚫ | ⚪ | ⚪ | ⚪ |
| **2.1c** Cross-era citation | ⚫ referenced_works | ⚫ | ⚫ | ⚫ | ⚫ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| **2.1d** PhD genealogies | ◐ author institutions | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚫ | ⚪ | ◐ | ⚪ |
| **2.2a** "AI" as media term | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚫ | ⚫ | ⚪ | ⚪ | ⚪ | ⚪ |
| **2.2b** Legibility moments | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ◐ | ⚫ | ⚪ | ⚪ | ⚪ | ⚪ |
| **2.2c** Relabeling during winters | ⚫ title/abstract terms | ⚫ | ⚫ | ⚫ | ⚫ | ⚪ | ⚪ | ⚪ | ⚫ | ⚪ | ⚪ |
| **2.3a** Retrospective narratives | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚫ | ⚫ | ⚪ |
| **2.3b** Individual label changes | ⚫ author term use | ⚫ | ⚫ | ⚫ | ⚫ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| **2.4a** Conceptual vs hardware | ⚫ citation to 1980s | ⚪ | ⚫ | ⚪ | ⚫ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚫ |
| **2.4b** "Revival" vs "breakthrough" | ⚫ abstract framing | ⚪ | ⚫ | ⚪ | ⚫ | ⚪ | ⚪ | ⚪ | ◐ | ⚫ | ⚪ |

---

## Phase 2b: Gap Analysis & Efficiencies

### 🚨 Gaps (Questions Without Strong Sources)

| Question | Gap Description | Mitigation Strategy |
|----------|-----------------|---------------------|
| **1.1a** Pre-1978 funder categorization | NSF API has sparse data before 1978 | Congressional budget docs (S17), DARPA histories (S16) |
| **1.3a** Cross-field legitimacy | Requires careful field classification | Use OpenAlex field classifications; validate with manual sample |
| **2.2b** Legibility moments | Requires qualitative media analysis | Start with known events (Dartmouth 1956, Deep Blue 1997, AlphaGo 2016) |
| **2.3a** Retrospective narratives | Requires qualitative content analysis | Turing lectures (S15) are small corpus; AI Magazine editorials |
| **2.4a** Hardware timeline | No single authoritative source | Secondary sources: industry reports, Wikipedia hardware timeline |

### 🎯 High-Efficiency Sources (Multiple Questions)

| Source | Questions Addressed | Priority |
|--------|---------------------|----------|
| **S2 OpenAlex** | 1.1b, 1.1c, 1.2a-c, 1.3a-c, 2.1a-c, 2.2c, 2.3b, 2.4a-b | 🔴 CRITICAL |
| **S1 NSF Awards** | 1.1a-c, 1.2a-c, 1.3b | 🔴 CRITICAL |
| **S3 DARPA OpenAlex** | 1.1c, 1.2a-c, 1.3b | 🟡 HIGH |
| **S6 Google Ngrams** | 2.2a-b | 🟢 MODERATE |
| **S15 Turing Lectures** | 2.1d, 2.3a, 2.4b | 🟢 MODERATE |
| **S13 Math Genealogy** | 2.1b, 2.1d | 🟡 HIGH |

---

## Phase 3: Minianalyses

### Source-Based Task Organization

#### S1: NSF Awards Database (✅ Data Collected)
| Mini ID | Task | Data Subset | Output |
|---------|------|-------------|--------|
| M1.1 | Extract program code taxonomy | All 8,373 AI awards | `program_codes.csv` with temporal coding |
| M1.2 | Build funding time series by term | 9,505 multi-term awards | Time series visualization + stats |
| M1.3 | Trace winter-period relabeling | Awards 1974-80, 1988-95 | Term succession graph |
| M1.4 | PI career trajectories | Top 50 PIs by funding | Individual funding histories |
| M1.5 | Institutional leadership shifts | All awards by institution | Decade-by-decade rankings |

#### S2: OpenAlex (⚠️ Bulk Download Needed)
| Mini ID | Task | Data Subset | Output |
|---------|------|-------------|--------|
| M2.1 | Download AI subfield bulk data | Subfield 1702, 1950-2025 | ~3.5M works dataset |
| M2.2 | Build author co-authorship network | All AI authors | Network file (edges + nodes) |
| M2.3 | Extract citation network | Top 100K cited works | Citation graph |
| M2.4 | Track topic evolution | Topic assignments by year | Topic emergence timeline |
| M2.5 | Author overlap: AAAI/IJCAI vs NeurIPS/ICML | Venue-filtered authors | Jaccard similarity by era |
| M2.6 | Cross-era citation analysis | Post-2012 → pre-1990 citations | "Canon" identification |
| M2.7 | Term frequency in abstracts | "AI" vs "ML" vs "neural" by year | Terminology shift visualization |
| M2.8 | Cross-field citation | AI works ← citations from other fields | Legitimacy pattern by field |
| M2.9 | Science/Nature/PNAS tracking | AI authors publishing in general venues | Prestige-seeking pattern |

#### S3-S5: Military Funding (✅ Data Collected)
| Mini ID | Task | Data Subset | Output |
|---------|------|-------------|--------|
| M3.1 | DARPA funding timeline | 7,584 DARPA-AI papers | Year-by-year publication counts |
| M3.2 | Compare DARPA vs ARPA vs ONR | All three datasets | Funder comparison visualization |
| M3.3 | Top DARPA-funded researchers | Author extraction | Key personnel list |
| M3.4 | DARPA topic evolution | Topic assignments over time | Program-to-topic mapping |

#### S6: Google Ngrams (⬜ Not Started)
| Mini ID | Task | Data Subset | Output |
|---------|------|-------------|--------|
| M6.1 | Term frequency: "artificial intelligence" | 1950-2019 | Time series graph |
| M6.2 | Compare: AI vs cybernetics vs ML vs neural | All terms | Relative frequency comparison |
| M6.3 | Identify inflection points | All terms | Event-to-term correlation |

#### S7: News Database (⬜ Not Started)
| Mini ID | Task | Data Subset | Output |
|---------|------|-------------|--------|
| M7.1 | Article count: "artificial intelligence" | 1970-2025 | Time series |
| M7.2 | Identify "legibility events" | Spikes in coverage | Event list with dates |
| M7.3 | Sentiment analysis (if time permits) | Major coverage periods | Tone shift analysis |

#### S13: Mathematics Genealogy Project (⬜ Not Started)
| Mini ID | Task | Data Subset | Output |
|---------|------|-------------|--------|
| M13.1 | Trace PhD lineages of 2010s leaders | Hinton, LeCun, Bengio, Ng, etc. | Genealogy trees |
| M13.2 | Identify advisor field classifications | All traced advisors | Field continuity assessment |
| M13.3 | Compare symbolic vs connectionist lineages | 1980s AAAI vs 2010s NeurIPS leaders | Population replacement analysis |

#### S15: Turing Award Lectures (⬜ Not Started)
| Mini ID | Task | Data Subset | Output |
|---------|------|-------------|--------|
| M15.1 | Collect AI-related Turing lectures | McCarthy 1971, Minsky 1969, Hinton/LeCun/Bengio 2018, etc. | Lecture corpus |
| M15.2 | Code for retrospective lineage claims | All collected lectures | Coded themes |
| M15.3 | Identify "revolution" vs "continuation" framing | 2018 lecture specifically | Qualitative memo |

#### S16: DARPA Histories (⬜ Not Started)
| Mini ID | Task | Data Subset | Output |
|---------|------|-------------|--------|
| M16.1 | Compile AI program timeline | All DARPA AI programs | Program list with dates, budgets |
| M16.2 | Extract program category definitions | Strategic Computing, IPTO, etc. | How DARPA defined "AI" |
| M16.3 | Cross-reference with publications | Programs → OpenAlex papers | Funding-to-output mapping |

---

## Phase 4: Controlled Vocabulary

### Initial Keywords (Empirical Terms)

| Domain | Keywords |
|--------|----------|
| **Terminology** | artificial intelligence, machine learning, neural network, deep learning, expert system, connectionist, pattern recognition, knowledge-based, cybernetics, automata, cognitive science, intelligent systems |
| **Institutions** | DARPA, NSF, ARPA, ONR, MIT, Stanford, CMU, Berkeley |
| **Venues** | AAAI, IJCAI, NeurIPS, ICML, JAIR, AI Magazine, Artificial Intelligence (journal) |
| **Periods** | Dartmouth (1956), First winter (1974-80), Expert systems boom (1985-92), Second winter (1988-95), Deep learning (2012+) |
| **People** | McCarthy, Minsky, Newell, Simon, Feigenbaum, Hinton, LeCun, Bengio, Ng |

### Theoretical Abstractions (Phase 4 Refinement)

| Empirical Term | Theoretical Concept |
|----------------|---------------------|
| "AI winter" | Delegitimation period |
| "AI" label persistence | Symbolic continuity |
| Relabeling (AI → ML → DL) | Boundary work |
| Funding categories | Material patronage |
| Citation to founders | Lineage construction |
| Venue membership | Community boundary |
| PhD lineage | Personnel reproduction |
| Cross-field publication | Legitimation strategy |
| "Revolution" vs "revival" framing | Discontinuity rhetoric |

### Controlled Vocabulary for Coding

| Code | Definition | Indicators |
|------|------------|------------|
| **CONT_SYMB** | Symbolic continuity | Use of "AI" label, citation to canonical works, origin story references |
| **CONT_PERS** | Personnel continuity | Same researchers across eras, PhD lineages, co-authorship persistence |
| **CONT_INST** | Institutional continuity | Same venues, departments, funding programs |
| **CONT_IDEA** | Ideational continuity | Same theories, methods, or problems across eras |
| **DISCONT_TERM** | Terminological discontinuity | Label avoidance, relabeling work |
| **DISCONT_METH** | Methodological discontinuity | New methods displacing old (symbolic → connectionist) |
| **DISCONT_PERS** | Personnel discontinuity | Population replacement, exodus |
| **LEGIT_FUND** | Funder legitimation | Funding acknowledged, program alignment |
| **LEGIT_PEER** | Peer legitimation | Citations from other CS fields |
| **LEGIT_PUBLIC** | Public legitimation | Media coverage, general-science publication |

---

## Phase 5: Feasibility & Misfit Management

### Data Quality Concerns

| Source | Concern | Mitigation |
|--------|---------|------------|
| NSF Awards | Pre-1978 sparse; keyword false positives | Use alternative terms (cybernetics, automata) for earlier periods; manual validation sample |
| OpenAlex | Retroactive ML-based classification | Triangulate with venue-based definition; acknowledge methodologically |
| DARPA OpenAlex | Acknowledgment parsing misses pre-1990 | Supplement with DARPA program histories |
| Google Ngrams | Ends 2019; books ≠ research discourse | Supplement with Semantic Scholar term trends |
| Math Genealogy | Incomplete for non-math fields | Supplement with OpenAlex author-institution histories |

### Provenance Requirements

Every data point in analysis must include:
1. **Source ID** (from S1-S20 list)
2. **Query/filter** used to obtain it
3. **Date retrieved**
4. **Unique identifier** (NSF award ID, OpenAlex work ID, etc.)

### Defensive Definitions

| Term | Defensive Definition | Why Defensible |
|------|---------------------|----------------|
| "AI research" | Work published in venues self-identifying as AI (AAAI, IJCAI, etc.) OR classified as AI subfield (1702) by OpenAlex | Combines community self-identification with bibliometric classification |
| "AI winter" | Period where federal funding AND publication counts using "artificial intelligence" terminology decline >20% | Quantifiable, avoids circular definition |
| "Personnel continuity" | Jaccard similarity >0.3 between author sets of era A venues and era B venues | Measurable overlap threshold |
| "Lineage claim" | Citation to pre-1990 work in post-2012 paper OR explicit reference to historical figure in abstract | Observable in citation data |

### Misfit Contingencies

| If this happens... | Then pivot to... |
|--------------------|------------------|
| OpenAlex classification too noisy | Restrict to venue-based definition (AAAI/IJCAI/NeurIPS/ICML only) |
| Pre-1978 NSF data too sparse | Shift to publication-only analysis for early period; use congressional docs for funding |
| Personnel overlap is high (undermines discontinuity thesis) | Reframe as "how did continuous personnel maintain field across methodological ruptures?" |
| Personnel overlap is low (undermines continuity thesis) | Reframe as "how did discontinuous personnel construct rhetorical continuity?" |
| Cannot access ProQuest/LexisNexis | Use Google Trends (2004+) as proxy for public awareness |

---

## Appendix A: Files Created During Data Collection

```
research/data/
├── nsf_awards/
│   ├── consolidated_ai_awards.csv      # 8,373 unique AI awards
│   ├── combined_ai_terminology.csv     # 9,505 awards across all terms
│   └── [raw JSON files]
├── darpa/
│   └── openalex_darpa_ai_page*.json   # 7,584 DARPA-AI papers
├── arpa/
│   └── [ARPA-funded papers]            # 2,399 papers
├── onr/
│   └── [ONR-funded papers]             # 6,618 papers
└── README.md
```

---

## Appendix B: Key Findings Already Obtained

### Terminology Succession (from NSF data)
```
1983-1992: Expert Systems peak (20-24 awards/year)
1987-1993: Connectionist peak
1991-1995: Neural Networks wave 1
1993-1997: Knowledge-based peak
1996-2009: Machine Learning gradual rise
2010-2015: Neural Networks wave 2
2013+: Deep Learning emerges
2018+: "AI" terminology explosion (1000+ awards/year)
```

### Institutional Leadership Shifts
```
1980s: Stanford, Maryland, MIT
1990s: Stanford, CMU, Maryland
2000s: CMU dominant
2010s: CMU, Georgia Tech, USC
2020s: Purdue, CMU, Georgia Tech, Penn State
```

### AI Winter Evidence
- 2000s had *fewer* NSF AI awards than 1990s (256 vs 493)
- Term "AI" avoided; "machine learning" used instead
- Funding persisted but under different labels

---

## Next Steps

### Immediate (This Week)
1. ⬜ M2.1: Download OpenAlex bulk data
2. ⬜ M6.1-M6.3: Run Google Ngrams queries
3. ⬜ M15.1: Collect Turing Award lectures (small corpus)

### Short-term (Next 2 Weeks)
4. ⬜ M2.5: Author overlap analysis (AAAI/IJCAI vs NeurIPS/ICML)
5. ⬜ M13.1: Trace PhD genealogies for key researchers
6. ⬜ M1.2-M1.3: Visualize funding time series

### Medium-term (Month)
7. ⬜ M2.6: Cross-era citation analysis
8. ⬜ M7.1-M7.2: News database analysis (if access available)
9. ⬜ M16.1-M16.2: DARPA program history compilation

---

*Document version: 1.0*
*Last updated: 2026-02-03 21:45 CST*
