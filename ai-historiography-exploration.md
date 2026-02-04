# AI Historiography Research Memo
*Initial exploration for Max | February 3, 2026*

---

## 1. Koch & Peterson's "Epistemic Monoculture" Thesis

**Paper**: "From Protoscience to Epistemic Monoculture: How Benchmarking Set the Stage for the Deep Learning Revolution"  
**Authors**: Bernard J. Koch & David Peterson  
**Published**: arXiv:2404.06647 (April 2024)  
**Link**: https://arxiv.org/abs/2404.06647

### Key Arguments

Koch & Peterson present a **three-era history** of AI research:

1. **Era 1 (1950s–late 1980s)**: "Basic science" approach
   - Autonomous exploration, organic progress assessments
   - Peer review, theoretical consensus as success metrics
   - Field approached AI as fundamental research without clear metrics

2. **Era 2 (late 1980s–2010s)**: The benchmarking turn
   - "AI Winter" forced a retrenchment after funding dried up
   - **U.S. government intervention** reoriented field toward measurable progress
   - "Benchmarking" emerged: quantify progress via predictive accuracy on datasets
   - Focused on tasks of military and commercial interest

3. **Era 3 (2010s–present)**: Epistemic monoculture
   - Consolidation around scaling deep learning models
   - Achievements + limitations (explainability, ethics, efficiency)
   - Benchmarking created "clear signals of significance and progress" but...
   - **Tradeoff**: "consolidation around external interests and inherent conservatism of benchmarking has disincentivized exploration beyond scaling monoculture"

### Relevance to Your Project

Koch's thesis aligns with but differs from your angle:
- **Koch**: Benchmarking created monoculture (structural/methodological argument)
- **Your angle**: Researchers *actively constructed* the field's continuity/identity

This is a **productive tension** — you can explore how the benchmarking turn was *also* a moment of active identity construction. The introduction of benchmarks didn't just reshape methods; it gave researchers a shared language for claiming progress and lineage.

---

## 2. Methods for Defining "AI Research" Historically

This is **the crux of your methodological challenge**. I found several approaches used in scientometric literature:

### A. Keyword/Term-Based Approaches

**Pros**: Simple, reproducible, works at scale  
**Cons**: Field terminology changes drastically over time; misses work that doesn't self-identify as "AI"

**Key issue**: "AI" wasn't stable terminology. Pre-1956: cybernetics, automata theory, operations research. 1980s-2000s: ML, neural networks, expert systems often distinguished from "AI." 2010s+: everything is "AI" again.

### B. Venue-Based Delineation

Use publication venues as proxy for field membership (e.g., AAAI, NeurIPS, ICML, JAIR).

**Pros**: Clean boundaries, conferences/journals have clear membership  
**Cons**: Misses interdisciplinary work, venues themselves evolved (NeurIPS was founded 1987), doesn't capture historical pre-venue work

### C. Citation Network / Co-Citation Analysis

**Key paper**: Van den Besselaar & Heimeriks (1996) "Mapping change in scientific specialties: A scientometric reconstruction of the development of artificial intelligence"  
*Journal of the American Society for Information Science, 47(6), 415-436*

This is **highly relevant** — they used co-citation clustering to reconstruct AI as a field over time, distinguishing AI-as-specialty vs. AI-as-methodology (techniques spreading to other fields).

**Pros**: Lets the field define itself through citation practices; captures evolution  
**Cons**: Requires extensive data work; depends on starting "seed" papers

### D. Self-Identification + Snowball

Start with acknowledged "AI" works (Turing 1950, Dartmouth proposal, etc.) and follow citations forward/backward.

**Pros**: Captures researchers' own sense of lineage  
**Cons**: Reproduces origin myths; may miss heterodox lineages

### E. Hybrid: JRC AI Watch Methodology

The EU's Joint Research Centre maintains an "AI Watch" that delineates AI into **8 domains** using a combination of keyword taxonomies + expert curation. Worth investigating for contemporary definition.

### Recommendation

For your historical work, I'd suggest:
1. **Multi-method triangulation**: Start with venues + keywords, validate with citation networks
2. **Explicitly theorize the boundary problem** — make "how did researchers define AI at different moments?" a research question itself
3. Use Koch's "eras" as periodization hypothesis, then test whether boundary definitions shift at those breaks

---

## 3. Funding Data Sources

### A. NSF Awards Search
**URL**: https://www.nsf.gov/awardsearch/  
**Coverage**: 1952–present  
**Access**: Public, searchable, downloadable  
**How to use**:
- Search by keyword ("artificial intelligence", "machine learning", "neural network", etc.)
- Filter by directorate/division (CISE = Computer & Information Science & Engineering)
- Can download bulk data via API

**Historical note**: NSF didn't have a dedicated AI program until recently, so early AI funding was scattered across programs. Look at:
- Division of Computer and Computation Research (1967–)
- Information, Robotics, and Intelligent Systems (IRIS) program
- Computing and Communication Foundations

### B. DARPA Programs

**Key resource**: Fouse, Cross & Lapin (2020) "DARPA's Impact on Artificial Intelligence" *AI Magazine* 41(2): 3-8  
https://doi.org/10.1609/aimag.v41i2.5294

**DARPA's "Three Waves of AI"**:
1. Wave 1: Handcrafted knowledge (expert systems)
2. Wave 2: Statistical learning
3. Wave 3: Contextual adaptation

**Historical programs to investigate**:
- Information Processing Techniques Office (IPTO) — key early funder
- Strategic Computing Initiative (1983-1993) — $1B program
- DARPA AI Next (2018+)

**Challenge**: DARPA historical data is harder to access than NSF. Some approaches:
- Published DARPA histories and program summaries
- Congressional testimony and budget documents
- Roland & Shiman "Strategic Computing" (2002) book

### C. Other U.S. Federal Sources

- **NIH**: For AI in medicine/biology (RePORTER database)
- **DOE**: For scientific computing applications
- **IARPA**: Intelligence community AI
- **Congressional Research Service reports**: Good for funding landscape overviews

### D. International Funding

- **EU Framework Programmes**: CORDIS database (searchable)
- **UKRI**: UK Research and Innovation database
- **Japan**: MITI/METI Fifth Generation Computer project (1982-1992)

### E. Private/Corporate

This is **harder to track** but crucial:
- IBM research spending (some in annual reports)
- Bell Labs (pre-breakup)
- Xerox PARC
- Google/DeepMind, OpenAI, etc. (recent era)

**Note**: No single database captures private R&D spending on AI historically.

---

## 4. Existing Datasets & Prior Historiographic Work

### A. AI Index Report (Stanford HAI)
https://aiindex.stanford.edu/report/

Annual report with data on publications, citations, funding (recent decades). Good for 2010s+ but limited historical depth.

### B. Semantic Scholar / OpenAlex

Large-scale bibliometric databases. OpenAlex (successor to Microsoft Academic Graph) has open data you could use to build AI publication datasets.

### C. Prior Historiographic Scholarship

**BJHS Themes Special Issue (2023)**: "Histories of Artificial Intelligence: A Genealogy of Power"  
https://www.cambridge.org/core/journals/bjhs-themes/article/histories-of-artificial-intelligence-a-genealogy-of-power/

**Key framing**: They argue against origin myths (Dartmouth 1956, Turing 1950) and for **plural genealogies** situating AI within:
- Histories of management and control
- Colonial power structures
- Cold War social science
- Industrial and military logics

**Methodological note**: They emphasize AI's continuities come from **managerial logics** (population management, biometrics, bureaucratic rationalization) more than technical lineages. This resonates with your interest in "constructed continuity."

**Other key historiographic works**:
- Pamela McCorduck, *Machines Who Think* (1979, updated 2004) — classic insider history
- Stuart Russell & Peter Norvig, *AIMA* textbook historical chapters
- Nils Nilsson, *The Quest for Artificial Intelligence* (2009)
- Daniel Crevier, *AI: The Tumultuous History* (1993)
- Jonnie Penn's work on AI and social control

---

## 5. Methodological Challenges

### A. Presentism
Contemporary definitions of "AI" (especially post-2015 deep learning boom) may not map onto historical work. Researchers in 1970 wouldn't recognize today's AI as continuous with their work, necessarily.

### B. Field vs. Method
Is AI a *field* (community of researchers) or a *method* (set of techniques used across fields)? Van den Besselaar (1996) found evidence for both. This distinction matters for how you operationalize continuity.

### C. Funding ≠ Research
Government funding patterns don't perfectly capture research activity. Much foundational work happened in industry labs (Bell Labs, IBM, Xerox PARC) or with minimal funding.

### D. Multiple Lineages
Your four dimensions (funding, ideas, personnel, institutions) may tell different stories:
- Funding: discontinuous (AI winters)
- Ideas: more continuous (problems persist)
- Personnel: moderately continuous (PhD lineages)
- Institutions: complex (venues emerge/die, labs form/close)

---

## 6. Recommended Next Steps

1. **Get the Koch paper** — Read the full text, especially methodology section (qualitative interviews + computational analysis)

2. **Van den Besselaar (1996)** — Key methodological model for bibliometric field reconstruction. Your library should have it; if not I can help find it.

3. **Define your own operationalization**: Draft 2-3 candidate definitions of "AI research" and see how they perform on sample historical data

4. **Web of Science pilot**: Pick a test period (e.g., 1980-1990) and try keyword vs. venue-based delineation. How different are the results?

5. **NSF pilot**: Search NSF awards database for "artificial intelligence" across decades. Map funding trends.

6. **Contact Koch/Peterson?** — Their paper is recent; they might share data or methods.

---

## 7. Open Questions for You

1. **Temporal scope**: 1950s–present is huge. Any natural breakpoints where you'd focus? Koch's three eras? AI winters (1974-80, 1987-93)?

2. **What counts as "actively constructed continuity"?** Are you looking at:
   - Rhetorical moves (papers citing foundational works)?
   - Institutional moves (conferences claiming lineage)?
   - Definitional moves (what gets called "AI")?
   - All of the above?

3. **Comparative angle?** Would it help to compare AI to another field that *didn't* maintain continuity claims (or did so differently)?

4. **Primary sources?** Do you want oral histories, archival documents, or primarily published literature?

---

*This is a starting point. Let me know what threads to pull on next.*
