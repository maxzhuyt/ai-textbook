# AI Historiography: Data Collection Summary
*Session 1 | February 3, 2026*

## What Was Accomplished

### Data Collected

| Source | Records | Time Period | Notes |
|--------|---------|-------------|-------|
| NSF "artificial intelligence" | 8,373 | 1978-2026 | Deduplicated |
| NSF other AI terms | ~2,000 | 1983-2026 | Expert systems, neural networks, etc. |
| DARPA AI papers | 7,584 | 1974-2025 | Via OpenAlex |
| ARPA AI papers | 2,399 | 1962-2025 | Pre-DARPA funding |
| ONR AI papers | 6,618 | 1951-2025 | Navy funding |

**Total: ~26,000 unique records, 231MB**

### Key Findings

1. **AI Winters Are Visible in Terminology, Not Volume**
   - Research continued during "winters" but under different names
   - "Expert systems" peaked 1985-1992, then crashed
   - "Connectionist" briefly popular 1987-1993
   - "AI" term avoided until late 2010s

2. **Funding Trajectory**
   - 1970s: $0.77M (7 awards)
   - 1980s: $17.33M (88 awards)
   - 1990s: $122.05M (493 awards)
   - 2000s: $65.72M (256 awards) ← **possible AI winter effect**
   - 2010s: $842.52M (1,585 awards)
   - 2020s: $6,703M (6,815 awards)

3. **Institutional Leadership Shifted**
   - 1980s-1990s: Stanford, MIT dominated
   - 2000s-2010s: CMU became leader
   - 2020s: Purdue, Georgia Tech, Penn State rise

4. **DARPA Was Crucial But Hard to Track**
   - Most cited DARPA-AI paper: XGBoost (43K citations)
   - CMU leads DARPA-funded AI (1,469 papers)
   - Historical DARPA influence underestimated in data

### Methodology Documented

- Full research log: `RESEARCH_LOG.md`
- Data README: `data/README.md`
- All raw JSON files preserved for reproducibility

### OpenAlex vs Web of Science

**Recommendation: Use OpenAlex**
- Free, larger coverage (240M vs 90M works)
- Funder tracking (DARPA, NSF, ONR)
- Citation networks accessible
- Recent research confirms it's "superset of Scopus"

### Next Steps Suggested

1. Cross-reference NSF PIs with OpenAlex authors
2. Build citation network from DARPA papers
3. Track individual researchers across terminology shifts
4. Validate with Web of Science if available
5. Look at international funding (EU, Japan, China)

---

*Files are in `/research/data/`. Ready for next iteration.*
