# Reference verification notes — PVI audit repair (21 Oct 2026)

Method: every entry below was confirmed against a real source actually opened
via browser search/open (publisher page, DBLP, PMLR, Crossref publisher
metadata, journal site, JSTOR scan, or the paper/book itself). No entry rests
on memory alone. Result: **19 of 19 citation targets VERIFIED; 0 UNVERIFIED.**
One entry (Patrini 2017) carries a genuine pagination conflict in the wild —
resolved below in favour of the publisher's own metadata.

---

## New / replacement entries

### 1. Dawid & Skene 1979 — VERIFIED
A. P. Dawid & A. M. Skene, "Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm", *Journal of the Royal Statistical
Society, Series C (Applied Statistics)* 28(1):20–28, 1979. DOI 10.2307/2346806.
- RePEc suggested citation: https://ideas.repec.org/a/bla/jorssc/v28y1979i1p20-28.html
- Corroborated by reference lists in https://arxiv.org/pdf/1505.02463 and
  https://arxiv.org/pdf/2101.06171 ("Applied statistics, 20–28, 1979").

### 2. Whitehill et al. 2009 (GLAD) — VERIFIED
Jacob Whitehill, Paul Ruvolo, Tingfan Wu, Jacob Bergsma, Javier R. Movellan,
"Whose Vote Should Count More: Optimal Integration of Labels from Labelers of
Unknown Expertise", NeurIPS 22, pp. 2035–2043, 2009. (Author order is as
printed on the paper — Ruvolo second.)
- Official proceedings page: https://proceedings.neurips.cc/paper/2009/hash/f899139df5e1059396431415e770c6dd-Abstract.html
- Pages from the reference list of Raykar et al.'s JMLR paper (ACM DL):
  http://dl.acm.org/doi/10.5555/1756006.1859894 and a survey bibliography
  (ResearchGate "Learning from Disagreement"), both 2035–2043.

### 3. Raykar et al. 2010 — VERIFIED
Vikas C. Raykar, Shipeng Yu, Linda H. Zhao, Gerardo Hermosillo Valadez,
Charles Florin, Luca Bogoni, Linda Moy, "Learning From Crowds", *JMLR*
11:1297–1322, 2010.
- JMLR volume 11 listing / first author's own UMD research statement, plus
  multiple independent bibliographies, all 1297–1322.

### 4. Raykar & Yu 2012 — VERIFIED (task brief's title was wrong)
Vikas C. Raykar & Shipeng Yu, "Eliminating Spammers and Ranking Annotators
**for Crowdsourced Labeling Tasks**", *JMLR* 13:491–518, 2012.
- JMLR's own PDF: https://www.jmlr.org/papers/volume13/raykar12a/raykar12a.pdf
- Corroborated: https://arxiv.org/pdf/1912.09528 ref. [18]
  ("Journal of Machine Learning Research, 13(Feb):491–518, 2012").
- ⚠️ The brief's candidate title "…on Crowdsourcing Websites" does not exist;
  use "…for Crowdsourced Labeling Tasks".

### 5. Kim & Ghahramani 2012 (IBCC) — VERIFIED (audit was too pessimistic)
Hyun-Chul Kim & Zoubin Ghahramani, "Bayesian Classifier Combination",
AISTATS 2012 (15th), *PMLR* 22:619–627.
- PMLR record as quoted verbatim in citing bibliographies:
  http://proceedings.mlr.press/v22/kim12.html — see
  https://arxiv.org/pdf/2101.06171 and https://arxiv.org/pdf/2110.04117
  ref. [15], both "vol. 22 … pp. 619–627".
- ⚠️ The audit listed these page numbers as unverifiable. They are findable
  and consistent across independent sources: **619–627**.

### 6. Zhao et al. 2012 — VERIFIED (issue number corrected: 5(6), not 5(7))
Bo Zhao, Benjamin I. P. Rubinstein, Jim Gemmell, Jiawei Han, "A Bayesian
Approach to Discovering Truth from Conflicting Sources for Data
Integration", *PVLDB* **5(6)**:550–561, 2012. arXiv:1203.0058. The paper's
abstract explicitly describes the two-error-type (false positive / false
negative) generative source-quality model — it is the paper the audit meant.
- Microsoft Research publication page:
  https://www.microsoft.com/research/publication/a-bayesian-approach-to-discovering-truth-from-conflicting-sources-for-data-integration/
- Jiawei Han's own Illinois page ("PVLDB 5(6):550-561, 2012"):
  https://siebelschool.illinois.edu/about/people/faculty/hanj
- Official VLDB 2012 PVLDB vol. 5 listing (paper appears under "Volume 5,
  No. 6, February 2012", pp. 550–561):
  https://vldb2012.org/general-information/pvldb-vol5/

### 7. Natarajan et al. 2013 — VERIFIED
Nagarajan Natarajan, Inderjit S. Dhillon, Pradeep K. Ravikumar, Ambuj Tewari,
"Learning with Noisy Labels", NeurIPS 26, pp. 1196–1204, 2013.
- UT Austin CBDA publication page; corroborated by the bibliography in
  https://ar5iv.labs.arxiv.org/html/2006.13554 ("In NeurIPS, pp. 1196–1204").

### 8. Patrini et al. 2017 — VERIFIED, WITH A PAGINATION CONFLICT (read this)
Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock,
Lizhen Qu, "Making Deep Neural Networks Robust to Label Noise: A Loss
Correction Approach", CVPR 2017, DOI 10.1109/CVPR.2017.240.
- **Publisher side (IEEE via Crossref): first page 2233 → pp. 2233–2241.**
  Crossref record marked "doi-asserted-by: publisher" with the paper's DOI:
  https://api.crossref.org/works/10.1162%2FTACL_A_00574 (reference bib45);
  same range in https://api.crossref.org/works/10.1145%2F3447548.3467222
  and in *Frontiers of Computer Science*'s bibliography
  (https://academic.hep.com.cn/fcs/EN/10.1007/s11704-021-1208-9).
- **Secondary side: pp. 1944–1952** — the value the audit prints — appears in
  many author-side ML bibliographies, e.g.
  https://ar5iv.labs.arxiv.org/abs/2108.04063.
- Recommendation: print **2233–2241** with the DOI (publisher's pagination);
  the .bib in this folder does so. If the venue's copy-editors follow the
  secondary convention, 1944–1952 is the alternative in circulation. Either
  way, the audit's 1944–1952 should not be presented as the only value.

### 9. Yin et al. 2018 — VERIFIED (replaces the fabricated yin2018)
Dong Yin, Yudong Chen, **Kannan Ramchandran**, Peter L. Bartlett,
"Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates",
ICML 2018, *PMLR* 80:5650–5659.
- PMLR's own PDF (byline reads "Kannan Ramchandran"):
  http://proceedings.mlr.press/v80/yin18a/yin18a.pdf
- arXiv:1803.01498 (https://arxiv.org/pdf/1803.01498), same byline;
  Bartlett's Berkeley publications page carries the same full citation.
- ⚠️ Author-name trap: the third author is **Kannan Ramchandran** (surname
  Ramchandran). The audit's shorthand "Yin, Chen, Kannan & Bartlett" and the
  task brief repeat the common inversion; some bibliographies even print
  "Ramchandran Kannan". Also note one secondary source cites pp. 5636–5645 —
  wrong; PMLR's official range is 5650–5659.

### 10. Lewis 1969 — VERIFIED
David K. Lewis, *Convention: A Philosophical Study*, Harvard University
Press, 1969. (Book — no page range applies.)
- Bibliography in https://en.wikipedia.org/wiki/David_Lewis_(philosopher)
  ("*Convention: A Philosophical Study*, Harvard University Press 1969"),
  corroborated by multiple reference works.

### 11. Skyrms 1996 — VERIFIED
Brian Skyrms, *Evolution of the Social Contract*, Cambridge University
Press, 1996. ISBN 0-521-55471-3 (hardback) / 0-521-55583-3 (paperback).
- The book's own title/copyright pages (© Cambridge University Press 1996,
  "First published 1996", ISBNs printed):
  http://faculty.washington.edu/conormw/Teaching/Files/Models/Summer_School/Recommended_Readings/Summer_School_Further_Readings/Skyrms-Evolution_of_the_Social_Contract.pdf
- PhilPapers record: "Brian Skyrms — 1996 — New York: Cambridge University
  Press."
- This closes the audit's "Skyrms publisher details unverified" gap.

### 12. Crawford & Sobel 1982 — VERIFIED
Vincent P. Crawford & Joel Sobel, "Strategic Information Transmission",
*Econometrica* 50(6):1431–1451, 1982. DOI 10.2307/1913390.
- JSTOR scan header ("Source: Econometrica, Vol. 50, No. 6 (Nov., 1982),
  pp. 1431-1451"): https://www.edegan.com/pdfs/Crawford Sobel (1982) - Strategic Information Transmission.pdf
- RePEc bibliography, same volume/pages:
  https://ideas.repec.Org/p/nwu/cmsems/1245.html

### 13. FIRE — Huynh, Jennings & Shadbolt 2006 — VERIFIED
Trung Dong Huynh, Nicholas R. Jennings, Nigel R. Shadbolt, "An Integrated
Trust and Reputation Model for Open Multi-Agent Systems", *Autonomous
Agents and Multi-Agent Systems* 13(2):119–154, 2006.
DOI 10.1007/s10458-005-6825-4.
- The journal article itself (header: "Auton Agent Multi-Agent Sys (2006)
  13: 119–154"; abstract names FIRE):
  https://eprints.soton.ac.uk/262593/1/jaamas-dong.pdf
- Note: the canonical FIRE citation is this **journal** article, not an
  AAMAS 2006 conference paper (the brief's guess).

### 14. REGRET — Sabater & Sierra 2002 — VERIFIED
Jordi Sabater & Carles Sierra, "Reputation and Social Network Analysis in
Multi-Agent Systems", AAMAS '02, pp. 475–482, ACM, 2002.
DOI 10.1145/544741.544854.
- Crossref's own record for the DOI (page "475-482", authors, venue):
  https://api.crossref.org/works/10.1145%2F544741.544854
- Sabater's thesis publication list gives the same citation
  (AAMAS-02, Bologna, pp. 475–482). The REGRET model was first sketched in
  Sabater & Sierra, "REGRET: A Reputation Model for Gregarious Societies",
  Autonomous Agents 2001, pp. 194–195 — cite the 2002 paper as canonical.

### 15. BLADE — Regan, Poupart & Cohen 2006 — VERIFIED (identified)
Kevin Regan, Pascal Poupart, Robin Cohen, "Bayesian Reputation Modeling in
E-Marketplaces Sensitive to Subjectivity, Deception and Change",
*Proceedings of the 21st National Conference on Artificial Intelligence
(AAAI-06)*, AAAI Press, pp. 1206–1212, 2006.
- The paper itself, hosted by AAAI (paper AAAI06-189; its abstract says
  "Our model, called BLADE"): https://cdn.aaai.org/AAAI/2006/AAAI06-189.pdf
- AAAI technical-papers archive lists it at p. 1206:
  https://aaai.org/proceeding/technical-papers/page/4/
- Full range 1206–1212 in multiple citing bibliographies (e.g. the reference
  list reproduced at https://www.scitepress.org/publishedPapers/2023/116904/pdf/index.html)
  and Crossref first-page 1206 (via https://api.crossref.org/works/10.1145%2F2631775.2631821).
- (Not to be confused with the unrelated 2024 "BLADE" LM-agent benchmark.)

---

## Page completions for entries already in the bib (audit item 16)

### 16a. Blanchard et al. 2017 — VERIFIED: pp. 119–129
Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, Julien Stainer,
"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent",
NeurIPS 30, 2017. Audit's value (119–129) is **correct**.
- Authors' own lab page (EPFL DCL): "NIPS 2017: 119-129" —
  https://dcl.epfl.ch/site/safeai
- Venue page: https://nips.cc/virtual/2017/poster/8810

### 16b. Dong et al. 2009 — VERIFIED: PVLDB 2(1):562–573
Xin Luna Dong, Laure Berti-Equille, Divesh Srivastava, "Truth Discovery and
Copying Detection in a Dynamic World", 2009. Audit's value is **correct**.
- DBLP (via VLDB's mirror):
  https://mail.vldb.org/dblp/db/indices/a-tree/s/Srivastava:Divesh.html
  ("Truth Discovery and Copying Detection in a Dynamic World. PVLDB 2(1):
  562-573 (2009)"), corroborated by ACL Anthology and ACM DL citations.

### 16c. Grice 1975 — VERIFIED: pp. 41–58
H. Paul Grice, "Logic and Conversation", in P. Cole & J. L. Morgan (eds.),
*Syntax and Semantics, Vol. 3: Speech Acts*, Academic Press, 1975.
Audit's value is **correct**.
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2953817/ ("Syntax and semantics:
  Speech acts, 1975, pp. 41–58, Vol. 3"); multiple philosophy bibliographies
  agree (Academic Press, New York).

### 16d. Smit et al. 2024 — VERIFIED: PMLR 235:45883–45905
Andries Petrus Smit, Nathan Grinsztajn, Paul Duckworth, Thomas D. Barrett,
Arnu Pretorius, "Should We Be Going MAD? A Look at Multi-Agent Debate
Strategies for LLMs", ICML 2024. Audit's value is **correct** — taken from
PMLR's own page, which also supplies the official BibTeX used in the .bib.
- https://proceedings.mlr.press/v235/smit24a.html

---

## Where the audit's (or brief's) citation details were wrong

1. **Patrini et al. 2017 pages.** Audit: 1944–1952. Publisher (IEEE/Crossref,
   DOI 10.1109/CVPR.2017.240) metadata: **2233–2241**. Both circulate; the
   publisher's is recommended. See entry 8.
2. **Kim & Ghahramani 2012 "pages unverifiable".** They verify fine:
   PMLR 22:619–627. See entry 5.
3. **Yin et al. 2018 author name.** The third author is Kannan **Ramchandran**,
   not "Ramchandran Kannan" / surname "Kannan". The audit's replacement
   citation is otherwise right (ICML 2018, PMLR 80:5650–5659). See entry 9.
4. **Raykar & Yu 2012 title** (task brief, not the audit): "…on Crowdsourcing
   Websites" is wrong; the JMLR title is "…for Crowdsourced Labeling Tasks".
   See entry 4.
5. **Zhao et al. 2012 issue** (a detail circulating in our working notes):
   PVLDB **5(6)**, not 5(7) — per the senior author's own page and the
   official VLDB 2012 volume listing. Pages 550–561 stand. See entry 6.
6. **FIRE venue guess** (task brief): canonical citation is the JAAMAS
   **journal** article (2006), not an AAMAS conference paper. See entry 13.

Everything else the audit printed for these entries (Dawid & Skene 28(1):
20–28; GLAD 2035–2043; Raykar 2010 1297–1322; Raykar & Yu 491–518; Natarajan
1196–1204; Yin PMLR 80:5650–5659; Blanchard 119–129; Dong 562–573; Grice
41–58; Smit 45883–45905) checked out exactly.
