"""Build the TD-One Phase-2 update deck (exp63 -> present, mortgage framing)."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

DARK = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT = RGBColor(0x00, 0x96, 0x88)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT = RGBColor(0xF5, 0xF5, 0xF5)


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def box(slide, x, y, w, h, text, size=20, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.alignment = align
    return tb


def bullets(slide, x, y, w, h, items, size=19):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (txt, sub) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " + txt) if not sub else ("    " + txt)
        p.font.size = Pt(size if not sub else size - 3)
        p.font.color.rgb = DARK if not sub else GRAY
        p.space_after = Pt(10)


def content(title, items, note=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg(slide, LIGHT)
    box(slide, 0.8, 0.5, 11.7, 1.0, title, size=34, bold=True)
    bullets(slide, 0.8, 1.7, 11.7, 4.6, items)
    if note:
        box(slide, 0.8, 6.5, 11.7, 0.6, note, size=14, color=GRAY)


# 1. Title
s = prs.slides.add_slide(prs.slide_layouts[6])
bg(s, DARK)
box(s, 1, 2.0, 11.3, 1.6, "Agentic Solutions for Mortgage Attrition", size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
box(s, 1, 3.6, 11.3, 1.0, "Phase 2 update — from benchmark forecasting to a loan-book decision system", size=22, color=RGBColor(0xAA, 0xAA, 0xAA), align=PP_ALIGN.CENTER)
box(s, 1, 5.6, 11.3, 0.8, "Results marked ✓ are final and verified. Slides marked PENDING are experiments in flight.", size=16, color=RGBColor(0x88, 0x88, 0x88), align=PP_ALIGN.CENTER)

# 2. Where we are
content("Where the program stands", [
    ("Proposed Method v2 is the forecasting core: classical regime router + a gated LLM expert", False),
    ("Around it, four new layers were built this phase:", False),
    ("Explainability layer — borrower groups with citation-grounded risk narratives", True),
    ("Safety layer — a second decision stage that blocks the router when validation says so", True),
    ("Uncertainty layer — calibrated prediction intervals instead of overconfident ones", True),
    ("Signal layer — monthly macro signals read from news by a dedicated LLM", True),
    ("Everything is evaluated on mortgage-relevant data: real Canadian bank balances, Freddie Mac attrition, and synthetic loan books with planted shocks", False),
])

# 3. v2 headline
content("✓ Headline result: the gated LLM expert", [
    ("On Canadian Food CPI (a hard, smooth macro series), Proposed Method v2 beats v1 by 45%: CRPS 0.53 vs 0.97 (p = 4.3e-06)", False),
    ("The win comes from one LLM expert with a domain prompt — switched ON only where validation says the LLM helps", False),
    ("On Tourism, M3 and M4 the gate stays OFF and v2 is identical to v1 — the gate is the contribution, not the LLM alone", False),
    ("On mortgage attrition (SMM), the LLM expert is neutral (p = 0.69) — reported as such", False),
], note="Takeaway: LLMs help inside a rigid statistical scaffold, on the series where semantics matter — and we can prove where those are.")

# 4. Real Canadian data
content("✓ Real loan-book data: OSFI M4 balances", [
    ("121 months (2016–2026) of real Canadian bank mortgage balances from StatCan/OSFI, 4 segments: insured, uninsured, variable insured, variable uninsured", False),
    ("Implied portfolio runoff averages $9.8B/month insured and $23.1B/month uninsured", False),
    ("Honest null result: on aggregate runoff, our method ties a naive forecast — the runoff rate is near-unforecastable at this level", False),
    ("Value: this is now the real-data null benchmark every future runoff claim must beat", False),
], note="Data assets added this phase: BoC posted rates 1975–2026 (622 months), broker discount rates (54 months, Wayback archives).")

# 5. Red team
content("✓ Stress-testing the loan book: 7 planted shocks", [
    ("We inject distributional shifts into loan-portfolio series and measure how forecasters break", False),
    ("Rate shock is the universal killer: at 4σ, forecast error roughly triples for every model; Chronos-2 interval coverage collapses 0.89 → 0.53", False),
    ("Policy interventions and regional divergence also break all models at high severity", False),
    ("Seasonal breaks and correlation breakdowns never break univariate models — the blind spots are specific, not general", False),
], note="Mortgage translation: the scenario that matters most for a mortgage book — a fast rate cycle — is exactly the one where every forecaster's confidence is least trustworthy.")

# 6. Layer-2
content("✓ Layer-2: a second decision stage that stops the harm", [
    ("Diagnosis from the OSFI benchmark: the router sometimes picks bad experts (uninsured segment: +10.7% worse than naive, p = 0.030) and has no channel for outside information", False),
    ("Layer-2 fix: an analyst toolbox (diagnostics, backtests, lock-in meter, changepoint scan) + validation-gated decisions with an abstain-to-naive rule", False),
    ("Result: the uninsured harm is mechanically blocked (+0.8%, n.s.); at horizon 3 the gated system beats v2 significantly (−6.8%, p = 0.015)", False),
    ("Confirmed again in the integration test: Layer-2 also protects the loan-group forecaster under planted shocks (p < 1e-8)", False),
    ("Every decision emits an auditable trace — a human can follow why each forecast was trusted, overridden, or abandoned", False),
])

# 7. Uncertainty
content("✓ New direction that works: predict uncertainty, not corrections", [
    ("Three attempts to correct forecasts around shocks failed — timing precision is the binding constraint", False),
    ("Pivot: predict how uncertain the forecast is, and calibrate the intervals instead", False),
    ("Conformalized quantile regression (CQR) is the new default uncertainty layer: better coverage and better CRPS on Food CPI (p = 1.7e-04), never harmful", False),
    ("News-story features genuinely predict uncertainty: story-augmented model beats numeric-only (p = 2.0e-11) — the narrative tells you when not to trust the numbers", False),
], note="PENDING extension: CQR × story fusion and a sweep across Tourism / M3 / M4 / OSFI / SMM benchmarks.")

# 8. Explainability
content("✓ Explainability: borrower groups a manager can read", [
    ("Deterministic grouping (fixed seed, tree-extracted rules) — reproducible groups like 'variable rate and household income ≤ $44,000'", False),
    ("LLM writes the risk narrative for each group, but every factual sentence must cite a statistic or a source — 10/10 pass after the contract was made total", False),
    ("Cost is quantified: grouping costs +0.009 to +0.016 CRPS vs one global forecast — it is an explainability layer, not a forecasting improvement", False),
    ("SHAP-based clustering doubles recovery of the simulator's planted at-risk cohorts vs raw features (ARI 0.091 vs 0.046)", False),
])

# 9. Signal service (pending)
content("✓+⏳ The Macro Signal Service — calibration landed", [
    ("A dedicated LLM reads one month of news per category and emits a calibrated signal: direction (−2..+2), risk level, certainty", False),
    ("22 categories × 36 months (725 signals); 7 of 9 anchored categories LIVE — FX (holdout 0.79), food (holdout 0.82), labour, inflation, housing, services, wars", False),
    ("SMM integration failed honestly: every origin in the signal window is a 'calm' regime — the router is degenerate, so no signal could express itself there", False),
    ("The open test: Food CPI integration (exp77) — its origins span non-degenerate regimes, and this is where calibration-first news finally faces a real router", False),
], note="Design principle: the LLM classifies and summarizes; statistics decide whether the signal is real. Certainty does not calibrate — honest wrinkle recorded.")

# 10. Mini agents (pending)
content("PENDING: mini-agent decision engine", [
    ("One mini-agent per macro factor (food, rent, house prices, immigration, politics, war, climate…) — each reports only up / down / flat + confidence + one reason", False),
    ("A separate diagnostics family reports on the series itself: stability, seasonality, expert disagreement", False),
    ("An orchestrator consumes ~25 narrow signals and makes the decision — forecast routing or interval width — it never forecasts itself", False),
    ("Being tested on: OSFI runoff, Food CPI, SMM attrition, planted-shift scenarios, and synthetic loan books", False),
])

# 11. Honest negatives
content("✓ What did not work (kept in the record)", [
    ("Anticipatory LLM overlays: failed on synthetic AND real shocks — even a perfect-information oracle cannot beat the timing noise", False),
    ("News as direct forecast context: no measurable gain in three separate tests", False),
    ("Competitor-rate spread as an aggregate covariate: strong in-sample (r = −0.85), collapses out-of-sample — a regime artifact, settled with 54 months of real broker data", False),
    ("Uplift modelling and LLM-driven grouping for cohort recovery: chance-level", False),
    ("Counterfactual risk tool as classifier input: the single-seed win did not survive 5 seeds — killed with a shuffled-control proof", True),
    ("Borrower groups do NOT isolate regional shocks — grouping doesn't shield forecasts from a regional downturn (oracle geography identical)", True),
], note="Each negative ships with a diagnosis of the mechanism that failed — that is what pointed us at uncertainty prediction and Layer-2 gating.")

# 12. Roadmap
content("What's next", [
    ("Finish the signal service calibration; wire only live categories into the stack", False),
    ("Fuse the story signal into the CQR uncertainty layer; sweep all benchmarks", False),
    ("Complete the mini-agent orchestrator evaluation across the five test beds", False),
    ("Integration: planted shifts stress-testing the group forecaster, protected by Layer-2", False),
    ("Loan-level next step: Freddie Mac loan-level data (SFLLD) for cross-sectional attrition modelling", False),
])

prs.save("/home/hatch/workspace/TD-One_Phase2_Update.pptx")
print("saved", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
