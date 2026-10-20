"""Build the TD-One final presentation deck."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Colors
DARK = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT = RGBColor(0x00, 0x96, 0x88)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_BG = RGBColor(0xF5, 0xF5, 0xF5)


def add_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def title_slide():
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_bg(slide, DARK)
    txBox = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(11.3), Inches(2))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "TD-One: Regime-Routed Forecasting"
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    txBox2 = slide.shapes.add_textbox(Inches(1), Inches(3.8), Inches(11.3), Inches(1))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Classical statistics where they win \u2022 LLMs where they help \u2022 Honest negatives throughout"
    p2.font.size = Pt(24)
    p2.font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)
    p2.alignment = PP_ALIGN.CENTER

    txBox3 = slide.shapes.add_textbox(Inches(1), Inches(5.5), Inches(11.3), Inches(1))
    tf3 = txBox3.text_frame
    tf3.word_wrap = True
    p3 = tf3.paragraphs[0]
    p3.text = "Vandad Davoodnia \u2022 Vector Institute Agentic Forecasting Bootcamp \u2022 October 2026"
    p3.font.size = Pt(18)
    p3.font.color.rgb = GRAY
    p3.alignment = PP_ALIGN.CENTER


def section_slide(title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = DARK

    # accent bar
    shape = slide.shapes.add_shape(1, Inches(0.8), Inches(1.3), Inches(1.5), Inches(0.06))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT
    shape.line.fill.background()

    txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(1.7), Inches(11.7), Inches(5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = b
        p.font.size = Pt(22)
        p.font.color.rgb = DARK
        p.space_after = Pt(14)
        p.level = 0


def table_slide(title, headers, rows):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11.7), Inches(0.9))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = DARK

    n_rows = len(rows) + 1
    n_cols = len(headers)
    left = Inches(0.8)
    top = Inches(1.4)
    width = Inches(11.7)
    height = Inches(5.2)

    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = table_shape.table

    # header
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(16)
            paragraph.font.bold = True
            paragraph.font.color.rgb = WHITE
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = str(val)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(15)
                paragraph.font.color.rgb = DARK
            if i % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_BG


# --- Build ---
title_slide()

section_slide("The Method", [
    "Regime router: detects series regime (seasonal / trending / noisy) and routes among classical experts",
    "Classical experts: seasonal naive, ARIMA, ETS \u2014 with trust gates and conformal calibration",
    "Gated LLM expert (v2): Gemini Flash fires only on smooth macro series (+45% on Food CPI, do-no-harm neutral on SMM p=0.69, blocked on seasonal where LLMs hurt 36\u201364%)",
    "The gate is hand-written deterministic rules (not learned) \u2014 it knows when to use the LLM and when to stay classical",
])

table_slide("Results: Our Method vs Everything",
    ["Benchmark", "Ours (v2)", "Ours (v1)", "Best classical", "Vector LLM", "Literature"],
    [
        ["M4 Monthly (norm. CRPS \u2193)", "0.1451", "0.1451", "0.1505", "\u2014", "\u2014"],
        ["Tourism Monthly (norm. CRPS \u2193)", "0.1282", "0.1282", "0.1387", "\u2014", "1.409 MASE (DeepAR)*"],
        ["M3 Monthly (norm. CRPS \u2193)", "0.1476", "0.1476", "0.1630", "\u2014", "\u2014"],
        ["Food CPI (CRPS \u2193)", "0.5322\u2020", "0.9661", "0.9661", "0.5688", "\u2014"],
        ["Freddie Mac SMM (CRPS \u2193)", "0.000508", "0.000512", "0.000566 (ETS)", "\u2014", "\u2014"],
        ["BoC decisions (Acc. \u2191)", "72.2%", "72.2%", "69.4% (clim.)", "\u2014", "\u2014"],
    ])

# Footnote slide for TSFM + Food CPI caveats
section_slide("Results: Footnotes", [
    "* Tourism: TimesFM-2.5 gets 0.0965 CRPS zero-shot (25% better than our 0.1282) \u2014 but that's a 880MB foundation model vs our CPU-classical router. When added to our expert pool, the router ties it (p=0.42). Same pattern as Food CPI: the router can't add lift over a dominant expert.",
    "\u2020 Food CPI: three numbers, three stories \u2014 v2 generic 0.5322 (matches/edges Vector's 0.5688 directionally, likely n.s. on N=59); prompt-V1 hybrid 0.4526 (beats Vector by 20%, p=0.009); prompt-V1 alone 0.4262 (beats both hybrids). The prompt does the work; the router is a variance tax for a unified system.",
    "BoC: 72.2% is the classical rate_momentum rule (corrected 2026-10-19 from 83.7% \u2014 old number inflated by stale post-2023 data). v2 doesn't change the discrete path.",
])

section_slide("LLMs: Specialists, Not Generalists", [
    "Food CPI: prompt-V1 LLM alone (0.4262) wins decisively \u2014 the prompt does 100% of the work",
    "v2 generic (0.5322) matches/edges Vector's LLM (0.5688) directionally \u2014 but the router degrades 0.4262 \u2192 0.5322",
    "Tourism: LLM 64% WORSE than classical \u2022 M3: 36% worse \u2022 SMM: neutral (p=0.69)",
    "LLM as combiner: significantly worse than statistical router (p=1.7e-08)",
    "Verdict: LLMs help inside a rigid statistical scaffold, on smooth macro series only",
])

section_slide("Honest Negatives", [
    "News: fires on 90\u2013100% of origins, changes nothing \u2014 track closed on three datasets",
    "Learned combination weights: don't beat equal weights (forecast combination puzzle)",
    "Learned regimes: hand-written rules win (p=0.00046)",
    "LLM as calibrator: asking it to fix a classical grid destroys its skill",
    "High reasoning effort: 3.5\u20136\u00d7 slower, sometimes harmful \u2014 default to low",
    "Every negative published with the same rigor as positives",
])

section_slide("Takeaways", [
    "Regime-routed combination beats equal weights and best-single across 3 major benchmarks",
    "Structure beats flexibility: constrained priors win at small-n",
    "LLMs are a specialist tool \u2014 the gate knows when to call them",
    "Foundation models (TimesFM-2.5, Chronos-2) give ~30% zero-shot gains on Tourism",
    "All results PIT-safe, significance-tested, reproducible \u2014 code in repo",
])

# closing
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, DARK)
txBox = slide.shapes.add_textbox(Inches(1), Inches(3), Inches(11.3), Inches(1.5))
tf = txBox.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Thank you"
p.font.size = Pt(48)
p.font.bold = True
p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER
txBox2 = slide.shapes.add_textbox(Inches(1), Inches(4.5), Inches(11.3), Inches(1))
tf2 = txBox2.text_frame
tf2.word_wrap = True
p2 = tf2.paragraphs[0]
p2.text = "github.com/VDavoodnia/agentic-forecasting"
p2.font.size = Pt(20)
p2.font.color.rgb = GRAY
p2.alignment = PP_ALIGN.CENTER

out = "/home/hatch/workspace/TD-One_Final_Presentation.pptx"
prs.save(out)
print(f"Saved to {out}")
