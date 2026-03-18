#!/usr/bin/env python3
"""生成 TextMamba3D 过拟合分析与改进计划 PDF 报告"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import datetime

# ── 字体注册 ──
FONT_PATHS = [
    ("/System/Library/Fonts/STHeiti Medium.ttc", "STHeiti"),
    ("/System/Library/Fonts/PingFang.ttc", "PingFang"),
    ("/System/Library/Fonts/Supplemental/Songti.ttc", "Songti"),
]

CN_FONT = "Helvetica"
CN_FONT_BOLD = "Helvetica-Bold"

for path, name in FONT_PATHS:
    if os.path.exists(path):
        try:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=0))
            CN_FONT = name
            CN_FONT_BOLD = name
            break
        except Exception:
            continue

# ── 颜色方案 ──
PRIMARY = HexColor("#1a1a2e")
ACCENT = HexColor("#16213e")
BLUE = HexColor("#0f3460")
HIGHLIGHT = HexColor("#e94560")
LIGHT_BG = HexColor("#f5f5f5")
WHITE = white
TABLE_HEADER_BG = HexColor("#1a1a2e")
TABLE_ALT_BG = HexColor("#f0f0f5")

# ── 样式 ──
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    "CNTitle", parent=styles["Title"],
    fontName=CN_FONT_BOLD, fontSize=22, leading=28,
    textColor=PRIMARY, alignment=TA_CENTER, spaceAfter=6*mm,
)
style_subtitle = ParagraphStyle(
    "CNSubtitle", parent=styles["Normal"],
    fontName=CN_FONT, fontSize=11, leading=14,
    textColor=HexColor("#666666"), alignment=TA_CENTER, spaceAfter=10*mm,
)
style_h1 = ParagraphStyle(
    "CNH1", parent=styles["Heading1"],
    fontName=CN_FONT_BOLD, fontSize=16, leading=22,
    textColor=PRIMARY, spaceBefore=8*mm, spaceAfter=4*mm,
    borderWidth=0, borderPadding=0,
)
style_h2 = ParagraphStyle(
    "CNH2", parent=styles["Heading2"],
    fontName=CN_FONT_BOLD, fontSize=13, leading=18,
    textColor=BLUE, spaceBefore=5*mm, spaceAfter=3*mm,
)
style_h3 = ParagraphStyle(
    "CNH3", parent=styles["Heading3"],
    fontName=CN_FONT_BOLD, fontSize=11, leading=15,
    textColor=ACCENT, spaceBefore=3*mm, spaceAfter=2*mm,
)
style_body = ParagraphStyle(
    "CNBody", parent=styles["Normal"],
    fontName=CN_FONT, fontSize=10, leading=15,
    textColor=black, alignment=TA_JUSTIFY, spaceAfter=2*mm,
)
style_bullet = ParagraphStyle(
    "CNBullet", parent=style_body,
    leftIndent=8*mm, bulletIndent=3*mm,
    spaceBefore=1*mm, spaceAfter=1*mm,
)
style_code = ParagraphStyle(
    "CNCode", parent=styles["Code"],
    fontName="Courier", fontSize=9, leading=12,
    backColor=LIGHT_BG, borderWidth=0.5, borderColor=HexColor("#dddddd"),
    borderPadding=4, spaceAfter=3*mm,
)
style_note = ParagraphStyle(
    "CNNote", parent=style_body,
    fontSize=9, leading=13, textColor=HexColor("#555555"),
    leftIndent=5*mm, rightIndent=5*mm,
    backColor=HexColor("#fff8e1"), borderWidth=0.5,
    borderColor=HexColor("#ffcc02"), borderPadding=6,
    spaceBefore=2*mm, spaceAfter=3*mm,
)


def make_table(headers, rows, col_widths=None):
    """创建带样式的表格"""
    header_row = [Paragraph(f'<b>{h}</b>', ParagraphStyle(
        "TH", fontName=CN_FONT_BOLD, fontSize=9, leading=12, textColor=WHITE,
        alignment=TA_CENTER,
    )) for h in headers]

    data_rows = []
    for row in rows:
        data_rows.append([
            Paragraph(str(cell), ParagraphStyle(
                "TD", fontName=CN_FONT, fontSize=9, leading=12,
                alignment=TA_LEFT,
            )) for cell in row
        ])

    table_data = [header_row] + data_rows

    if col_widths is None:
        available = 170 * mm
        col_widths = [available / len(headers)] * len(headers)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), CN_FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_BG))

    t.setStyle(TableStyle(style_cmds))
    return t


def build_pdf():
    output_path = os.path.join(os.path.dirname(__file__), "TextMamba3D_Analysis_Report.pdf")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    story = []
    W = 170 * mm  # available width

    # ════════════════════════════════════════
    # 封面
    # ════════════════════════════════════════
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph("TextMamba3D", style_title))
    story.append(Paragraph(
        "Overfitting Analysis &amp; Improvement Plan",
        ParagraphStyle("EN_Sub", fontName="Helvetica", fontSize=14,
                        leading=18, textColor=BLUE, alignment=TA_CENTER, spaceAfter=4*mm)
    ))
    story.append(Paragraph(
        f"Lei Yuxuan | {datetime.date.today().strftime('%Y-%m-%d')}",
        style_subtitle
    ))

    # 分隔线
    story.append(Table(
        [[""]], colWidths=[W],
        style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.5, HIGHLIGHT)])
    ))
    story.append(Spacer(1, 10*mm))

    # 摘要
    story.append(Paragraph(
        "TextMamba3D is a text-guided 3D brain tumor segmentation framework based on "
        "a unified Mamba (State Space Model) architecture. This report systematically analyzes "
        "the overfitting risks identified during training and proposes a three-phase improvement "
        "plan covering experimental design, regularization, and model slimming.",
        style_body
    ))
    story.append(Spacer(1, 5*mm))

    # 关键数字
    story.append(make_table(
        ["Metric", "TextMamba3D", "SegMamba", "TextBraTS Paper"],
        [
            ["Total Params", "~180M", "~23M", "N/A"],
            ["Trainable Params", "~70M", "~23M", "N/A"],
            ["Training Samples", "295", "~877", "220"],
            ["Param/Sample Ratio", "237K:1", "26K:1", "-"],
            ["Dataset", "TextBraTS 369", "BraTS2023 1251", "TextBraTS 369"],
            ["Current Best Dice", "0.637", "-", "0.853"],
        ],
        col_widths=[W*0.28, W*0.24, W*0.24, W*0.24]
    ))

    story.append(PageBreak())

    # ════════════════════════════════════════
    # 目录概览
    # ════════════════════════════════════════
    story.append(Paragraph("Contents", style_h1))
    toc_items = [
        "1. Core Problem Diagnosis",
        "    1.1 Overfitting Risk",
        "    1.2 Experimental Design Flaws",
        "    1.3 Insufficient Regularization",
        "    1.4 Hardware Limitations",
        "2. Performance Benchmarks",
        "3. Three-Phase Improvement Plan",
        "    3.1 Phase 1: Fix Experimental Design",
        "    3.2 Phase 2: Regularization Enhancement",
        "    3.3 Phase 3: Model Slimming",
        "4. GPU Requirements &amp; Selection",
        "5. Text Guidance Value Assessment",
        "6. Idea2Paper RAG System Overview",
        "7. Key Concepts Reference",
        "    7.1 LoRA (Low-Rank Adaptation)",
        "    7.2 Domain Shift",
        "    7.3 FiLM Modulation",
        "8. Action Checklist",
    ]
    for item in toc_items:
        indent = 8*mm if item.startswith("    ") else 0
        story.append(Paragraph(
            item.strip(),
            ParagraphStyle("TOC", fontName=CN_FONT, fontSize=10, leading=16,
                           leftIndent=indent, textColor=BLUE if indent == 0 else black)
        ))
    story.append(PageBreak())

    # ════════════════════════════════════════
    # 1. 核心问题诊断
    # ════════════════════════════════════════
    story.append(Paragraph("1. Core Problem Diagnosis", style_h1))

    # 1.1
    story.append(Paragraph("1.1 Overfitting Risk (Critical)", style_h2))
    story.append(Paragraph(
        "TextMamba3D uses 3x the trainable parameters to fit 1/3 of the data compared to SegMamba. "
        "The parameter-to-sample ratio is ~9x that of SegMamba, creating a structurally "
        "over-parameterized scenario.",
        style_body
    ))
    story.append(make_table(
        ["Dimension", "TextMamba3D", "SegMamba", "Ratio"],
        [
            ["Trainable Params", "~70M", "~23M", "3x"],
            ["Training Samples", "295", "~877", "1/3x"],
            ["Param/Sample", "237K:1", "26K:1", "9x"],
        ],
        col_widths=[W*0.3, W*0.23, W*0.23, W*0.24]
    ))
    story.append(Spacer(1, 3*mm))

    # Component breakdown
    story.append(Paragraph("Architecture Component Breakdown:", style_h3))
    story.append(make_table(
        ["Component", "Module", "Params", "Key Design"],
        [
            ["Modality Grouping", "PatchEmbed3D", "~2M", "T1+T1ce / T2+FLAIR"],
            ["Image Backbone", "CrossScanBiMamba3D x4", "~44M", "6-direction scan"],
            ["Text Encoder", "PubMedBERT + Mamba", "~110M (109.5M frozen)", "Last 2 layers unfrozen"],
            ["Fusion", "FiLM + CrossAttn + MambaFusion", "~7M", "Dual-path fusion"],
            ["Uncertainty Gating", "UncertaintyGating", "~1M", "Noise suppression"],
            ["Decoder", "MambaDecoder + DS Heads", "~16M", "Deep supervision"],
        ],
        col_widths=[W*0.2, W*0.28, W*0.24, W*0.28]
    ))
    story.append(Spacer(1, 3*mm))

    # 1.2
    story.append(Paragraph("1.2 Experimental Design Flaws", style_h2))
    story.append(Paragraph(
        "<b>Fatal Issue: val == test.</b> In <font face='Courier'>_split_cases()</font>, both "
        "<font face='Courier'>split='val'</font> and <font face='Courier'>split='test'</font> "
        "return the same 74 cases. Best checkpoint selection is based on these same cases, "
        "making all reported metrics optimistically biased.",
        style_body
    ))
    story.append(Paragraph(
        "The TextBraTS original paper uses a proper 220/55/94 (train/val/test) three-way split "
        "with 94 independent test cases. TextMamba3D has no independent test set and no "
        "cross-validation.",
        style_body
    ))
    story.append(Paragraph(
        "<font face='Courier'>if split == 'train': indices[:train_size]<br/>"
        "else:  # val or test -- returns same data!<br/>"
        "    indices[train_size:]</font>",
        style_code
    ))

    # 1.3
    story.append(Paragraph("1.3 Insufficient Regularization", style_h2))
    story.append(make_table(
        ["Technique", "Current Value", "Problem"],
        [
            ["Dropout", "0.0 (entire model)", "70M params with zero stochastic regularization"],
            ["Weight Decay", "1e-5", "Too small to be effective"],
            ["PubMedBERT", "Last 2 layers unfrozen (+14M)", "14M text params on 295 samples"],
            ["No-text Ratio", "0.15", "Could be higher for better regularization"],
        ],
        col_widths=[W*0.22, W*0.3, W*0.48]
    ))
    story.append(Paragraph(
        "Only existing regularization: data augmentation (7 types), deep supervision, "
        "gradient clipping (max_norm=1.0), and early stopping (patience=30).",
        style_body
    ))

    # 1.4
    story.append(Paragraph("1.4 Hardware Limitations (RTX 4060 8GB)", style_h2))
    story.append(make_table(
        ["Constraint", "Current", "Ideal", "Impact"],
        [
            ["Batch size", "1", "4~8", "Noisy gradients, unstable training"],
            ["Patch size", "64^3", "96^3~128^3", "Only ~3% of original volume (240x240x155)"],
            ["Contrastive loss", "Disabled", "batch>1 needed", "Designed but unused"],
            ["Training speed", "Very slow", "-", "Hyperparameter search infeasible"],
        ],
        col_widths=[W*0.2, W*0.15, W*0.2, W*0.45]
    ))
    story.append(Paragraph(
        "patch_size=64^3 is the most critical hardware constraint. Brain tumor segmentation "
        "requires global context (tumor location, ventricle relationship, edema extent). "
        "This partially explains the Dice plateau at 0.63 -- not just overfitting, but also "
        "underfitting due to insufficient spatial context.",
        style_note
    ))

    story.append(PageBreak())

    # ════════════════════════════════════════
    # 2. Performance Benchmarks
    # ════════════════════════════════════════
    story.append(Paragraph("2. Performance Benchmarks", style_h1))
    story.append(Paragraph("Same Dataset (TextBraTS 369 Cases)", style_h2))
    story.append(make_table(
        ["Method", "ET Dice", "WT Dice", "TC Dice", "Avg Dice"],
        [
            ["3D-UNet", "80.4%", "87.3%", "81.6%", "83.1%"],
            ["nnU-Net", "82.2%", "87.5%", "82.6%", "84.1%"],
            ["SegResNet", "80.9%", "88.4%", "82.3%", "83.8%"],
            ["SwinUNETR", "81.0%", "89.5%", "80.8%", "83.8%"],
            ["TextBraTS (paper)", "83.3%", "89.9%", "82.8%", "85.3%"],
            ["TextMamba3D (ep.11)", "-", "-", "-", "62.3%"],
            ["TextMamba3D (best ep.62)", "-", "-", "-", "63.7%"],
        ],
        col_widths=[W*0.28, W*0.18, W*0.18, W*0.18, W*0.18]
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "TextMamba3D uses 295 training samples (vs 220 in the original paper) but achieves "
        "significantly lower performance, confirming that overfitting + hardware constraints "
        "are actively harming generalization.",
        style_note
    ))

    story.append(Paragraph("Expected Dice After Improvements", style_h2))
    story.append(make_table(
        ["Scenario", "Expected Avg Dice", "Basis"],
        [
            ["Current (no changes)", "0.65~0.70", "Historical best 0.637, val=test bias"],
            ["Phase 2 (regularization)", "0.75~0.80", "Dropout + weight decay + freeze BERT"],
            ["Phase 2+3 (+ slimming + better GPU)", "0.80~0.85", "Approaching nnU-Net level"],
            ["Theoretical ceiling", "~0.85", "TextBraTS paper SOTA"],
        ],
        col_widths=[W*0.35, W*0.25, W*0.40]
    ))

    story.append(PageBreak())

    # ════════════════════════════════════════
    # 3. Three-Phase Improvement Plan
    # ════════════════════════════════════════
    story.append(Paragraph("3. Three-Phase Improvement Plan", style_h1))
    story.append(Paragraph(
        "Principle: Fix evaluation first, then regularization, then architecture. "
        "Never tune the model while the evaluation system is flawed.",
        style_note
    ))

    # Phase 1
    story.append(Paragraph("3.1 Phase 1: Fix Experimental Design", style_h2))
    story.append(Paragraph("Goal: Make evaluation trustworthy. No model code changes.", style_body))
    story.append(make_table(
        ["Task", "File", "Details"],
        [
            ["Separate val/test", "brats_textbrats_dataset.py", "Ref: original paper 220/55/94 split"],
            ["Or: implement 5-fold CV", "train.py + dataset", "369 cases fully utilized, report mean +/- std"],
            ["Add train/val loss curves", "train.py", "TensorBoard logging for overfitting detection"],
            ["Run baseline", "-", "Full training with current model as control"],
        ],
        col_widths=[W*0.25, W*0.3, W*0.45]
    ))

    # Phase 2
    story.append(Paragraph("3.2 Phase 2: Regularization Enhancement", style_h2))
    story.append(Paragraph("Goal: Reduce trainable params, add regularization. Expect Dice 0.75~0.80.", style_body))
    story.append(make_table(
        ["Change", "Current", "Target", "Effect"],
        [
            ["Dropout", "0.0", "0.1~0.2", "Stochastic regularization in Mamba blocks"],
            ["Weight Decay", "1e-5", "1e-2~5e-3", "Stronger L2 penalty"],
            ["Freeze PubMedBERT", "Last 2 layers unfrozen", "Fully frozen", "-14M trainable params"],
            ["Or: LoRA (r=8)", "14M unfrozen", "+0.3M LoRA", "97% fewer text params"],
            ["No-text ratio", "0.15", "0.25~0.30", "Stronger modality regularization"],
        ],
        col_widths=[W*0.22, W*0.22, W*0.22, W*0.34]
    ))
    story.append(Paragraph("Trainable params: ~70M -> ~56M (freeze BERT) or ~56.3M (LoRA).", style_body))

    # Phase 3
    story.append(Paragraph("3.3 Phase 3: Model Slimming (Only If Phase 2 Still Overfits)", style_h2))
    story.append(Paragraph("Goal: Match model capacity to data scale. Expect Dice 0.80~0.85.", style_body))
    story.append(make_table(
        ["Change", "Effect", "Risk"],
        [
            ["Base dim: 96 -> 48", "Image encoder ~44M -> ~11M", "May reduce expressiveness"],
            ["Depths: [2,2,2,2] -> [1,1,2,1]", "~50% param reduction", "Low risk, keep bottleneck depth=2"],
            ["Remove Cross-Attention", "Save ~3-4M", "Needs ablation validation"],
            ["Scan directions: 6 -> 3", "~50% less compute", "Aligns with SegMamba's design"],
        ],
        col_widths=[W*0.35, W*0.35, W*0.30]
    ))
    story.append(Paragraph("Trainable params: ~56M -> ~20M.", style_body))

    story.append(PageBreak())

    # ════════════════════════════════════════
    # 4. GPU Requirements
    # ════════════════════════════════════════
    story.append(Paragraph("4. GPU Requirements &amp; Selection", style_h1))

    story.append(Paragraph("VRAM Estimation (with AMP + Gradient Checkpointing)", style_h2))
    story.append(make_table(
        ["Phase", "Trainable", "Patch", "Batch", "Est. VRAM", "Min GPU"],
        [
            ["Phase 2 (dim=96)", "~56M", "64^3", "1", "~7 GB", "8 GB (4060)"],
            ["Phase 2 (dim=96)", "~56M", "96^3", "1", "~14 GB", "16 GB (T4)"],
            ["Phase 2 (dim=96)", "~56M", "96^3", "2", "~24 GB", "24 GB (3090)"],
            ["Phase 2+3 (dim=48)", "~20M", "96^3", "2", "~12 GB", "16 GB (T4)"],
            ["Phase 2+3 (dim=48)", "~20M", "96^3", "4", "~21 GB", "24 GB (3090)"],
            ["Phase 2 (dim=96)", "~56M", "128^3", "2", "~38 GB", "40 GB (A100)"],
        ],
        col_widths=[W*0.22, W*0.13, W*0.1, W*0.1, W*0.17, W*0.28]
    ))

    story.append(Paragraph(
        "Safety margin: use only 80-85% of nominal VRAM. "
        "If estimated need is 20GB, choose a 24GB GPU. "
        "Full VRAM utilization causes OOM crashes, cuDNN tuning failures, and fragmentation.",
        style_note
    ))

    story.append(Paragraph("Recommended Configurations", style_h2))
    story.append(make_table(
        ["Plan", "GPU", "Config", "Cost"],
        [
            ["Best value", "T4 16GB", "Phase 2+3, patch 96^3, batch 2", "Colab Free"],
            ["Balanced", "RTX 3090 24GB", "Phase 2, patch 96^3, batch 2", "AutoDL ~3 CNY/hr"],
            ["Best experience", "A100 40GB", "Phase 2, patch 128^3, batch 2", "Colab Pro ~$12/mo"],
        ],
        col_widths=[W*0.18, W*0.22, W*0.38, W*0.22]
    ))
    story.append(Paragraph(
        "Local RTX 4060 8GB: code development, debugging, smoke tests only (--max-samples 50). "
        "Formal training and ablation experiments on remote 16GB+ GPU.",
        style_body
    ))

    story.append(PageBreak())

    # ════════════════════════════════════════
    # 5. Text Guidance Value Assessment
    # ════════════════════════════════════════
    story.append(Paragraph("5. Text Guidance Value Assessment", style_h1))

    story.append(Paragraph("Current Evidence", style_h2))
    story.append(make_table(
        ["Metric", "Value"],
        [
            ["With text Dice", "0.6233"],
            ["Without text Dice", "0.5959"],
            ["Absolute gain", "+2.74%"],
            ["Measurement point", "Epoch 11/150 (far from convergence)"],
            ["Statistical test", "None (single split, single run)"],
        ],
        col_widths=[W*0.35, W*0.65]
    ))

    story.append(Paragraph("Why Current Evidence Is Unreliable", style_h2))
    for item in [
        "Model Dice is 0.62 vs baseline 0.84 -- measuring gains on severely underfitting model is meaningless",
        "Text branch adds ~21M trainable params -- the +2.7% could be parameter capacity, not semantic information",
        "TextBraTS original paper showed only +1.2% gain (85.3% vs 84.1%) with simpler fusion",
        "No statistical significance test (single split, single run)",
    ]:
        story.append(Paragraph(f"- {item}", style_bullet))

    story.append(Paragraph("Required Experiments to Validate", style_h2))
    for item in [
        "Fair ablation: remove text branch, add equivalent params to image encoder, compare at same total params",
        "Converge first: measure text gain only when Dice > 0.80",
        "Multiple runs: at least 3 seeds, report mean +/- std",
        "Statistical test: paired t-test or Wilcoxon signed-rank test",
    ]:
        story.append(Paragraph(f"- {item}", style_bullet))

    story.append(PageBreak())

    # ════════════════════════════════════════
    # 6. Idea2Paper RAG System
    # ════════════════════════════════════════
    story.append(Paragraph("6. Idea2Paper RAG System Overview", style_h1))
    story.append(Paragraph(
        "The Idea2Paper RAG system retrieves relevant research patterns from a knowledge graph "
        "of 8,285 ICLR 2025 papers using a three-path fusion retrieval strategy.",
        style_body
    ))

    story.append(Paragraph("Knowledge Base", style_h2))
    story.append(make_table(
        ["Entity", "Count", "Source"],
        [
            ["Papers", "8,285", "ICLR 2025"],
            ["Ideas", "8,284", "Extracted from papers"],
            ["Patterns", "124", "Clustered research paradigms"],
            ["Domains", "98", "Research domains"],
            ["Edges", "485,508", "Directed relationships"],
        ],
        col_widths=[W*0.25, W*0.2, W*0.55]
    ))

    story.append(Paragraph("Three-Path Fusion Retrieval", style_h2))
    story.append(make_table(
        ["Path", "Weight", "Flow"],
        [
            ["Path 1: Idea->Idea->Pattern", "0.4", "User idea matches 8,284 ideas -> top-20 -> trace Patterns"],
            ["Path 2: Idea->Domain->Pattern", "0.2", "Best idea -> graph traversal -> find applicable Patterns"],
            ["Path 3: Idea->Paper->Pattern", "0.4", "User idea matches 8,285 titles -> top-20 -> trace Patterns"],
        ],
        col_widths=[W*0.32, W*0.1, W*0.58]
    ))
    story.append(Paragraph(
        "Each path uses two-stage coarse-to-fine ranking: Jaccard token overlap (top-100) "
        "then embedding cosine similarity (re-rank). Embedding: Gemini text-embedding-004. "
        "Vector store: raw NumPy .npy (brute-force exact search, sufficient for 8K papers).",
        style_body
    ))

    story.append(Paragraph("Novelty Check", style_h2))
    story.append(make_table(
        ["Parameter", "Value"],
        [
            ["High risk threshold", "cosine >= 0.88"],
            ["Medium risk threshold", "cosine >= 0.82"],
            ["Action on high risk", "Auto-pivot (max 2 times)"],
            ["Index text", "title + domain + idea + pattern_details"],
        ],
        col_widths=[W*0.35, W*0.65]
    ))

    story.append(PageBreak())

    # ════════════════════════════════════════
    # 7. Key Concepts Reference
    # ════════════════════════════════════════
    story.append(Paragraph("7. Key Concepts Reference", style_h1))

    # 7.1 LoRA
    story.append(Paragraph("7.1 LoRA (Low-Rank Adaptation)", style_h2))
    story.append(Paragraph(
        "Parameter-efficient fine-tuning: W' = W + A x B, where A (d x r) and B (r x d), r &lt;&lt; d. "
        "Freezes original weights, trains only low-rank matrices. "
        "For a 7B model: full fine-tuning = 7B params, LoRA (r=16) = ~20M params (0.3%).",
        style_body
    ))
    story.append(Paragraph("Relevance to TextMamba3D:", style_h3))
    story.append(make_table(
        ["Approach", "Trainable Params", "Overfitting Risk"],
        [
            ["Current: unfreeze BERT last 2 layers", "+14M", "High"],
            ["Fully freeze BERT", "+0", "Lowest"],
            ["LoRA (r=8) on BERT", "+0.3M", "Low (recommended)"],
        ],
        col_widths=[W*0.4, W*0.25, W*0.35]
    ))

    # 7.2 Domain Shift
    story.append(Paragraph("7.2 Domain Shift", style_h2))
    story.append(Paragraph(
        "Distribution mismatch between training and deployment data. "
        "In medical imaging: scanner differences (GE vs Siemens), acquisition protocols "
        "(1.5T vs 3T), patient demographics, and annotation standards.",
        style_body
    ))
    story.append(make_table(
        ["Type", "Definition", "Example"],
        [
            ["Covariate Shift", "P(X) changes, P(Y|X) unchanged", "Different MRI scanners"],
            ["Label Shift", "P(Y) changes", "Different tumor type ratios by region"],
            ["Concept Shift", "P(Y|X) changes", "Different radiologist annotation standards"],
        ],
        col_widths=[W*0.22, W*0.38, W*0.40]
    ))
    story.append(Paragraph(
        "Current priority: overfitting is more urgent than domain shift. "
        "Address domain shift after model converges (Dice > 0.80). "
        "But adding Domain Randomization (brightness/contrast/gamma) during augmentation "
        "is low-cost and can be done now.",
        style_body
    ))

    # 7.3 FiLM
    story.append(Paragraph("7.3 FiLM Modulation", style_h2))
    story.append(Paragraph(
        "Feature-wise Linear Modulation: output = gamma * features + beta, "
        "where gamma (scale) and beta (shift) are predicted by conditional input (text). "
        "Channel-level global modulation, O(C) complexity -- much lighter than Cross-Attention O(n*m).",
        style_body
    ))
    story.append(make_table(
        ["Dimension", "FiLM", "Cross-Attention"],
        [
            ["Granularity", "Channel-level (global)", "Token/pixel-level (local)"],
            ["Compute", "O(C), very light", "O(n*m), heavier"],
            ["Expressiveness", "Global scale/shift", "Fine-grained spatial alignment"],
            ["Use case", "'Overall style adjustment'", "'Which pixel maps to which word'"],
        ],
        col_widths=[W*0.2, W*0.4, W*0.4]
    ))

    story.append(PageBreak())

    # ════════════════════════════════════════
    # 8. Action Checklist
    # ════════════════════════════════════════
    story.append(Paragraph("8. Action Checklist", style_h1))

    story.append(Paragraph("Local (RTX 4060 8GB) -- Code Changes", style_h2))
    checklist_local = [
        ["1", "Modify _split_cases()", "Three-way split or 5-fold CV", "brats_textbrats_dataset.py"],
        ["2", "Add dropout=0.1", "All Mamba blocks", "mamba_block.py, encoder_3d.py"],
        ["3", "Update YAML config", "weight_decay=1e-2, unfreeze=0, no_text=0.25", "textbrats.yaml"],
        ["4", "Smoke test", "--max-samples 50", "train.py"],
    ]
    story.append(make_table(
        ["#", "Task", "Details", "File"],
        checklist_local,
        col_widths=[W*0.06, W*0.25, W*0.38, W*0.31]
    ))

    story.append(Paragraph("Remote GPU (16GB+) -- Formal Training", style_h2))
    checklist_remote = [
        ["5", "Phase 2 training", "patch 96^3, batch 2+, full 150 epochs", "Remote GPU"],
        ["6", "Evaluate Phase 2", "Check if overfitting persists", "train/val curves"],
        ["7", "Phase 3 (if needed)", "dim=48, reduced depths", "Remote GPU"],
        ["8", "Text guidance ablation", "After Dice > 0.80, fair comparison", "Remote GPU"],
    ]
    story.append(make_table(
        ["#", "Task", "Details", "Environment"],
        checklist_remote,
        col_widths=[W*0.06, W*0.25, W*0.38, W*0.31]
    ))

    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("Target: Dice 0.80+ (matching nnU-Net 0.841), then validate text guidance gain.", style_h2))

    story.append(Spacer(1, 15*mm))

    # 参考文献
    story.append(Paragraph("References", style_h1))
    refs = [
        "TextBraTS: Text-Guided Volumetric Brain Tumor Segmentation (MICCAI 2025) -- arxiv.org/abs/2506.16784",
        "SegMamba: Long-range Sequential Modeling Mamba for 3D Medical Image Segmentation -- arxiv.org/abs/2401.13560",
        "LoRA: Low-Rank Adaptation of Large Language Models -- arxiv.org/abs/2106.09685",
        "FiLM: Visual Reasoning with a General Conditioning Layer -- arxiv.org/abs/1709.07871",
        "Mamba: Linear-Time Sequence Modeling with Selective State Spaces -- arxiv.org/abs/2312.00752",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", ParagraphStyle(
            "Ref", fontName=CN_FONT, fontSize=9, leading=13,
            leftIndent=8*mm, spaceAfter=1*mm,
        )))

    # ── Build ──
    doc.build(story)
    print(f"PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
