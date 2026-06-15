"""Generate the final student deliverable PDF (comments + submitted-run analysis)."""

from __future__ import annotations

import io
import os
import textwrap
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

REPORT_FIELDS = (
    ("feedback_strategy", "1. Estratégia de Feedbacks"),
    ("final_analysis", "2. Análise dos Resultados Finais"),
)

SKIP_PLOTS = {"heatmap", "trajectory_heatmap"}
PLOT_TITLES = {
    "outcomes": "Resultados por oponente",
    "contributions": "Contribuição das recompensas",
    "efficiency": "Uso de mísseis vs. abates",
    "agent_profile": "Mapa de perfil dos agentes",
}


def _pct(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "-"


def _curve_png(curve: Dict, out_path: str) -> Optional[str]:
    t = curve.get("t") or []
    if len(t) < 2:
        return None
    score = curve.get("score") or curve.get("mission") or []
    reward = curve.get("reward") or []
    fig, ax1 = plt.subplots(figsize=(6.5, 2.8))
    ax1.plot(t, score, color="#5ec27a", linewidth=1.8, label="eval score")
    ax1.set_ylabel("score")
    ax1.set_xlabel("training steps")
    ax1.grid(True, alpha=0.25)
    if reward:
        ax2 = ax1.twinx()
        ax2.plot(t, reward, color="#4c9be8", linewidth=1.2, alpha=0.85, label="train reward")
        ax2.set_ylabel("reward")
    ax1.set_title("Curva de aprendizado")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _para(text: str, style) -> Paragraph:
    safe = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\n", "<br/>")
    return Paragraph(safe or "<i>—</i>", style)


def generate_student_report_pdf(
    out_path: str,
    *,
    student_name: str,
    report_data: Dict,
    context: Dict,
    run: Optional[Dict] = None,
    stats: Optional[Dict] = None,
    plot_paths: Optional[Dict[str, str]] = None,
    curve: Optional[Dict] = None,
    rewards: Optional[Dict] = None,
    training_enemies: Optional[List[str]] = None,
) -> str:
    """Build a multi-section PDF and return ``out_path``."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"], fontSize=16, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=13)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=9, textColor=colors.grey)

    story = []
    story.append(Paragraph("Relatório Final — Atividade BVR (TE-276)", title))
    story.append(Paragraph(f"<b>Aluno:</b> {student_name}", body))
    story.append(Paragraph(
        f"<b>Gerado em:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}", small))
    story.append(Spacer(1, 0.35 * cm))

    if not context.get("has_submission"):
        story.append(Paragraph(
            "<i>Nenhum run submetido na competição. Envie seu melhor modelo em "
            "My runs antes de entregar o relatório final.</i>", body))
    else:
        story.append(Paragraph("Run submetido", h2))
        metrics = [
            ["Run", context.get("run_uid", "-")],
            ["Score", _pct(context.get("score"))],
            ["Missão", _pct(context.get("mission_rate"))],
            ["Abate", _pct(context.get("kill_rate"))],
            ["Sobrevivência", _pct(context.get("survival_rate"))],
            ["Efic. míssil", str(context.get("missile_efficiency", "-"))],
            ["Reward médio", str(context.get("mean_reward", "-"))],
            ["Passos treino", str(context.get("steps", run.get("steps") if run else "-"))],
        ]
        mt = Table(metrics, colWidths=[4.5 * cm, 11 * cm])
        mt.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(mt)

        if training_enemies:
            story.append(Spacer(1, 0.15 * cm))
            story.append(Paragraph(
                f"<b>Oponentes no treino:</b> {', '.join(training_enemies)}", small))

        if context.get("best") or context.get("worst"):
            story.append(Spacer(1, 0.2 * cm))
            best = ", ".join(f"{m['enemy']} ({_pct(m.get('mission_rate'))})"
                             for m in (context.get("best") or []))
            worst = ", ".join(f"{m['enemy']} ({_pct(m.get('mission_rate'))})"
                              for m in (context.get("worst") or []))
            if best:
                story.append(Paragraph(f"<b>Melhor vs:</b> {best}", small))
            if worst:
                story.append(Paragraph(f"<b>Pior vs:</b> {worst}", small))

    if stats and stats.get("per_enemy"):
        story.append(Paragraph("Análise do comportamento (avaliação oficial)", h2))
        story.append(Paragraph(
            f"Score agregado: <b>{_pct(stats.get('score'))}</b> · "
            f"missão {_pct(stats.get('mission_rate'))} · "
            f"abate {_pct(stats.get('kill_rate'))} · "
            f"sobrevivência {_pct(stats.get('survival_rate'))}", body))
        rows = [["Oponente", "Missão", "Sobrev.", "Abate", "Reward", "Mísseis"]]
        for enemy, e in stats["per_enemy"].items():
            rows.append([
                enemy,
                _pct(e.get("mission_rate")),
                _pct(e.get("survival")),
                _pct(e.get("kills")),
                str(e.get("mean_reward", "-")),
                str(e.get("missiles_used", "-")),
            ])
        tbl = Table(rows, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28324a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fa")]),
        ]))
        story.append(tbl)

    tmp_curve = None
    if curve:
        tmp_curve = out_path.replace(".pdf", "_curve.png")
        if _curve_png(curve, tmp_curve):
            story.append(Paragraph("Curva de aprendizado", h2))
            story.append(Image(tmp_curve, width=16 * cm, height=6.5 * cm))

    plot_paths = plot_paths or {}
    for key in ("agent_profile", "outcomes", "contributions", "efficiency"):
        path = plot_paths.get(key)
        if path and os.path.isfile(path):
            story.append(Paragraph(PLOT_TITLES.get(key, key), h2))
            story.append(Image(path, width=15 * cm, height=9 * cm))

    if rewards:
        nonzero = {k: v for k, v in rewards.items()
                   if k != "global_scale" and abs(float(v or 0)) > 1e-6}
        if nonzero:
            story.append(Paragraph("Pesos de recompensa usados no run submetido", h2))
            rw_rows = [["Termo", "Peso"]] + [
                [k.replace("_", " "), str(v)] for k, v in sorted(nonzero.items(),
                key=lambda kv: -abs(float(kv[1])))]
            rw = Table(rw_rows, colWidths=[8 * cm, 4 * cm])
            rw.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#28324a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]))
            story.append(rw)

    story.append(Paragraph("Comentários do aluno", h2))
    for key, label in REPORT_FIELDS:
        story.append(Paragraph(label, ParagraphStyle("lbl", parent=h2, fontSize=11)))
        story.append(_para(report_data.get(key, ""), body))
        story.append(Spacer(1, 0.2 * cm))

    doc.build(story)
    if tmp_curve and os.path.isfile(tmp_curve):
        try:
            os.remove(tmp_curve)
        except OSError:
            pass
    return out_path
