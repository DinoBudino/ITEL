from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


ROOT_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT_DIR / "research" / "checkpoints" / "v3_final" / "layermix" / "metadata.json"
OUTPUT_DIR = ROOT_DIR / "research" / "reports"
MODEL_SUBTITLE = "V3_HuBERT_BASE -> AttentionBiLSTM + SupCon (LayerMix)"


@dataclass(slots=True)
class ArchitectureSpec:
    arch: str
    tag: str
    input_dim: int
    lstm_hidden: int
    lstm_layers: int
    dropout: float
    fc_hidden: int
    num_heads: int
    layer_mix: bool
    embedding_noise_std: float
    attention_entropy_lambda: float
    epochs: int
    lr: float
    weight_decay: float
    pos_weight: float
    target_sr: int
    highpass_hz: int
    target_rms: float
    preprocessing_profile: str
    source_manifest: str

    @property
    def context_dim(self) -> int:
        return self.lstm_hidden * 2 * self.num_heads

    @property
    def classifier_shape(self) -> str:
        return f"FC {self.context_dim} -> {self.fc_hidden} -> 1"

    @property
    def sequence_shape(self) -> str:
        return f"T x {self.input_dim}"

    @property
    def bilstm_label(self) -> str:
        return f"BiLSTM {self.lstm_layers} x {self.lstm_hidden}"


def load_spec(metadata_path: Path) -> ArchitectureSpec:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_config = payload["model_config"]
    training_config = payload["training_config"]
    preprocessing = payload["preprocessing"]
    improvements = payload["improvements"]
    return ArchitectureSpec(
        arch=payload["arch"],
        tag=payload["tag"],
        input_dim=int(model_config["input_dim"]),
        lstm_hidden=int(model_config["lstm_hidden"]),
        lstm_layers=int(model_config["lstm_layers"]),
        dropout=float(model_config["dropout"]),
        fc_hidden=int(model_config["fc_hidden"]),
        num_heads=int(model_config["num_heads"]),
        layer_mix=bool(improvements["layer_mix"]),
        embedding_noise_std=float(improvements["embedding_noise_std"]),
        attention_entropy_lambda=float(improvements["attention_entropy_lambda"]),
        epochs=int(training_config["epochs"]),
        lr=float(training_config["lr"]),
        weight_decay=float(training_config["weight_decay"]),
        pos_weight=float(training_config["pos_weight"]),
        target_sr=int(preprocessing["target_sr"]),
        highpass_hz=int(preprocessing["highpass_hz"]),
        target_rms=float(preprocessing["target_rms"]),
        preprocessing_profile=str(preprocessing["default_profile"]),
        source_manifest=str(payload["source_manifest"]),
    )


def svg_document(width: int, height: int, body: str, background: str = "#f8fafc") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        f'<title id="title">{escape(MODEL_SUBTITLE)}</title>'
        f'<desc id="desc">Architecture diagram for the LayerMix final model.</desc>'
        f'<rect width="100%" height="100%" fill="{background}"/>{body}</svg>'
    )


def text(x: float, y: float, value: str, size: int = 16, weight: str = "400", fill: str = "#0f172a", anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def rounded_rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "#cbd5e1", rx: int = 18, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" opacity="{opacity}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#94a3b8", width: float = 2.0, dash: str | None = None, opacity: float = 1.0) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linecap="round" opacity="{opacity}"{dash_attr}/>'
    )


def path_curve(d: str, stroke: str = "#94a3b8", width: float = 2.0, fill: str = "none", opacity: float = 1.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
        f'stroke-linecap="round" opacity="{opacity}"{dash_attr}/>'
    )


def circle(cx: float, cy: float, radius: float, fill: str, stroke: str = "#cbd5e1", width: float = 1.5, opacity: float = 1.0) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{width}" opacity="{opacity}"/>'
    )


def polygon(points: Iterable[tuple[float, float]], fill: str, stroke: str = "#cbd5e1", width: float = 1.5, opacity: float = 1.0) -> str:
    serialized = " ".join(f"{x},{y}" for x, y in points)
    return (
        f'<polygon points="{serialized}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{width}" opacity="{opacity}"/>'
    )


def label_block(x: float, y: float, width: float, height: float, title: str, subtitle: str, fill: str, accent: str) -> str:
    parts = [rounded_rect(x, y, width, height, fill=fill, stroke=accent, rx=22)]
    parts.append(rounded_rect(x + 12, y + 12, 8, height - 24, fill=accent, stroke=accent, rx=6))
    parts.append(text(x + 34, y + 40, title, size=21, weight="700"))
    parts.append(text(x + 34, y + 72, subtitle, size=13, fill="#475569"))
    return "".join(parts)


def add_common_header(title_value: str, width: int) -> str:
    parts = [text(40, 48, title_value, size=28, weight="700")]
    parts.append(text(40, 76, MODEL_SUBTITLE, size=15, fill="#475569"))
    parts.append(line(40, 92, width - 40, 92, stroke="#dbe4f0", width=1.5))
    return "".join(parts)


def render_schematic(spec: ArchitectureSpec) -> str:
    width = 1600
    height = 860
    parts = [add_common_header("LayerMix Architecture - Schematic", width)]

    box_w = 210
    box_h = 108
    y = 230
    xs = [40, 280, 520, 760, 1000, 1240]
    blocks = [
        ("Audio", f"16 kHz WAV | HP {spec.highpass_hz} Hz", "#e0f2fe", "#0284c7"),
        ("HuBERT_BASE", f"Frozen encoder | per-frame {spec.input_dim}D", "#ede9fe", "#7c3aed"),
        ("ScalarMix", "12 layers -> weighted layer fusion", "#f5f3ff", "#8b5cf6"),
        ("Sequence", spec.sequence_shape, "#eef2ff", "#4f46e5"),
        (spec.bilstm_label, f"Bidirectional -> context features {spec.context_dim}", "#ecfeff", "#0891b2"),
        ("Attention + Head", f"Context {spec.context_dim} | {spec.classifier_shape}", "#f0fdf4", "#16a34a"),
    ]

    for index, (title_value, subtitle, fill, accent) in enumerate(blocks):
        parts.append(label_block(xs[index], y, box_w, box_h, title_value, subtitle, fill, accent))
        if index < len(blocks) - 1:
            x1 = xs[index] + box_w
            x2 = xs[index + 1]
            mid_y = y + box_h / 2
            parts.append(line(x1 + 8, mid_y, x2 - 18, mid_y, stroke="#94a3b8", width=3))
            parts.append(polygon([(x2 - 18, mid_y - 8), (x2, mid_y), (x2 - 18, mid_y + 8)], fill="#94a3b8", stroke="#94a3b8"))

    parts.append(label_block(1240, 420, 220, 108, "Sigmoid + Output", "Healthy / Patients", "#fff7ed", "#ea580c"))
    parts.append(line(1450, 284, 1450, 420, stroke="#94a3b8", width=3))
    parts.append(polygon([(1442, 404), (1450, 420), (1458, 404)], fill="#94a3b8", stroke="#94a3b8"))

    info_x = 70
    info_y = 600
    parts.append(text(info_x, info_y, "Interpretation", size=20, weight="700"))
    info_lines = [
        f"LayerMix enabled: {'yes' if spec.layer_mix else 'no'} | attention heads: {spec.num_heads}",
        f"Classifier head: LayerNorm -> Dropout({spec.dropout}) -> Linear({spec.context_dim},{spec.fc_hidden}) -> ReLU -> Dropout({spec.dropout}) -> Linear({spec.fc_hidden},1)",
        f"Training: epochs={spec.epochs} | lr={spec.lr} | weight_decay={spec.weight_decay} | pos_weight={spec.pos_weight}",
        f"Source manifest: {spec.source_manifest}",
    ]
    for idx, line_value in enumerate(info_lines):
        parts.append(text(info_x, info_y + 36 + idx * 28, line_value, size=14, fill="#334155"))

    parts.append(rounded_rect(1120, 590, 380, 180, fill="#ffffff", stroke="#cbd5e1", rx=24))
    parts.append(text(1150, 628, "Notes", size=20, weight="700"))
    notes = [
        "T = variable number of HuBERT time frames.",
        f"Each frame carries {spec.input_dim} embedding features.",
        f"BiLSTM output is concatenated across directions -> {spec.context_dim}.",
        "Sigmoid produces a single Patients probability before thresholding.",
    ]
    for idx, note in enumerate(notes):
        parts.append(text(1150, 662 + idx * 26, note, size=14, fill="#475569"))

    return svg_document(width, height, "".join(parts))


def render_playground(spec: ArchitectureSpec) -> str:
    width = 1600
    height = 940
    parts = [add_common_header("LayerMix Architecture - Playground", width)]

    col_x = [160, 430, 760, 1080, 1390]
    top_y = 220
    gap = 88
    radius = 24
    layers = [
        [("Audio", "#111111"), ("Wave", "#2f2f2f"), ("Frames", "#5a5a5a")],
        [("HuBERT", "#111111"), ("L1-L4", "#2f2f2f"), ("L5-L8", "#4a4a4a"), ("L9-L12", "#6a6a6a"), ("Mix", "#111111")],
        [("Seq", "#111111"), ("Bi", "#2f2f2f"), ("LSTM", "#4a4a4a"), ("Attn", "#6a6a6a")],
        [("Ctx", "#111111"), ("FC128", "#3a3a3a"), ("Logit", "#6a6a6a")],
        [("Healthy", "#111111"), ("Patients", "#4a4a4a")],
    ]

    centers: list[list[tuple[float, float]]] = []
    for x, nodes in zip(col_x, layers):
        current = []
        start_y = top_y + (5 - len(nodes)) * 40
        for idx, (label_value, color) in enumerate(nodes):
            cy = start_y + idx * gap
            current.append((x, cy))
            parts.append(circle(x, cy, radius, fill="#ffffff", stroke=color, width=4))
            parts.append(circle(x, cy, radius - 9, fill=color, stroke=color, width=1, opacity=0.14))
            parts.append(text(x, cy + 48, label_value, size=13, fill="#334155", anchor="middle"))
        centers.append(current)

    connection_palette = ["#d4d4d4", "#b8b8b8", "#8f8f8f", "#5f5f5f"]
    for layer_idx in range(len(centers) - 1):
        from_nodes = centers[layer_idx]
        to_nodes = centers[layer_idx + 1]
        for src_idx, (x1, y1) in enumerate(from_nodes):
            for dst_idx, (x2, y2) in enumerate(to_nodes):
                ctrl_dx = (x2 - x1) * 0.45
                d = f"M {x1 + radius} {y1} C {x1 + ctrl_dx} {y1}, {x2 - ctrl_dx} {y2}, {x2 - radius} {y2}"
                color = connection_palette[(src_idx + dst_idx + layer_idx) % len(connection_palette)]
                opacity = 0.26 + (((src_idx + dst_idx) % 3) * 0.18)
                stroke_w = 1.2 + (((src_idx * 2 + dst_idx) % 3) * 0.8)
                parts.append(path_curve(d, stroke=color, width=stroke_w, opacity=opacity))

    headers = [
        (160, "INPUT"),
        (430, "ENCODER + LAYER MIX"),
        (760, "SEQUENCE MODEL"),
        (1080, "CLASSIFIER"),
        (1390, "OUTPUT"),
    ]
    for x, label_value in headers:
        parts.append(text(x, 170, label_value, size=18, weight="700", fill="#0f172a", anchor="middle"))

    parts.append(text(70, 664, "Model signals", size=20, weight="700"))
    detail_lines = [
        f"HuBERT_BASE emits sequential embeddings with shape {spec.sequence_shape}.",
        "ScalarMix learns a weighted combination over the 12 transformer layers.",
        f"BiLSTM uses {spec.lstm_layers} layers with hidden size {spec.lstm_hidden} per direction.",
        f"Attention compresses time into a {spec.context_dim}-dimensional context vector.",
        f"Final head maps {spec.context_dim} -> {spec.fc_hidden} -> 1, then sigmoid scores Patients probability.",
    ]
    for idx, line_value in enumerate(detail_lines):
        parts.append(text(70, 700 + idx * 28, line_value, size=15, fill="#334155"))

    parts.append(rounded_rect(1110, 630, 420, 210, fill="#ffffff", stroke="#d4d4d4", rx=28))
    parts.append(text(1140, 670, "Hyperparameters", size=20, weight="700"))
    metrics = [
        f"dropout={spec.dropout}",
        f"fc_hidden={spec.fc_hidden}",
        f"epochs={spec.epochs}",
        f"lr={spec.lr}",
        f"weight_decay={spec.weight_decay}",
        f"pos_weight={spec.pos_weight}",
    ]
    for idx, value in enumerate(metrics):
        col = idx // 3
        row = idx % 3
        parts.append(text(1140 + col * 180, 708 + row * 36, value, size=15, fill="#475569"))

    parts.append(text(70, 850, "Legend: T = number of time frames extracted from the input audio.", size=14, fill="#64748b"))
    return svg_document(width, height, "".join(parts), background="#ffffff")


def render_technical(spec: ArchitectureSpec) -> str:
    width = 1640
    height = 980
    parts = [add_common_header("LayerMix Architecture - Technical", width)]

    stage_x = 90
    stage_y = 170
    stage_h = 560
    stage_w = 980
    parts.append(rounded_rect(stage_x, stage_y, stage_w, stage_h, fill="#ffffff", stroke="#d7e0ea", rx=28))
    parts.append(text(stage_x + 30, stage_y + 44, "Tensor flow", size=22, weight="700"))

    boxes = [
        (130, 260, 170, 86, "Audio", "raw waveform", "#e0f2fe", "#0284c7"),
        (340, 250, 200, 106, "HuBERT_BASE", f"12 layers | frame dim {spec.input_dim}", "#ede9fe", "#7c3aed"),
        (590, 250, 170, 106, "ScalarMix", "weighted 12-layer fusion", "#f5f3ff", "#8b5cf6"),
        (800, 250, 210, 106, "Sequence tensor", spec.sequence_shape, "#eef2ff", "#4f46e5"),
        (275, 450, 260, 118, spec.bilstm_label, f"bidirectional output -> T x {spec.context_dim}", "#ecfeff", "#0891b2"),
        (595, 450, 220, 118, "Attention", f"{spec.num_heads} head -> Context {spec.context_dim}", "#ecfccb", "#65a30d"),
        (870, 435, 150, 150, "Classifier", f"{spec.classifier_shape} | sigmoid", "#fff7ed", "#ea580c"),
    ]

    for x, y, w, h, title_value, subtitle, fill, accent in boxes:
        parts.append(label_block(x, y, w, h, title_value, subtitle, fill, accent))

    arrow_segments = [
        ((300, 303), (340, 303)),
        ((540, 303), (590, 303)),
        ((760, 303), (800, 303)),
        ((905, 356), (905, 435)),
        ((535, 509), (595, 509)),
        ((815, 509), (870, 509)),
    ]
    for (x1, y1), (x2, y2) in arrow_segments:
        parts.append(line(x1, y1, x2 - 18 if x1 != x2 else x2, y2 - 18 if y1 != y2 else y2, stroke="#64748b", width=3))
        if x1 != x2:
            parts.append(polygon([(x2 - 18, y2 - 8), (x2, y2), (x2 - 18, y2 + 8)], fill="#64748b", stroke="#64748b"))
        else:
            parts.append(polygon([(x2 - 8, y2 - 18), (x2, y2), (x2 + 8, y2 - 18)], fill="#64748b", stroke="#64748b"))

    parts.append(text(120, 768, "Tensor notes", size=20, weight="700"))
    tensor_notes = [
        f"Input after HuBERT: {spec.sequence_shape}",
        f"BiLSTM hidden state per time step: T x {spec.context_dim}",
        f"Attention reduces temporal axis T -> Context {spec.context_dim}",
        "Final classifier emits one logit; sigmoid converts it to P(Patients)",
    ]
    for idx, value in enumerate(tensor_notes):
        parts.append(text(120, 802 + idx * 28, value, size=15, fill="#334155"))

    panel_x = 1120
    panel_y = 170
    panel_w = 450
    panel_h = 730
    parts.append(rounded_rect(panel_x, panel_y, panel_w, panel_h, fill="#ffffff", stroke="#d7e0ea", rx=28))
    parts.append(text(panel_x + 30, panel_y + 44, "Metadata and training", size=22, weight="700"))

    metadata_lines = [
        f"arch: {spec.arch}",
        f"tag: {spec.tag}",
        f"source_manifest: {spec.source_manifest}",
        f"layer_mix: {spec.layer_mix}",
        f"embedding_noise_std: {spec.embedding_noise_std}",
        f"attention_entropy_lambda: {spec.attention_entropy_lambda}",
        "",
        "Model config",
        f"input_dim={spec.input_dim}",
        f"lstm_hidden={spec.lstm_hidden}",
        f"lstm_layers={spec.lstm_layers}",
        f"num_heads={spec.num_heads}",
        f"dropout={spec.dropout}",
        f"fc_hidden={spec.fc_hidden}",
        "",
        "Preprocessing",
        f"target_sr={spec.target_sr}",
        f"highpass_hz={spec.highpass_hz}",
        f"target_rms={spec.target_rms}",
        f"profile={spec.preprocessing_profile}",
        "",
        "Training config",
        f"epochs={spec.epochs}",
        f"lr={spec.lr}",
        f"weight_decay={spec.weight_decay}",
        f"pos_weight={spec.pos_weight}",
    ]
    y = panel_y + 84
    for value in metadata_lines:
        if not value:
            y += 18
            continue
        is_section = value in {"Model config", "Preprocessing", "Training config"}
        parts.append(text(panel_x + 30, y, value, size=16 if is_section else 14, weight="700" if is_section else "400", fill="#0f172a" if is_section else "#475569"))
        y += 28 if is_section else 24

    parts.append(text(panel_x + 30, panel_y + panel_h - 24, "Legend: T = number of valid HuBERT time frames for one audio sample.", size=13, fill="#64748b"))
    return svg_document(width, height, "".join(parts), background="#f8fafc")


def write_svg(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    spec = load_spec(METADATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        OUTPUT_DIR / "model_architecture_layermix_schematic.svg": render_schematic(spec),
        OUTPUT_DIR / "model_architecture_layermix_playground.svg": render_playground(spec),
        OUTPUT_DIR / "model_architecture_layermix_technical.svg": render_technical(spec),
    }

    for path, content in outputs.items():
        write_svg(path, content)
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
