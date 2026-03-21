"""Keyword extraction for structured TextBraTS prompts.

This module converts the expert-style free text into a structured keyword
sentence so we can keep the current tokenizer + text encoder stack while
reducing textual noise.
"""

from __future__ import annotations

import re


LOCATION_MARKER = "the lesion area is in "
EDEMA_MARKER = "edema is "
NECROSIS_MARKER = "necrosis is "
VENTRICULAR_MARKER = "ventricular compression is "

NEXT_MARKERS = (
    " edema is ",
    " necrosis is ",
    " ventricular compression is ",
)

ABSENT_PATTERNS = (
    "not observed",
    "not evident",
    "not seen",
    "absent",
    "without ",
    "no ",
)

SIDE_WORDS = ("left", "right", "bilateral")
LOBE_WORDS = ("frontal", "parietal", "temporal", "occipital", "insular")

EXTENT_PATTERNS = (
    ("significant", "significant"),
    ("marked", "marked"),
    ("considerable", "considerable"),
    ("extensive", "extensive"),
    ("pronounced", "pronounced"),
    ("severe", "severe"),
    ("moderate", "moderate"),
    ("notable", "notable"),
    ("mild", "mild"),
    ("slight", "slight"),
    ("partial", "partial"),
    ("partially", "partial"),
)

NECROSIS_PATTERN_PATTERNS = (
    ("clustered", "clustered"),
    ("scattered", "scattered"),
    ("patchy", "patchy"),
    ("mixed", "mixed"),
    ("central", "central"),
    ("diffuse", "diffuse"),
    ("focal", "focal"),
)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _normalize_text(text: str) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_span(text: str, marker: str) -> str | None:
    lower_text = text.lower()
    start = lower_text.find(marker)
    if start == -1:
        return None

    start += len(marker)
    end = len(text)
    for next_marker in NEXT_MARKERS:
        idx = lower_text.find(next_marker, start)
        if idx != -1:
            end = min(end, idx)

    span = text[start:end].strip(" .;,:")
    return span or None


def _extract_location(text: str) -> str:
    span = _extract_span(text, LOCATION_MARKER)
    if not span:
        return "unknown"

    # Keep the anatomical phrase before appearance/signal details.
    for splitter in (" with ", ", with ", " showing ", ", showing ", " exhibiting ", ", exhibiting "):
        idx = span.lower().find(splitter)
        if idx != -1:
            span = span[:idx]
            break

    span = span.strip(" .;,:")
    return span if span else "unknown"


def _extract_side(span: str | None) -> str:
    if not span:
        return "unknown"
    lower_span = span.lower()
    found = [side for side in SIDE_WORDS if side in lower_span]
    if not found:
        return "unknown"
    return ",".join(_dedupe_keep_order(found))


def _extract_lobes(span: str | None) -> str:
    if not span:
        return "unknown"
    lower_span = span.lower()
    found = [lobe for lobe in LOBE_WORDS if lobe in lower_span]
    if not found:
        return "unknown"
    return ",".join(_dedupe_keep_order(found))


def _extract_binary_status(text: str, marker: str) -> str:
    span = _extract_span(text, marker)
    if not span:
        return "unknown"

    lower_span = span.lower()
    if any(pattern in lower_span for pattern in ABSENT_PATTERNS):
        return "absent"
    return "present"


def _extract_extent(span: str | None) -> str:
    if not span:
        return "unknown"

    lower_span = span.lower()
    found = [label for pattern, label in EXTENT_PATTERNS if pattern in lower_span]
    if not found:
        return "unknown"
    return ",".join(_dedupe_keep_order(found))


def _extract_necrosis_pattern(span: str | None) -> str:
    if not span:
        return "unknown"

    lower_span = span.lower()
    found = [label for pattern, label in NECROSIS_PATTERN_PATTERNS if pattern in lower_span]
    if not found:
        return "unknown"
    return ",".join(_dedupe_keep_order(found))


def _extract_feature_summary(text: str, marker: str) -> tuple[str, str, str, str]:
    span = _extract_span(text, marker)
    if not span:
        return "unknown", "unknown", "unknown", "unknown"

    status = _extract_binary_status(text, marker)
    if status == "absent":
        return status, "none", "none", "none"

    side = _extract_side(span)
    lobes = _extract_lobes(span)
    extent = _extract_extent(span)
    return status, extent, side, lobes


def build_keyword_text(text: str) -> str:
    """Convert a free-text report into a structured keyword prompt."""
    normalized = _normalize_text(text)
    lesion_span = _extract_location(normalized)
    lesion_side = _extract_side(lesion_span)
    lesion_lobes = _extract_lobes(lesion_span)

    edema_status, edema_extent, edema_side, edema_lobes = _extract_feature_summary(
        normalized, EDEMA_MARKER
    )
    necrosis_status, _, necrosis_side, necrosis_lobes = _extract_feature_summary(
        normalized, NECROSIS_MARKER
    )
    ventricular_status, ventricular_extent, _, _ = _extract_feature_summary(
        normalized, VENTRICULAR_MARKER
    )
    necrosis_pattern = _extract_necrosis_pattern(_extract_span(normalized, NECROSIS_MARKER))
    if necrosis_status == "absent":
        necrosis_pattern = "none"

    return (
        f"lesion_side: {lesion_side}; "
        f"lesion_lobes: {lesion_lobes}; "
        f"edema_status: {edema_status}; "
        f"edema_extent: {edema_extent}; "
        f"edema_side: {edema_side}; "
        f"edema_lobes: {edema_lobes}; "
        f"necrosis_status: {necrosis_status}; "
        f"necrosis_pattern: {necrosis_pattern}; "
        f"necrosis_side: {necrosis_side}; "
        f"necrosis_lobes: {necrosis_lobes}; "
        f"ventricular_compression_status: {ventricular_status}; "
        f"ventricular_compression_extent: {ventricular_extent}"
    )
