"""Simple Streamlit UI for lerobot-doctor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from lerobot_doctor.dataset_loader import load_dataset
from lerobot_doctor.report import report_to_markdown
from lerobot_doctor.runner import CheckResult, DiagnosticReport, Severity, run_checks


def build_report_summary(report: DiagnosticReport) -> dict[str, Any]:
    """Convert a diagnostic report into a UI-friendly summary."""
    return {
        "dataset_path": report.dataset_path,
        "overall_severity": report.overall_severity.value,
        "counts": report.summary_counts,
        "checks": [
            {
                "name": check.name,
                "severity": check.severity.value,
                "messages": [message.message for message in check.messages if message.severity != Severity.PASS],
            }
            for check in report.results
        ],
        "total_episodes": report.total_episodes,
        "total_frames": report.total_frames,
        "fps": report.fps,
        "codebase_version": report.codebase_version,
        "format_version": report.format_version,
    }


def _render_summary(summary: dict[str, Any]) -> None:
    st.title("lerobot-doctor")
    st.caption("Interactive diagnostics for local and Hugging Face LeRobot datasets")

    cols = st.columns(4)
    cols[0].metric("Overall", summary["overall_severity"])
    cols[1].metric("Episodes", summary["total_episodes"] or "—")
    cols[2].metric("Frames", f"{summary['total_frames']:,}" if summary["total_frames"] is not None else "—")
    cols[3].metric("FPS", summary["fps"] if summary["fps"] is not None else "—")

    details = []
    if summary["codebase_version"]:
        details.append(f"Codebase: {summary['codebase_version']}")
    if summary["format_version"]:
        details.append(f"Format: {summary['format_version']}")
    if details:
        st.write(" • ".join(details))

    counts = summary["counts"]
    if counts:
        st.write("Summary: " + " | ".join(f"{value} {key}" for key, value in counts.items() if value))


def _render_checks(summary: dict[str, Any]) -> None:
    for check in summary["checks"]:
        severity = check["severity"]
        icon = "✅" if severity == "PASS" else "⚠️" if severity == "WARN" else "❌"
        with st.expander(f"{icon} {check['name']} — {severity}", expanded=severity != "PASS"):
            if check["messages"]:
                for message in check["messages"]:
                    st.write(f"- {message}")
            else:
                st.write("No issues reported.")


def _render_markdown(report: DiagnosticReport) -> None:
    st.download_button(
        label="Download markdown report",
        data=report_to_markdown(report),
        file_name="lerobot-doctor-report.md",
        mime="text/markdown",
    )


def _run_analysis(dataset_input: str, max_episodes: int | None, selected_checks: list[str]) -> DiagnosticReport:
    dataset = load_dataset(dataset_input, max_episodes=max_episodes)
    if dataset.info is None:
        raise RuntimeError(dataset.info_error or "Dataset could not be loaded")
    return run_checks(dataset, checks=selected_checks or None)


def main() -> None:
    st.set_page_config(page_title="lerobot-doctor", page_icon="🩺", layout="wide")

    with st.sidebar:
        st.header("Run analysis")
        dataset_input = st.text_input(
            "Dataset path or Hugging Face repo",
            value=str(Path("../datasets/pusht").resolve()),
            help="Examples: /path/to/dataset or lerobot/pusht",
        )
        max_episodes = st.number_input(
            "Max episodes (optional)",
            min_value=1,
            max_value=10000,
            value=50,
            step=1,
        )
        selected_checks = st.multiselect(
            "Checks",
            options=["metadata", "temporal", "actions", "videos", "statistics", "episodes", "consistency", "training", "anomalies", "portability", "per_episode"],
            default=["metadata", "temporal", "actions", "videos", "statistics", "episodes", "consistency", "training", "anomalies", "portability", "per_episode"],
        )
        run_button = st.button("Run analysis", type="primary")

    if run_button:
        if not dataset_input.strip():
            st.warning("Please provide a dataset path or Hugging Face repo id.")
            st.stop()

        with st.spinner("Analyzing dataset..."):
            try:
                report = _run_analysis(dataset_input, max_episodes=max_episodes or None, selected_checks=selected_checks)
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Analysis failed: {exc}")
                st.stop()

        summary = build_report_summary(report)
        _render_summary(summary)
        _render_checks(summary)
        _render_markdown(report)
        with st.expander("Raw JSON"):
            st.code(json.dumps(summary, indent=2), language="json")
    else:
        st.info("Enter a dataset path or Hugging Face repo and click Run analysis to begin.")


if __name__ == "__main__":
    main()
