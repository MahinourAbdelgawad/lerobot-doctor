from lerobot_doctor.runner import CheckResult, DiagnosticReport, Severity
from lerobot_doctor.streamlit_app import build_report_summary


def test_build_report_summary_includes_counts_and_checks():
    report = DiagnosticReport(dataset_path="/tmp/demo")
    check = CheckResult(name="metadata", severity=Severity.WARN)
    check.warn("Example warning")
    report.results.append(check)

    summary = build_report_summary(report)

    assert summary["overall_severity"] == "WARN"
    assert summary["counts"]["WARN"] == 1
    assert summary["checks"][0]["name"] == "metadata"
    assert summary["checks"][0]["messages"] == ["Example warning"]
