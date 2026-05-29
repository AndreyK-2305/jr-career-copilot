from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from models import OptimizedCV


class Alucinacion(BaseModel):
    linea_cv: str = Field(description="Generated CV line, bullet, or field where unsupported data appears.")
    dato_inventado: str = Field(description="Specific claim that is not supported by the original profile.")
    severidad: str = Field(description="Severity: baja, media, or alta.")


class Inconsistencia(BaseModel):
    elemento_cv: str = Field(description="Generated CV element with a contradiction or mismatch.")
    conflicto: str = Field(description="Explanation of the inconsistency against the profile or job description.")
    severidad: str = Field(description="Severity: baja, media, or alta.")


class ViolacionEtica(BaseModel):
    descripcion: str = Field(description="Ethical or professional integrity issue found in the generated CV.")
    regla_afectada: str = Field(description="Truthfulness, discrimination, privacy, or misleading representation rule affected.")
    severidad: str = Field(description="Severity: baja, media, or alta.")


class ReporteRobustez(BaseModel):
    score_honestidad: int = Field(ge=0, le=100, description="Truthfulness score from 0 to 100.")
    alucinaciones_detectadas: list[Alucinacion] = Field(default_factory=list)
    inconsistencias_detectadas: list[Inconsistencia] = Field(default_factory=list)
    violaciones_eticas: list[ViolacionEtica] = Field(default_factory=list)
    comentario_auditor: str = Field(description="Concise auditor summary with remediation advice.")
    aprobado: bool = Field(description="True only when the generated CV is safe and faithful enough to use.")


class RobustnessJudgeService:
    """LLM-as-a-Judge validator for hallucinations and ethical issues in generated CVs."""

    def __init__(
        self,
        profile: dict[str, Any],
        job_description: str,
        optimized_cv: OptimizedCV,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        markdown_output_path: str = "output/robustness_report.md",
        json_output_path: str = "output/robustness_report.json",
    ) -> None:
        self.profile = profile
        self.job_description = job_description
        self.optimized_cv = optimized_cv
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.markdown_output_path = markdown_output_path
        self.json_output_path = json_output_path

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Create a .env file or export the variable before using --robustness."
            )

        self.client = genai.Client(api_key=self.api_key)

    def audit(self) -> ReporteRobustez:
        """Audit a generated CV against the source profile and target job description."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=self._build_prompt(),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReporteRobustez,
                temperature=0.1,
                system_instruction=(
                    "You are a strict AI safety and career ethics auditor. "
                    "Your job is to detect unsupported claims, contradictions, and ethical risks in generated resumes. "
                    "Judge only against the provided source profile and job description."
                ),
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty robustness report.")

        return ReporteRobustez.model_validate_json(response.text)

    def export_report(
        self,
        report: ReporteRobustez,
        markdown_output_path: str | None = None,
        json_output_path: str | None = None,
    ) -> tuple[str, str]:
        """Export the validated report to Markdown and JSON files."""
        markdown_target = Path(markdown_output_path or self.markdown_output_path)
        json_target = Path(json_output_path or self.json_output_path)
        markdown_target.parent.mkdir(parents=True, exist_ok=True)
        json_target.parent.mkdir(parents=True, exist_ok=True)

        markdown_target.write_text(self._render_markdown(report), encoding="utf-8")
        json_target.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return str(markdown_target), str(json_target)

    def _build_prompt(self) -> str:
        return (
            "Audit the generated CV below. Detect hallucinations, inconsistencies, and ethical violations.\n\n"
            "Rules:\n"
            "1. A hallucination is any job, metric, certification, technology, degree, date, seniority, or achievement "
            "that does not appear in the original profile.\n"
            "2. Rewording and better organization are acceptable only when the underlying fact remains true.\n"
            "3. Skills from the job description may be emphasized only if the candidate profile supports them.\n"
            "4. Penalize misleading exaggeration, fabricated impact metrics, invented experience, or privacy-sensitive claims.\n"
            "5. Return only valid JSON matching the requested schema.\n\n"
            "ORIGINAL CANDIDATE PROFILE YAML:\n"
            f"{yaml.dump(self.profile, allow_unicode=True, sort_keys=False)}\n\n"
            "TARGET JOB DESCRIPTION:\n"
            f"{self.job_description}\n\n"
            "GENERATED OPTIMIZED CV JSON:\n"
            f"{self.optimized_cv.model_dump_json(indent=2)}\n"
        )

    def _render_markdown(self, report: ReporteRobustez) -> str:
        lines = [
            "# Robustness Report",
            "",
            f"- Honesty score: `{report.score_honestidad}/100`",
            f"- Approved: `{'yes' if report.aprobado else 'no'}`",
            "",
            "## Hallucinations",
        ]
        lines.extend(self._render_findings(report.alucinaciones_detectadas, ["linea_cv", "dato_inventado", "severidad"]))
        lines.append("## Inconsistencies")
        lines.extend(self._render_findings(report.inconsistencias_detectadas, ["elemento_cv", "conflicto", "severidad"]))
        lines.append("## Ethical Violations")
        lines.extend(self._render_findings(report.violaciones_eticas, ["descripcion", "regla_afectada", "severidad"]))
        lines.extend(["## Auditor Comment", report.comentario_auditor.strip(), ""])
        return "\n".join(lines)

    def _render_findings(self, findings: list[BaseModel], fields: list[str]) -> list[str]:
        if not findings:
            return ["No findings.", ""]

        lines: list[str] = []
        for index, finding in enumerate(findings, start=1):
            lines.append(f"{index}.")
            for field in fields:
                value = getattr(finding, field)
                label = field.replace("_", " ").title()
                lines.append(f"   - {label}: {value}")
            lines.append("")
        return lines
