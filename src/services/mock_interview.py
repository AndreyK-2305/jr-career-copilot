from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import yaml
from google import genai
from google.genai import types


Message = dict[str, Any]
TranscriptEntry = dict[str, str]


class MockInterviewService:
    """Interactive technical interview simulator backed by Gemini chat memory."""

    def __init__(
        self,
        profile: dict[str, Any],
        job_description: str,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        max_questions: int = 7,
        output_path: str = "output/interview_transcript.md",
    ) -> None:
        if max_questions < 1 or max_questions > 7:
            raise ValueError("max_questions must be between 1 and 7.")

        self.profile = profile
        self.job_description = job_description
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.max_questions = max_questions
        self.output_path = output_path
        self.messages: list[Message] = []
        self.transcript: list[TranscriptEntry] = []
        self.questions_asked = 0
        self.system_instruction = self._build_system_instruction()

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Create a .env file or export the variable before using --mock-interview."
            )

        self.client = genai.Client(api_key=self.api_key)

    def run_interactive(
        self,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
    ) -> None:
        """Run a terminal interview with up to seven recruiter questions."""
        self.messages = []
        self.transcript = []
        self.questions_asked = 0

        output_func("[INFO] Mock interview started. Type 'salir' or 'exit' to close early.\n")
        recruiter_reply = self._ask_model(self._build_opening_prompt())
        final_feedback = False

        while True:
            speaker = "Feedback final" if final_feedback else "Entrevistador"
            self._record_transcript(speaker, recruiter_reply)
            output_func(f"{speaker}:\n{recruiter_reply}\n")

            if final_feedback:
                return

            self.questions_asked += 1
            answer = input_func("Tu respuesta: ").strip()
            while not answer:
                answer = input_func("Tu respuesta no puede estar vacia. Intenta de nuevo: ").strip()

            self._record_transcript("Candidato", answer)

            if answer.lower() in {"salir", "exit", "quit"}:
                recruiter_reply = self._ask_model(
                    self._build_feedback_prompt(
                        answer,
                        closing_reason="The candidate asked to close the interview early.",
                    )
                )
                final_feedback = True
                continue

            if self.questions_asked >= self.max_questions:
                recruiter_reply = self._ask_model(self._build_feedback_prompt(answer))
                final_feedback = True
                continue

            recruiter_reply = self._ask_model(self._build_next_question_prompt(answer))

    def export_transcript(self, output_path: str | None = None) -> str:
        """Export the visible interview transcript to Markdown."""
        target = Path(output_path or self.output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Mock Interview Transcript",
            "",
            f"- Model: `{self.model}`",
            f"- Max questions: `{self.max_questions}`",
            "",
        ]

        for entry in self.transcript:
            lines.append(f"## {entry['speaker']}")
            lines.append(entry["text"].strip())
            lines.append("")

        target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return str(target)

    def _ask_model(self, prompt: str) -> str:
        self.messages.append({"role": "user", "parts": [{"text": prompt}]})
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            temperature=0.4,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=self.messages,
            config=config,
        )
        response_text = getattr(response, "text", None)
        if not response_text:
            raise RuntimeError("Gemini returned an empty response during the mock interview.")

        clean_response = response_text.strip()
        self.messages.append({"role": "model", "parts": [{"text": clean_response}]})
        return clean_response

    def _record_transcript(self, speaker: str, text: str) -> None:
        self.transcript.append({"speaker": speaker, "text": text})

    def _build_system_instruction(self) -> str:
        return (
            "Eres un reclutador tecnico senior que conduce entrevistas simuladas para perfiles junior. "
            "Debes mantener memoria de la conversacion y adaptar cada pregunta a las respuestas del candidato. "
            "REGLA CRITICA: SOLO puedes preguntar sobre tecnologias, herramientas, proyectos, experiencia, "
            "educacion y responsabilidades que aparezcan en el CV/perfil YAML del candidato o en la descripcion "
            "del cargo. No inventes tecnologias ni cambies el cargo objetivo. "
            "Haz una sola pregunta por turno. No des la respuesta correcta antes de que el candidato responda. "
            "Realiza como maximo 7 preguntas. Al cerrar, entrega feedback claro con fortalezas, brechas tecnicas "
            "y recomendaciones accionables, y no hagas mas preguntas."
        )

    def _build_opening_prompt(self) -> str:
        return (
            "Start the mock interview now. Ask question 1 only.\n\n"
            "Candidate profile YAML:\n"
            f"{yaml.dump(self.profile, allow_unicode=True, sort_keys=False)}\n"
            "Target job description:\n"
            f"{self.job_description}\n\n"
            "Remember: ask only about technologies and requirements that appear in these inputs."
        )

    def _build_next_question_prompt(self, answer: str) -> str:
        next_question = self.questions_asked + 1
        return (
            f"Candidate answer to question {self.questions_asked}:\n{answer}\n\n"
            f"Ask technical question {next_question} of {self.max_questions}. "
            "Use the candidate answer plus the profile/JD context. Ask only one question and keep it contextual."
        )

    def _build_feedback_prompt(self, answer: str, closing_reason: str | None = None) -> str:
        reason = closing_reason or f"The interview reached the maximum of {self.max_questions} questions."
        return (
            f"Candidate answer to question {self.questions_asked}:\n{answer}\n\n"
            f"{reason} Do not ask another question. Close the interview with concise final feedback. "
            "Evaluate technical clarity, truthfulness to the CV/JD, communication, strengths, and next steps."
        )
