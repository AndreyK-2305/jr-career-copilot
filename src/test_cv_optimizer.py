import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import yaml
from pydantic import ValidationError

# Importar las clases y funciones del optimizador
from cv_optimizer import (
    OptimizedCV,
    OptimizedExperience,
    OptimizedEducation,
    ContactInfo,
    load_profile,
    load_job_description,
    generate_markdown,
    generate_html,
    optimize_cv,
    MockInterviewService,
    RobustnessJudgeService,
    ReporteRobustez
)

class TestCVOptimizer(unittest.TestCase):
    """
    Pruebas unitarias para validar la lógica del Optimizador de CV.
    Escrito de forma robusta e instructiva para ingenieros junior.
    """

    def setUp(self) -> None:
        """
        Configura datos de prueba iniciales antes de cada test.
        """
        self.sample_profile_data = {
            "personal_info": {
                "full_name": "Ingeniero Junior de Prueba",
                "email": "test@junior.com",
                "phone": "12345678",
                "location": "Ciudad de Prueba"
            },
            "professional_summary": "Ingeniero Junior enfocado en pruebas automatizadas.",
            "skills": ["Python", "Unit Testing"],
            "education": [
                {
                    "institution": "Universidad de Ingeniería",
                    "degree": "Ingeniería en Sistemas",
                    "period": "2020-2025",
                    "achievements": ["Promedio de 9.5"]
                }
            ],
            "experiences": [
                {
                    "company": "TestCorp",
                    "role": "QA Intern",
                    "period": "2023-2024",
                    "achievements": ["Desarrollé 10 scripts de prueba."]
                }
            ]
        }

        # Estructura simulada de respuesta de Gemini en JSON
        self.mock_llm_json_response = """{
            "full_name": "Ingeniero Junior de Prueba",
            "contact_info": {
                "email": "test@junior.com",
                "phone": "12345678",
                "location": "Ciudad de Prueba"
            },
            "professional_summary": "Ingeniero Junior altamente capacitado y enfocado en pruebas de software automatizadas avanzadas.",
            "optimized_skills": ["Python", "Unit Testing", "Pytest", "CI/CD"],
            "experiences": [
                {
                    "company": "TestCorp",
                    "role": "QA Intern (Optimizado)",
                    "period": "2023-2024",
                    "tailored_achievements": [
                        "Diseñé e implementé 10 scripts de prueba automatizados con un enfoque de código limpio.",
                        "Colaboré en la optimización de procesos de control de calidad."
                    ]
                }
            ],
            "education": [
                {
                    "institution": "Universidad de Ingeniería",
                    "degree": "Ingeniería en Sistemas",
                    "period": "2020-2025",
                    "achievements": ["Promedio sobresaliente de 9.5", "Proyecto integrador en pruebas."]
                }
            ]
        }"""

    def test_pydantic_schema_validation(self) -> None:
        """
        Verifica que el esquema de Pydantic valide correctamente un JSON correcto.
        """
        try:
            cv = OptimizedCV.model_validate_json(self.mock_llm_json_response)
            self.assertEqual(cv.full_name, "Ingeniero Junior de Prueba")
            self.assertEqual(len(cv.optimized_skills), 4)
            self.assertEqual(cv.experiences[0].company, "TestCorp")
        except ValidationError as exc:
            self.fail(f"La validación del esquema Pydantic falló inesperadamente: {exc}")

    def test_pydantic_schema_invalid_data(self) -> None:
        """
        Verifica que Pydantic capture errores cuando falten campos obligatorios.
        """
        invalid_json = '{"full_name": "Ingeniero Junior Incompleto"}'
        with self.assertRaises(ValidationError):
            OptimizedCV.model_validate_json(invalid_json)

    def test_load_profile_success(self) -> None:
        """
        Verifica la carga exitosa de un archivo YAML válido.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as temp_file:
            yaml.dump(self.sample_profile_data, temp_file, allow_unicode=True)
            temp_file_name = temp_file.name

        try:
            loaded_data = load_profile(temp_file_name)
            self.assertEqual(loaded_data["personal_info"]["full_name"], "Ingeniero Junior de Prueba")
            self.assertEqual(loaded_data["skills"][0], "Python")
        finally:
            os.remove(temp_file_name)

    def test_load_job_description_success(self) -> None:
        """
        Verifica la lectura exitosa de una descripción de puesto desde un archivo de texto.
        """
        sample_text = "Se busca desarrollador Python con experiencia en pruebas unitarias."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(sample_text)
            temp_file_name = temp_file.name

        try:
            loaded_text = load_job_description(temp_file_name)
            self.assertEqual(loaded_text, sample_text)
        finally:
            os.remove(temp_file_name)

    def test_generate_markdown_spanish(self) -> None:
        """
        Verifica que el generador de Markdown cree un string bien formateado en español.
        """
        cv = OptimizedCV.model_validate_json(self.mock_llm_json_response)
        markdown_output = generate_markdown(cv, lang="es")
        
        self.assertIn("# Ingeniero Junior de Prueba", markdown_output)
        self.assertIn("## Resumen Profesional", markdown_output)
        self.assertIn("QA Intern (Optimizado)", markdown_output)
        self.assertIn("Universidad de Ingeniería", markdown_output)

    def test_generate_markdown_english(self) -> None:
        """
        Verifica que el generador de Markdown use las cabeceras en inglés si se selecciona dicho idioma.
        """
        cv = OptimizedCV.model_validate_json(self.mock_llm_json_response)
        markdown_output = generate_markdown(cv, lang="en")
        
        self.assertIn("# Ingeniero Junior de Prueba", markdown_output)
        self.assertIn("## Professional Summary", markdown_output)
        self.assertIn("## Professional Experience", markdown_output)
        self.assertIn("## Education & Academic Projects", markdown_output)

    def test_generate_html_premium(self) -> None:
        """
        Verifica que el generador de HTML premium cree un archivo de marcado estructurado con estilos CSS y soporte A4.
        """
        cv = OptimizedCV.model_validate_json(self.mock_llm_json_response)
        html_output = generate_html(cv, lang="es")
        
        self.assertIn("<!DOCTYPE html>", html_output)
        self.assertIn('<html lang="es">', html_output)
        self.assertIn('<div class="resume-container">', html_output)
        self.assertIn("<h2>Resumen Profesional</h2>", html_output)
        self.assertIn("Ingeniero Junior de Prueba", html_output)
        # Comprobar la existencia de estilos de impresión claves
        self.assertIn("@media print", html_output)
        self.assertIn("page-break-inside: avoid", html_output)

    @patch("services.mock_interview.genai.Client")
    def test_mock_interview_runs_with_memory_and_exports_transcript(self, mock_client_class) -> None:
        """
        Verifica que el simulador mantenga historial, limite preguntas y exporte la transcripcion.
        """
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            MagicMock(text="Pregunta 1 sobre Python."),
            MagicMock(text="Pregunta 2 sobre Unit Testing."),
            MagicMock(text="Feedback final: buen dominio tecnico."),
        ]
        mock_client_class.return_value = mock_client

        service = MockInterviewService(
            self.sample_profile_data,
            "Se busca desarrollador con Python y Unit Testing.",
            api_key="fake_api_key",
            max_questions=2,
        )
        answers = iter(["Uso Python para automatizar pruebas.", "Diseno casos unitarios con fixtures."])
        output_lines = []

        service.run_interactive(input_func=lambda _: next(answers), output_func=output_lines.append)

        self.assertEqual(service.questions_asked, 2)
        self.assertEqual(mock_client.models.generate_content.call_count, 3)
        self.assertIn("SOLO", service.system_instruction)
        self.assertEqual(
            [message["role"] for message in service.messages],
            ["user", "model", "user", "model", "user", "model"],
        )
        self.assertEqual(service.transcript[-1]["speaker"], "Feedback final")

        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_path = os.path.join(temp_dir, "interview_transcript.md")
            saved_path = service.export_transcript(transcript_path)

            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, "r", encoding="utf-8") as transcript_file:
                transcript = transcript_file.read()

            self.assertIn("# Mock Interview Transcript", transcript)
            self.assertIn("Pregunta 1 sobre Python.", transcript)
            self.assertIn("Feedback final: buen dominio tecnico.", transcript)

    def test_mock_interview_rejects_more_than_seven_questions(self) -> None:
        """
        Verifica que el servicio no permita exceder el maximo de 7 preguntas.
        """
        with self.assertRaises(ValueError):
            MockInterviewService(
                self.sample_profile_data,
                "Se busca desarrollador con Python.",
                api_key="fake_api_key",
                max_questions=8,
            )

    @patch("services.robustness_judge.genai.Client")
    def test_robustness_judge_audits_and_exports_structured_report(self, mock_client_class) -> None:
        """
        Verifica que el juez de robustez parsea JSON estructurado y exporta el reporte.
        """
        mock_report_json = """{
            "score_honestidad": 72,
            "alucinaciones_detectadas": [
                {
                    "linea_cv": "optimized_skills[2]",
                    "dato_inventado": "Pytest no aparece en el perfil original",
                    "severidad": "media"
                }
            ],
            "inconsistencias_detectadas": [],
            "violaciones_eticas": [],
            "comentario_auditor": "El CV es util, pero debe eliminar habilidades no soportadas.",
            "aprobado": false
        }"""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text=mock_report_json)
        mock_client_class.return_value = mock_client

        optimized_cv = OptimizedCV.model_validate_json(self.mock_llm_json_response)
        service = RobustnessJudgeService(
            self.sample_profile_data,
            "Se busca desarrollador con Python y Unit Testing.",
            optimized_cv,
            api_key="fake_api_key",
        )

        report = service.audit()

        self.assertIsInstance(report, ReporteRobustez)
        self.assertEqual(report.score_honestidad, 72)
        self.assertFalse(report.aprobado)
        self.assertEqual(report.alucinaciones_detectadas[0].severidad, "media")
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_path = os.path.join(temp_dir, "robustness_report.md")
            json_path = os.path.join(temp_dir, "robustness_report.json")
            saved_markdown, saved_json = service.export_report(report, markdown_path, json_path)

            self.assertTrue(os.path.exists(saved_markdown))
            self.assertTrue(os.path.exists(saved_json))
            with open(saved_markdown, "r", encoding="utf-8") as report_file:
                markdown_report = report_file.read()

            self.assertIn("# Robustness Report", markdown_report)
            self.assertIn("Pytest no aparece", markdown_report)
            self.assertIn("Honesty score", markdown_report)

    def test_robustness_report_rejects_invalid_score(self) -> None:
        """
        Verifica que el esquema Pydantic rechace scores fuera del rango 0-100.
        """
        invalid_report = """{
            "score_honestidad": 120,
            "alucinaciones_detectadas": [],
            "inconsistencias_detectadas": [],
            "violaciones_eticas": [],
            "comentario_auditor": "Score invalido.",
            "aprobado": false
        }"""

        with self.assertRaises(ValidationError):
            ReporteRobustez.model_validate_json(invalid_report)

    @patch("google.genai.Client")
    def test_optimize_cv_with_mock_client(self, mock_client_class) -> None:
        """
        Prueba la función de optimización simulando la respuesta de la API de Gemini.
        """
        # Configurar variables de entorno ficticias para el test
        os.environ["GEMINI_API_KEY"] = "mock_api_key_for_testing"

        # Simular el cliente de GenAI y su respuesta
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = self.mock_llm_json_response
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        try:
            cv = optimize_cv(self.sample_profile_data, "Job Description", lang="es")
            self.assertIsInstance(cv, OptimizedCV)
            self.assertEqual(cv.full_name, "Ingeniero Junior de Prueba")
            self.assertEqual(cv.experiences[0].role, "QA Intern (Optimizado)")
        finally:
            # Limpiar la variable de entorno
            del os.environ["GEMINI_API_KEY"]


if __name__ == "__main__":
    unittest.main()
