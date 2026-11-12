import contextlib
import importlib.metadata
import io
import shutil
import subprocess
import sys
from unittest import mock

from support import PROJECT_SOURCE, ProjectTestCase, fake_torch


class LauncherAndLayoutTests(ProjectTestCase):
    def launcher_text(self, name):
        data = (PROJECT_SOURCE / name).read_bytes()
        self.assertNotIn(b"\n", data.replace(b"\r\n", b""), "Windows launcher lost CRLF line endings")
        return data.decode("utf-8").replace("\r\n", "\n")

    def test_run_bat_forwards_arguments_and_immediately_captures_exit_code(self):
        text = self.launcher_text("run.bat")
        self.assertIn('python src\\transcript.py %*\nset "EXIT_CODE=%errorlevel%"', text)
        self.assertIn("pause\nexit /b %EXIT_CODE%", text)
        self.assertNotIn("exit /b 0", text)

    def test_installer_keeps_explicit_cuda_first_installation_order(self):
        text = self.launcher_text("install.bat")
        cuda = "-m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
        requirements = "-m pip install -r requirements\\requirements.txt"
        self.assertIn(cuda, text)
        self.assertIn(requirements, text)
        self.assertLess(text.index(cuda), text.index(requirements))
        self.assertIn("sys.version_info[:2] == (3, 12)", text)
        self.assertIn("Delete .venv manually", text)
        self.assertNotRegex(text.lower(), r"(?m)^\s*(rmdir|rd|del)\s")

    def verification_script(self):
        lines = self.launcher_text("install.bat").splitlines()
        line = next(line for line in lines if ' -c "' in line and "torch.version.cuda" in line)
        return line.split(' -c "', 1)[1][:-1]

    def execute_verifier(self, *, available, cuda_version):
        runtime = fake_torch(available=available, cuda_version=cuda_version)
        output = io.StringIO()
        with mock.patch.dict(sys.modules, {"torch": runtime}):
            with mock.patch.object(importlib.metadata, "version", return_value="3.8.6"):
                with contextlib.redirect_stdout(output):
                    try:
                        exec(compile(self.verification_script(), "install.bat verifier", "exec"), {})
                    except SystemExit as result:
                        return result.code, output.getvalue()
        self.fail("verifier must exit with its status")

    def test_installer_embedded_python_rejects_cpu_only_wheel(self):
        code, output = self.execute_verifier(available=False, cuda_version=None)
        self.assertEqual(code, 1)
        self.assertIn("[ERROR]", output)
        self.assertIn("CPU-only", output)

    def test_installer_embedded_python_warns_but_succeeds_without_gpu(self):
        code, output = self.execute_verifier(available=False, cuda_version="12.8")
        self.assertEqual(code, 0)
        self.assertIn("[WARNING]", output)
        self.assertIn('device = "cpu"', output)

    def test_installer_embedded_python_succeeds_with_gpu(self):
        code, output = self.execute_verifier(available=True, cuda_version="12.8")
        self.assertEqual(code, 0)
        self.assertIn("CUDA GPU ready", output)

    def test_embedded_venv_version_check(self):
        text = self.launcher_text("install.bat")
        line = next(line for line in text.splitlines() if ' -c "' in line and "sys.version_info" in line)
        script = line.split(' -c "', 1)[1][:-1]
        for version, expected in (((3, 12, 7), 0), ((3, 11, 9), 1), ((3, 13, 0), 1)):
            with self.subTest(version=version), mock.patch.object(sys, "version_info", version):
                with self.assertRaises(SystemExit) as result:
                    exec(compile(script, "install.bat venv check", "exec"), {})
                self.assertEqual(result.exception.code, expected)

    def test_requirements_moved_without_duplicate_or_unqualified_torch(self):
        self.assertFalse((PROJECT_SOURCE / "requirements.txt").exists())
        text = (PROJECT_SOURCE / "requirements/requirements.txt").read_text(encoding="utf-8")
        dependencies = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
        self.assertEqual(dependencies, [
            "whisperx==3.8.6", "rich", 'nvidia-cublas-cu12 ; platform_system == "Windows"',
            'nvidia-cudnn-cu12  ; platform_system == "Windows"',
        ])
        self.assertNotRegex("\n".join(dependencies), r"(?m)^(torch|torchaudio)\b")

    def test_documentation_has_no_obsolete_prompt_or_root_setup_references(self):
        text = (PROJECT_SOURCE / "docs/README.md").read_text(encoding="utf-8")
        for obsolete in ("prompts/", "UK-llm-ai-processing-prompt.txt", "EN-llm-ai-processing-prompt.txt", "bundled system prompts"):
            with self.subTest(obsolete=obsolete):
                self.assertNotIn(obsolete, text)
        self.assertNotRegex(text, r"(?m)^├── SETUP-NOTES\.txt")
        self.assertIn("docs/SETUP-NOTES.txt", text)
        self.assertIn("requirements/requirements.txt", text)

    def test_ignore_rules_in_temporary_git_repository(self):
        if shutil.which("git") is None:
            self.skipTest("Git executable unavailable")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        (self.root / ".gitignore").write_bytes((PROJECT_SOURCE / ".gitignore").read_bytes())
        cases = {
            "config/config.toml": True,
            "input/a.mp3": True,
            "input/course/a.wav": True,
            "output/a/timecode-a--large-v3-turbo.txt": True,
            "output/a/.whisperblocks-source": True,
            "input/.gitkeep": False,
            "output/.gitkeep": False,
            "requirements/requirements.txt": False,
            "docs/SETUP-NOTES.txt": False,
            "tests/": False,
            "tests/test_config.py": False,
            "config/config.toml.example": False,
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "--no-index", "-q", path], cwd=self.root,
                    capture_output=True, text=True,
                )
                self.assertIn(result.returncode, (0, 1), result.stderr)
                self.assertEqual(result.returncode == 0, expected, result.stderr)
