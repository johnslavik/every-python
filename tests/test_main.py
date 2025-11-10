import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock, call
import subprocess

from every_python.main import (
    app,
    ensure_repo,
    resolve_ref,
    build_python,
    REPO_DIR,
    BUILDS_DIR,
)

runner = CliRunner()


class TestEnsureRepo:
    """Test repository initialization."""

    @patch("subprocess.run")
    def test_clone_if_not_exists(self, mock_run, tmp_path):
        """Test that repo is cloned if it doesn't exist."""
        repo_dir = tmp_path / "cpython"

        with patch("every_python.main.REPO_DIR", repo_dir):
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            ensure_repo()

            # Should have called git clone
            assert any(
                "clone" in str(call_args) for call_args in mock_run.call_args_list
            )

    def test_skip_clone_if_exists(self, tmp_path):
        """Test that clone is skipped if repo already exists."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)

        with patch("every_python.main.REPO_DIR", repo_dir), patch(
            "subprocess.run"
        ) as mock_run:
            ensure_repo()

            # Should not have called git clone
            mock_run.assert_not_called()


class TestResolveRef:
    """Test git ref resolution."""

    @patch("subprocess.run")
    def test_resolve_main(self, mock_run, tmp_path):
        """Test resolving 'main' ref."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)

        with patch("every_python.main.REPO_DIR", repo_dir):
            mock_run.return_value = Mock(
                returncode=0, stdout="abc123def456\n", stderr=""
            )
            commit = resolve_ref("main")

            assert commit == "abc123def456"
            assert "rev-parse" in str(mock_run.call_args)

    @patch("subprocess.run")
    def test_resolve_tag(self, mock_run, tmp_path):
        """Test resolving version tag."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)

        with patch("every_python.main.REPO_DIR", repo_dir):
            mock_run.return_value = Mock(
                returncode=0, stdout="def456abc123\n", stderr=""
            )
            commit = resolve_ref("v3.13.0")

            assert commit == "def456abc123"

    @patch("subprocess.run")
    def test_resolve_invalid_ref(self, mock_run, tmp_path):
        """Test resolving invalid ref raises error."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)

        with patch("every_python.main.REPO_DIR", repo_dir):
            mock_run.return_value = Mock(returncode=1, stdout="", stderr="fatal: bad")

            # typer.Exit is a subclass of click.exceptions.Exit
            from click.exceptions import Exit
            with pytest.raises(Exit):
                resolve_ref("invalid-ref")


class TestBuildPython:
    """Test Python building logic."""

    @patch("every_python.main.get_llvm_version_for_commit")
    @patch("every_python.main.check_llvm_available")
    @patch("subprocess.run")
    def test_build_without_jit(
        self, mock_run, mock_llvm_check, mock_llvm_version, tmp_path
    ):
        """Test building Python without JIT."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)
        builds_dir = tmp_path / "builds"

        with patch("every_python.main.REPO_DIR", repo_dir), patch(
            "every_python.main.BUILDS_DIR", builds_dir
        ):
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            build_dir = build_python("abc123d", enable_jit=False)

            # Should create non-JIT build directory
            assert build_dir.name == "abc123d"
            assert not build_dir.name.endswith("-jit")

            # Should NOT check for LLVM
            mock_llvm_version.assert_not_called()
            mock_llvm_check.assert_not_called()

    @patch("every_python.main.ensure_repo")
    @patch("subprocess.run")
    @patch("every_python.main.check_llvm_available")
    @patch("every_python.main.get_llvm_version_for_commit")
    def test_build_with_jit_available(
        self, mock_llvm_version, mock_llvm_check, mock_run, mock_ensure_repo, tmp_path
    ):
        """Test building Python with JIT when LLVM is available."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)
        builds_dir = tmp_path / "builds"

        with patch("every_python.main.REPO_DIR", repo_dir), patch(
            "every_python.main.BUILDS_DIR", builds_dir
        ):
            mock_llvm_version.return_value = "20"
            mock_llvm_check.return_value = True
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            build_dir = build_python("abc123d", enable_jit=True)

            # Should create JIT build directory
            assert build_dir.name == "abc123d-jit"

            # Should check for LLVM
            mock_llvm_version.assert_called_once_with("abc123d", repo_dir)
            mock_llvm_check.assert_called_once_with("20")

            # Should pass --enable-experimental-jit to configure
            configure_calls = [
                call_args
                for call_args in mock_run.call_args_list
                if "./configure" in str(call_args)
            ]
            assert len(configure_calls) > 0
            assert "--enable-experimental-jit" in str(configure_calls[0])

    @patch("every_python.main.get_llvm_version_for_commit")
    @patch("every_python.main.check_llvm_available")
    @patch("subprocess.run")
    @patch("typer.confirm")
    def test_build_with_jit_llvm_missing(
        self, mock_confirm, mock_run, mock_llvm_check, mock_llvm_version, tmp_path
    ):
        """Test building with JIT when LLVM is missing falls back to non-JIT."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)
        builds_dir = tmp_path / "builds"

        with patch("every_python.main.REPO_DIR", repo_dir), patch(
            "every_python.main.BUILDS_DIR", builds_dir
        ):
            mock_llvm_version.return_value = "20"
            mock_llvm_check.return_value = False  # LLVM not available
            mock_confirm.return_value = True  # User confirms to build without JIT
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            build_dir = build_python("abc123d", enable_jit=True)

            # Should fall back to non-JIT build
            assert build_dir.name == "abc123d"
            assert not build_dir.name.endswith("-jit")

            # Should have prompted user
            mock_confirm.assert_called_once()

    @patch("every_python.main.get_llvm_version_for_commit")
    @patch("typer.confirm")
    @patch("subprocess.run")
    def test_build_with_jit_not_available_in_commit(
        self, mock_run, mock_confirm, mock_llvm_version, tmp_path
    ):
        """Test building with JIT when JIT not available in commit."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)
        builds_dir = tmp_path / "builds"

        with patch("every_python.main.REPO_DIR", repo_dir), patch(
            "every_python.main.BUILDS_DIR", builds_dir
        ):
            mock_llvm_version.return_value = None  # JIT not in this commit
            mock_confirm.return_value = True
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            build_dir = build_python("abc123d", enable_jit=True)

            # Should fall back to non-JIT build
            assert build_dir.name == "abc123d"
            mock_confirm.assert_called_once()

    @patch("subprocess.run")
    def test_build_already_exists(self, mock_run, tmp_path):
        """Test that existing build is reused."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)
        builds_dir = tmp_path / "builds"
        builds_dir.mkdir(parents=True)

        # Create existing build
        existing_build = builds_dir / "abc123d"
        existing_build.mkdir()

        with patch("every_python.main.REPO_DIR", repo_dir), patch(
            "every_python.main.BUILDS_DIR", builds_dir
        ):
            build_dir = build_python("abc123d", enable_jit=False)

            # Should return existing build without running any git/configure/make
            assert build_dir == existing_build
            # git fetch might be called, but not configure/make
            configure_calls = [
                call_args
                for call_args in mock_run.call_args_list
                if "./configure" in str(call_args)
            ]
            assert len(configure_calls) == 0


class TestInstallCommand:
    """Test the install command."""

    @patch("every_python.main.build_python")
    @patch("every_python.main.resolve_ref")
    def test_install_main(self, mock_resolve, mock_build, tmp_path):
        """Test installing main branch."""
        mock_resolve.return_value = "abc123def456"
        mock_build.return_value = tmp_path / "abc123def456"

        result = runner.invoke(app, ["install", "main"])

        assert result.exit_code == 0
        assert "abc123d" in result.stdout
        mock_resolve.assert_called_once_with("main")
        mock_build.assert_called_once_with("abc123def456", enable_jit=False, verbose=False)

    @patch("every_python.main.build_python")
    @patch("every_python.main.resolve_ref")
    def test_install_with_jit(self, mock_resolve, mock_build, tmp_path):
        """Test installing with JIT flag."""
        mock_resolve.return_value = "abc123def456"
        mock_build.return_value = tmp_path / "abc123def456-jit"

        result = runner.invoke(app, ["install", "main", "--jit"])

        assert result.exit_code == 0
        mock_build.assert_called_once_with("abc123def456", enable_jit=True, verbose=False)

    @patch("every_python.main.build_python")
    @patch("every_python.main.resolve_ref")
    def test_install_verbose(self, mock_resolve, mock_build, tmp_path):
        """Test installing with verbose flag."""
        mock_resolve.return_value = "abc123def456"
        mock_build.return_value = tmp_path / "abc123def456"

        result = runner.invoke(app, ["install", "main", "--verbose"])

        assert result.exit_code == 0
        mock_build.assert_called_once_with("abc123def456", enable_jit=False, verbose=True)


class TestRunCommand:
    """Test the run command."""

    @patch("os.execv")
    @patch("every_python.main.resolve_ref")
    def test_run_existing_build(self, mock_resolve, mock_execv, tmp_path):
        """Test running with existing build."""
        mock_resolve.return_value = "abc123def456"

        builds_dir = tmp_path / "builds"
        build_dir = builds_dir / "abc123def456" / "bin"
        build_dir.mkdir(parents=True)
        python_bin = build_dir / "python3"
        python_bin.touch()

        with patch("every_python.main.BUILDS_DIR", builds_dir):
            runner.invoke(app, ["run", "main", "--", "python", "--version"])

            # Should call execv with the python binary
            mock_execv.assert_called_once()
            args = mock_execv.call_args[0]
            assert "python3" in args[0]

    @patch("every_python.main.build_python")
    @patch("every_python.main.resolve_ref")
    @patch("os.execv")
    def test_run_triggers_build(self, mock_execv, mock_resolve, mock_build, tmp_path):
        """Test that run triggers build if not found."""
        mock_resolve.return_value = "abc123def456"

        builds_dir = tmp_path / "builds"
        builds_dir.mkdir(parents=True)
        # Don't create build_dir - we want to test that it triggers a build

        # After build_python is called, it will return this
        build_dir = builds_dir / "abc123def456"

        def create_build_on_call(*args, **kwargs):
            # Simulate build_python creating the directory
            build_dir.mkdir(parents=True)
            (build_dir / "bin").mkdir()
            (build_dir / "bin" / "python3").touch()
            return build_dir

        mock_build.side_effect = create_build_on_call

        with patch("every_python.main.BUILDS_DIR", builds_dir):
            runner.invoke(app, ["run", "main", "--", "python", "--version"])

            # Should build the version
            mock_build.assert_called_once_with("abc123def456", enable_jit=False)


class TestListBuildsCommand:
    """Test the list-builds command."""

    def test_list_no_builds(self, tmp_path):
        """Test listing when no builds exist."""
        builds_dir = tmp_path / "builds"

        with patch("every_python.main.BUILDS_DIR", builds_dir):
            result = runner.invoke(app, ["list-builds"])

            assert result.exit_code == 0
            assert "No builds found" in result.stdout

    @patch("subprocess.run")
    def test_list_with_builds(self, mock_run, tmp_path):
        """Test listing with existing builds."""
        builds_dir = tmp_path / "builds"
        builds_dir.mkdir(parents=True)

        # Create mock builds
        build1 = builds_dir / "abc123d"
        build1_bin = build1 / "bin"
        build1_bin.mkdir(parents=True)
        (build1_bin / "python3").touch()

        build2 = builds_dir / "def456a-jit"
        build2_bin = build2 / "bin"
        build2_bin.mkdir(parents=True)
        (build2_bin / "python3").touch()

        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)

        # Mock subprocess calls for version and git info
        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if "--version" in cmd:
                return Mock(returncode=0, stdout="Python 3.14.0a1+", stderr="")
            elif "git" in cmd and "log" in cmd:
                return Mock(returncode=0, stdout="1234567890|Test commit", stderr="")
            return Mock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = mock_run_side_effect

        with patch("every_python.main.BUILDS_DIR", builds_dir), patch(
            "every_python.main.REPO_DIR", repo_dir
        ):
            result = runner.invoke(app, ["list-builds"])

            assert result.exit_code == 0
            assert "abc123d" in result.stdout
            assert "def456a" in result.stdout


class TestCleanCommand:
    """Test the clean command."""

    def test_clean_specific_build(self, tmp_path):
        """Test cleaning a specific build."""
        builds_dir = tmp_path / "builds"
        builds_dir.mkdir(parents=True)

        # Create builds - use the full commit hash that resolve_ref will return
        (builds_dir / "abc123def456").mkdir()
        (builds_dir / "abc123def456-jit").mkdir()
        (builds_dir / "def456a").mkdir()

        with patch("every_python.main.BUILDS_DIR", builds_dir), patch(
            "every_python.main.resolve_ref", return_value="abc123def456"
        ):
            result = runner.invoke(app, ["clean", "main"])

            assert result.exit_code == 0
            # Both JIT and non-JIT should be removed
            assert "Removed" in result.stdout

    def test_clean_all(self, tmp_path):
        """Test cleaning all builds."""
        builds_dir = tmp_path / "builds"
        builds_dir.mkdir(parents=True)
        (builds_dir / "abc123d").mkdir()
        (builds_dir / "def456a").mkdir()

        with patch("every_python.main.BUILDS_DIR", builds_dir):
            result = runner.invoke(app, ["clean", "--all"])

            assert result.exit_code == 0
            assert "Removed all builds" in result.stdout
            assert not builds_dir.exists()


class TestBisectCommand:
    """Test the bisect command."""

    @patch("subprocess.run")
    @patch("every_python.main.resolve_ref")
    @patch("every_python.main.build_python")
    def test_bisect_basic(self, mock_build, mock_resolve, mock_run, tmp_path):
        """Test basic bisect functionality."""
        repo_dir = tmp_path / "cpython"
        repo_dir.mkdir(parents=True)
        (repo_dir / ".git").mkdir()

        builds_dir = tmp_path / "builds"

        def resolve_side_effect(ref):
            if ref == "good-ref":
                return "abc123d"
            elif ref == "bad-ref":
                return "def456a"
            return "current123"

        mock_resolve.side_effect = resolve_side_effect

        # Create mock build
        build_dir = builds_dir / "current123"
        build_dir.mkdir(parents=True)
        (build_dir / "bin").mkdir()
        (build_dir / "bin" / "python3").touch()
        mock_build.return_value = build_dir

        # Mock git commands
        call_count = [0]

        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])

            # Git bisect commands
            if "bisect" in cmd and "start" in cmd:
                return Mock(returncode=0, stdout="", stderr="")
            elif "bisect" in cmd and "good" in cmd:
                call_count[0] += 1
                if call_count[0] == 1:
                    return Mock(
                        returncode=0,
                        stdout="Bisecting: 5 revisions left (roughly 3 steps)\n",
                        stderr="",
                    )
                return Mock(
                    returncode=0,
                    stdout="abc123 is the first bad commit\n",
                    stderr="",
                )
            elif "bisect" in cmd and "bad" in cmd:
                return Mock(returncode=0, stdout="", stderr="")
            elif "bisect" in cmd and "reset" in cmd:
                return Mock(returncode=0, stdout="", stderr="")
            elif "bisect" in cmd and "log" in cmd:
                return Mock(
                    returncode=0,
                    stdout="# first bad commit: [abc123]\n",
                    stderr="",
                )
            elif "rev-parse" in cmd and "HEAD" in cmd:
                return Mock(returncode=0, stdout="current123\n", stderr="")
            elif "git" in cmd and "show" in cmd:
                return Mock(
                    returncode=0,
                    stdout="abc123\nAuthor <email>\nDate\nCommit message",
                    stderr="",
                )
            elif "reset" in cmd:
                return Mock(returncode=0, stdout="", stderr="")
            elif "clean" in cmd:
                return Mock(returncode=0, stdout="", stderr="")

            return Mock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = mock_run_side_effect

        with patch("every_python.main.REPO_DIR", repo_dir), patch(
            "every_python.main.BUILDS_DIR", builds_dir
        ), patch("pathlib.Path.cwd", return_value=tmp_path), patch(
            "pathlib.Path.exists", return_value=False
        ):  # Simulate BISECT_LOG not existing
            result = runner.invoke(
                app,
                [
                    "bisect",
                    "--good",
                    "good-ref",
                    "--bad",
                    "bad-ref",
                    "--run",
                    "exit 0",
                ],
            )

            # Should complete successfully
            assert "Bisect complete" in result.stdout or result.exit_code == 0
