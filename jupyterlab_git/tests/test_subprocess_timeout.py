"""
Tests for subprocess timeout handling and process cleanup functionality.

This module tests the improvements made to git.py for:
1. Subprocess timeout handling with graceful termination
2. Process cleanup with terminate → wait → kill pattern
3. Credential daemon cleanup with proper timeout handling
4. Logging behavior for timeout and error scenarios
"""

import asyncio
import logging
import subprocess
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from pathlib import Path

import pexpect
import pytest

from jupyterlab_git.git import Git, call_subprocess_with_authentication


class TestSubprocessTimeout:
    """Test subprocess timeout handling in call_subprocess function."""

    def test_subprocess_timeout_terminates_gracefully(self):
        """Test that subprocess timeout follows terminate → wait → kill pattern."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.side_effect = subprocess.TimeoutExpired(
                cmd=["git", "status"], timeout=5
            )
            mock_process.wait.return_value = 0  # Graceful termination
            mock_popen.return_value = mock_process

            # Import after patching to ensure we get the patched version
            from jupyterlab_git.git import call_subprocess

            with pytest.raises(
                TimeoutError, match="Git command timed out after 5 seconds"
            ):
                call_subprocess(["git", "status"], timeout=5)

            # Verify graceful termination sequence
            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called_once_with(timeout=5)
            mock_process.kill.assert_not_called()

    def test_subprocess_timeout_force_kills_after_wait_timeout(self):
        """Test that subprocess force kills if process doesn't terminate gracefully."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.communicate.side_effect = subprocess.TimeoutExpired(
                cmd=["git", "status"], timeout=5
            )
            # First wait() call (with timeout=5) should timeout
            # Second wait() call (without timeout) should succeed
            mock_process.wait.side_effect = [
                subprocess.TimeoutExpired(cmd=["git", "status"], timeout=5),
                0,  # Second call succeeds
            ]
            mock_popen.return_value = mock_process

            from jupyterlab_git.git import call_subprocess

            with pytest.raises(
                TimeoutError, match="Git command timed out after 5 seconds"
            ):
                call_subprocess(["git", "status"], timeout=5)

            # Verify force kill sequence
            mock_process.terminate.assert_called_once()
            assert mock_process.wait.call_count == 2
            mock_process.wait.assert_any_call(timeout=5)
            mock_process.kill.assert_called_once()


class TestPexpectTimeout:
    """Test pexpect timeout handling in call_subprocess_with_authentication."""

    @pytest.mark.asyncio
    async def test_pexpect_timeout_uses_proper_timeout_parameter(self):
        """Test that pexpect.spawn uses the timeout parameter correctly."""
        with patch("pexpect.spawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.expect.side_effect = pexpect.exceptions.TIMEOUT("timeout")
            mock_spawn.return_value = mock_process

            with pytest.raises(
                TimeoutError, match="Git authentication timed out after 10 seconds"
            ):
                await call_subprocess_with_authentication(
                    ["git", "clone", "https://example.com/repo.git"],
                    username="user",
                    password="pass",
                    timeout=10,
                )

            # Verify pexpect.spawn was called with correct timeout
            mock_spawn.assert_called_once()
            call_args = mock_spawn.call_args
            assert call_args[1]["timeout"] == 10

    @pytest.mark.asyncio
    async def test_pexpect_timeout_cleans_up_process(self):
        """Test that pexpect timeout properly cleans up the process."""
        with patch("pexpect.spawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.expect.side_effect = pexpect.exceptions.TIMEOUT("timeout")
            mock_spawn.return_value = mock_process

            with pytest.raises(TimeoutError):
                await call_subprocess_with_authentication(
                    ["git", "clone", "https://example.com/repo.git"],
                    username="user",
                    password="pass",
                    timeout=10,
                )

            # Verify process cleanup
            mock_process.terminate.assert_called_once_with(force=True)
            mock_process.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_pexpect_exception_cleanup(self):
        """Test that any exception in pexpect properly cleans up the process."""
        with patch("pexpect.spawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.isalive.return_value = True
            mock_process.expect.side_effect = RuntimeError("Unexpected error")
            mock_spawn.return_value = mock_process

            with pytest.raises(RuntimeError, match="Unexpected error"):
                await call_subprocess_with_authentication(
                    ["git", "clone", "https://example.com/repo.git"],
                    username="user",
                    password="pass",
                    timeout=10,
                )

            # Verify process cleanup on exception
            mock_process.terminate.assert_called_once_with(force=True)
            mock_process.close.assert_called_once()


class TestGitCredentialDaemonCleanup:
    """Test Git credential daemon cleanup functionality."""

    def test_credential_daemon_cleanup_on_destruction(self):
        """Test that credential daemon is cleaned up when Git instance is destroyed."""
        with patch("subprocess.Popen") as mock_popen:
            mock_daemon = MagicMock()
            mock_daemon.poll.return_value = None  # Process is still running
            mock_daemon.wait.return_value = 0  # Graceful termination
            mock_daemon.pid = 12345

            git = Git()
            git._GIT_CREDENTIAL_CACHE_DAEMON_PROCESS = mock_daemon

            # Trigger cleanup
            git._cleanup_processes()

            # Verify graceful termination sequence
            mock_daemon.terminate.assert_called_once()
            mock_daemon.wait.assert_called_once_with(timeout=5)
            mock_daemon.kill.assert_not_called()

    def test_credential_daemon_force_kill_after_timeout(self, caplog):
        """Test that credential daemon is force killed if it doesn't terminate gracefully."""
        with patch("subprocess.Popen") as mock_popen:
            mock_daemon = MagicMock()
            mock_daemon.poll.return_value = None  # Process is still running
            # First wait() call (with timeout=5) should timeout
            # Second wait() call (with timeout=1) should also timeout to test the fallback
            mock_daemon.wait.side_effect = [
                subprocess.TimeoutExpired(cmd=["git-credential-cache"], timeout=5),
                subprocess.TimeoutExpired(cmd=["git-credential-cache"], timeout=1),
            ]
            mock_daemon.pid = 12345

            git = Git()
            git._GIT_CREDENTIAL_CACHE_DAEMON_PROCESS = mock_daemon

            git._cleanup_processes()

            # Verify force kill sequence
            mock_daemon.terminate.assert_called_once()
            assert mock_daemon.wait.call_count == 2
            mock_daemon.wait.assert_any_call(timeout=5)
            mock_daemon.wait.assert_any_call(timeout=1)
            mock_daemon.kill.assert_called_once()

    def test_credential_daemon_cleanup_logging(self, caplog):
        """Test that credential daemon cleanup uses proper logging."""
        # Mock the logger to verify it's called
        with patch("subprocess.Popen") as mock_popen, patch(
            "jupyterlab_git.git.get_logger"
        ) as mock_get_logger:

            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            mock_daemon = MagicMock()
            mock_daemon.poll.return_value = None  # Process is still running
            mock_daemon.wait.return_value = 0
            mock_daemon.pid = 12345

            git = Git()
            git._GIT_CREDENTIAL_CACHE_DAEMON_PROCESS = mock_daemon

            git._cleanup_processes()

            # Verify logging message includes PID and uses lazy evaluation
            mock_logger.debug.assert_called_once_with(
                "Git credential cache daemon process (PID: %s) cleaned up successfully",
                12345,
            )

    def test_credential_daemon_cleanup_exception_handling(self, caplog):
        """Test that credential daemon cleanup handles exceptions gracefully."""
        # Mock the logger to verify it's called
        with patch("subprocess.Popen") as mock_popen, patch(
            "jupyterlab_git.git.get_logger"
        ) as mock_get_logger:

            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            mock_daemon = MagicMock()
            mock_daemon.poll.return_value = None
            exception = OSError("Process not found")
            mock_daemon.terminate.side_effect = exception
            mock_daemon.pid = 12345

            git = Git()
            git._GIT_CREDENTIAL_CACHE_DAEMON_PROCESS = mock_daemon

            git._cleanup_processes()

            # Verify exception is logged
            mock_logger.warning.assert_called_once_with(
                "Failed to cleanup credential cache daemon: %s", exception
            )
            # Verify process is set to None even on exception
            assert git._GIT_CREDENTIAL_CACHE_DAEMON_PROCESS is None

    def test_credential_daemon_cleanup_skips_if_already_terminated(self):
        """Test that cleanup skips if daemon process is already terminated."""
        with patch("subprocess.Popen") as mock_popen:
            mock_daemon = MagicMock()
            mock_daemon.poll.return_value = 0  # Process already terminated

            git = Git()
            git._GIT_CREDENTIAL_CACHE_DAEMON_PROCESS = mock_daemon

            git._cleanup_processes()

            # Verify no termination calls since process is already done
            mock_daemon.terminate.assert_not_called()
            mock_daemon.wait.assert_not_called()
            mock_daemon.kill.assert_not_called()


class TestLoggingBehavior:
    """Test logging behavior improvements."""

    def test_timeout_error_includes_command_details(self, caplog):
        """Test that timeout errors include specific command details."""
        # Mock the logger to verify it's called
        with patch("subprocess.Popen") as mock_popen, patch(
            "jupyterlab_git.git.get_logger"
        ) as mock_get_logger:

            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            mock_process = MagicMock()
            mock_process.communicate.side_effect = subprocess.TimeoutExpired(
                cmd=["git", "clone", "https://example.com/repo.git"], timeout=30
            )
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            from jupyterlab_git.git import call_subprocess

            with pytest.raises(TimeoutError) as exc_info:
                call_subprocess(
                    ["git", "clone", "https://example.com/repo.git"], timeout=30
                )

            # Verify error message includes command and timeout details
            assert "Git command timed out after 30 seconds" in str(exc_info.value)
            assert "git clone https://example.com/repo.git" in str(exc_info.value)
            mock_logger.warning.assert_called_once_with(
                "Git command timed out: %s",
                ["git", "clone", "https://example.com/repo.git"],
            )


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple improvements."""

    @pytest.mark.asyncio
    async def test_authentication_timeout_with_cleanup_and_logging(self, caplog):
        """Test complete authentication timeout scenario with cleanup and logging."""
        with patch("pexpect.spawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.expect.side_effect = pexpect.exceptions.TIMEOUT("timeout")
            mock_spawn.return_value = mock_process

            with caplog.at_level(logging.WARNING, logger="Application.jupyterlab_git"):
                with pytest.raises(TimeoutError) as exc_info:
                    await call_subprocess_with_authentication(
                        ["git", "push", "origin", "main"],
                        username="user",
                        password="pass",
                        timeout=15,
                    )

            # Verify proper timeout error message
            assert "Git authentication timed out after 15 seconds" in str(
                exc_info.value
            )
            assert "git push origin main" in str(exc_info.value)

            # Verify process cleanup occurred
            mock_process.terminate.assert_called_once_with(force=True)
            mock_process.close.assert_called_once()

    def test_git_instance_full_lifecycle_with_cleanup(self, caplog):
        """Test full Git instance lifecycle with proper cleanup."""
        # Mock the logger to verify it's called
        with patch("subprocess.Popen") as mock_popen, patch(
            "jupyterlab_git.git.get_logger"
        ) as mock_get_logger:

            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            mock_daemon = MagicMock()
            mock_daemon.poll.return_value = None
            mock_daemon.wait.return_value = 0
            mock_daemon.pid = 54321

            git = Git()
            git._GIT_CREDENTIAL_CACHE_DAEMON_PROCESS = mock_daemon

            # Simulate destruction
            git.__del__()

            # Verify cleanup was called and logged properly
            mock_daemon.terminate.assert_called_once()
            mock_daemon.wait.assert_called_once_with(timeout=5)
            mock_logger.debug.assert_called_once_with(
                "Git credential cache daemon process (PID: %s) cleaned up successfully",
                54321,
            )
