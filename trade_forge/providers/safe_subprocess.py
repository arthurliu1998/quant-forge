"""subprocess.run() wrapper that strips sensitive environment variables."""
import os
import subprocess

from trade_forge.secrets import SecretManager


def safe_run(cmd, **kwargs):
    """subprocess.run() with sensitive env vars stripped from child process."""
    env = kwargs.pop('env', None)
    if env is None:
        env = os.environ.copy()
    clean_env = {k: v for k, v in env.items() if k not in SecretManager.SECRETS}
    return subprocess.run(cmd, env=clean_env, **kwargs)
