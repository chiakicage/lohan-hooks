import os
import logging
import json
from typing import List
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray.runtime_env.runtime_env import RuntimeEnv


class NumaPlugin(RuntimeEnvPlugin):
    name = "numa"

    def modify_context(
        self,
        uris: List[str],
        runtime_env: "RuntimeEnv",  # noqa: F821
        context: RuntimeEnvContext,
        logger: logging.Logger,
    ) -> None:
        """Modify context to change worker startup behavior.

        For example, you can use this to preprend "cd <dir>" command to worker
        startup, or add new environment variables.

        Args:
            uris: The URIs used by this resource.
            runtime_env: The RuntimeEnv object.
            context: Auxiliary information supplied by Ray.
            logger: A logger to log messages during the context modification.
        """
        node = runtime_env.get(self.name)
        if node:
            context.py_executable = f"numactl --cpunodebind={node} --membind={node} {context.py_executable}"
        return


def register_ray_plugins():
    plugins = [
        {"class": f"{__name__}.NumaPlugin"},
    ]
    os.environ["RAY_RUNTIME_ENV_PLUGINS"] = json.dumps(plugins)
