import asyncio
import json
import logging
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import cloudpickle
import ray
from hypha_rpc.utils.schema import schema_method

from bioengine_worker import __version__
from bioengine_worker.ray_cluster import RayCluster
from bioengine_worker.utils import check_permissions, create_logger


@ray.remote
def ray_task(func, args, kwargs):
    """
    Pure Ray task for executing a Python function with arguments.

    This task runs in a Ray worker and captures stdout/stderr output.
    It handles both synchronous and asynchronous functions, returning
    the result or error information as a dictionary.

    Args:
        func: The Python function to execute.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.

    Returns:
        Dict containing:
            - result: Function return value (if successful)
            - error: Error message string (if failed)
            - traceback: Full Python traceback (if failed)
            - stdout: Captured stdout output as string
            - stderr: Captured stderr output as string

    Raises:
        Exception: If the function execution fails, the error message and
                    traceback are captured and returned in the result.
    """
    import asyncio
    import contextlib
    import io
    import traceback

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with (
        contextlib.redirect_stdout(stdout_buffer),
        contextlib.redirect_stderr(stderr_buffer),
    ):
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            return {
                "result": result,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
            }


class CodeExecutor:
    """
    Secure Python code execution service for distributed Ray environments.

    The CodeExecutor provides a secure, isolated environment for executing arbitrary
    Python code in distributed Ray tasks with comprehensive resource management,
    output streaming, and error handling capabilities. It supports both source code
    execution and pre-serialized function execution for maximum flexibility.

    Key Features:
    - Admin-only access control with permission validation
    - Isolated execution in Ray tasks with configurable resources
    - Real-time stdout/stderr streaming to clients
    - Timeout protection and comprehensive error handling
    - Support for both source code strings and pickled functions
    - Configurable resource allocation (CPU, GPU, memory)

    Security Model:
    This service requires admin privileges as it allows execution of arbitrary
    Python code. All execution is isolated within Ray tasks, but should only
    be used by trusted administrators in controlled environments.

    Example:
        ```python
        # Initialize the code executer
        executer = CodeExecutor(debug=True)
        await executer.initialize(admin_users=["admin@example.com"])

        # Execute Python code
        result = await executer.execute_python_code(
            code=\"\"\"
            def process_data(data):
                return {"processed": len(data), "sum": sum(data)}
            \"\"\",
            function_name="process_data",
            args=[[1, 2, 3, 4, 5]],
            remote_options={"num_cpus": 2}
        )
        ```

    Attributes:
        logger: Configured logger instance for execution tracking
        notify_function: Optional callback for Ray cluster change notifications
        admin_users: List of authorized admin user identifiers
    """

    def __init__(
        self,
        ray_cluster: RayCluster,
        # Logger
        log_file: Optional[Union[str, Path]] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the CodeExecutor with optional notification and logging configuration.

        Args:
            notify_function: Optional async callback function to notify when tasks start.
                           Should be a callable with no parameters that returns None.
            log_file: Optional path to log file for execution tracking. If None,
                     logs will be written to stdout/stderr.
            debug: Enable debug-level logging for detailed execution information.
                  When False, only INFO level and above will be logged.

        Note:
            The admin_users must be set via initialize() before executing any code.
        """
        # Set up logging
        self.logger = create_logger(
            name="CodeExecutor",
            level=logging.DEBUG if debug else logging.INFO,
            log_file=log_file,
        )
        self.ray_cluster = ray_cluster
        self.admin_users = None

    def _load_func_from_source(self, code: str, function_name: str) -> Callable:
        """
        Load a function from a string of Python code.

        Executes the provided Python source code in an isolated namespace and
        extracts the specified function for execution. The code is dedented
        automatically to handle indented code blocks.

        Args:
            code: The Python source code as a string containing function definitions.
                 The code will be dedented automatically using textwrap.dedent.
            function_name: The name of the function to extract from the executed code.
                          Must match exactly with a function defined in the code.

        Returns:
            The extracted function object if found and callable, None otherwise.

        Raises:
            SyntaxError: If the provided code contains syntax errors.
            NameError: If the code references undefined variables during execution.
            Exception: If any other error occurs during code execution.

        Example:
            ```python
            code = '''
            def hello(name):
                return f"Hello, {name}!"
            '''
            func = executer._load_func_from_source(code, "hello")
            result = func("World")  # Returns "Hello, World!"
            ```
        """
        exec_namespace = {}
        exec(textwrap.dedent(code), exec_namespace)
        user_func = exec_namespace.get(function_name)
        return user_func

    async def initialize(self, admin_users: List[str]) -> None:
        """
        Initialize the CodeExecutor with admin user permissions.

        Sets up the list of authorized admin users who are permitted to execute
        arbitrary Python code. This method must be called before any code execution
        attempts, as the execute_python_code method will raise PermissionError
        if admin_users is not configured.

        Args:
            admin_users: List of admin user identifiers (typically email addresses
                        or user IDs) who are authorized to execute Python code.
                        These identifiers will be checked against the context.user.id
                        in code execution requests.

        Note:
            This is a security-critical method. Only trusted administrators should
            be included in the admin_users list as they will have the ability to
            execute arbitrary Python code in the Ray cluster.
        """
        self.admin_users = admin_users

    @schema_method(arbitrary_types_allowed=True)
    async def execute_python_code(
        self,
        code: Optional[str] = None,
        function_name: Optional[str] = "analyze",
        func_bytes: Optional[bytes] = None,
        mode: Literal["source", "pickle"] = "source",
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        remote_options: Optional[Dict[str, Any]] = None,
        write_stdout: Optional[Callable[[str], None]] = None,
        write_stderr: Optional[Callable[[str], None]] = None,
        timeout: int = 180,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code in a distributed Ray task with comprehensive resource management.

        Provides secure execution of user-provided Python code in isolated Ray tasks with
        configurable resource allocation, output streaming, and error handling. Supports
        both source code execution and pre-serialized function execution for maximum
        flexibility in distributed computing scenarios.

        The execution process:
        1. Validates admin permissions for code execution
        2. Deserializes function (from source or pickle) with error handling
        3. Creates isolated Ray task with specified resource requirements
        4. Executes function with stdout/stderr capture and timeout protection
        5. Returns results with comprehensive output and error information
        6. Streams output to client if callback functions provided

        Security Features:
        - Admin-only access control with permission validation
        - Isolated execution environment in Ray tasks
        - Resource limits through Ray remote options
        - Timeout protection to prevent infinite execution
        - Comprehensive error handling and logging

        Args:
            code: Optional Python source code string containing the function to execute.
                 Must define a function with the specified function_name when mode='source'.
                 Either code or func_bytes must be provided.
            function_name: Name of the function to execute. Defaults to 'main'.
                          When mode='source', this function must be defined in the code.
                          When mode='pickle', this parameter is used for logging only.
            func_bytes: Optional pre-serialized function bytes using cloudpickle.
                       Required when mode='pickle' for optimized execution.
                       Either code or func_bytes must be provided.
            mode: Execution mode determining input format:
                 - 'source': Execute function from source code string (default)
                 - 'pickle': Execute pre-serialized function from func_bytes
            args: Optional positional arguments to pass to the target function.
                 Defaults to empty list if not provided.
            kwargs: Optional keyword arguments to pass to the target function.
                   Defaults to empty dict if not provided.
            remote_options: Optional Ray remote decorator options for resource allocation:
                          - num_cpus: Number of CPU cores to allocate (default: 1)
                          - num_gpus: Number of GPU devices to allocate (default: 0)
                          - memory: Memory allocation in bytes (default: no limit)
                          - runtime_env: Python environment configuration (default: no additional packages)
            write_stdout: Optional callback function for streaming stdout output.
                         Called with each line of stdout as it's generated.
                         Must accept a single string parameter.
            write_stderr: Optional callback function for streaming stderr output.
                         Called with each line of stderr as it's generated.
                         Must accept a single string parameter.
            timeout: Maximum execution time in seconds for the function.
                    Defaults to 180 seconds (3 minutes). If the function execution
                    exceeds this time, a TimeoutError is raised.
            context: Optional request context containing user information for permission checking.
                    Must contain context['user']['id'] for admin validation.

        Returns:
            Dict containing comprehensive execution results:
                - result: Function return value (if execution successful)
                - error: Error message string (if execution failed)
                - traceback: Full Python traceback (if execution failed)
                - stdout: Captured stdout output as string
                - stderr: Captured stderr output as string
                - available_functions: List of functions found in source (if function not found)

        Raises:
            PermissionError: If user is not authorized to execute Python code.
                           Raised by check_permissions if context.user.id not in admin_users.
            ValueError: If both code and func_bytes are None, if invalid mode specified,
                       or if deserialized object is not callable.
            RuntimeError: If Ray cluster is not initialized or task execution fails.
            TimeoutError: If function execution exceeds 600 second timeout.
            asyncio.CancelledError: If the execution is cancelled externally.
            Exception: If function deserialization, code parsing, or execution
                      encounters any other critical errors.

        Example:
            ```python
            # Execute source code with resource allocation
            result = await worker.execute_python_code(
                code=\"\"\"
                def analyze(data):
                    return {"mean": sum(data) / len(data)}
                \"\"\",
                args=[[1, 2, 3, 4, 5]],
                remote_options={"num_cpus": 2, "memory": 1000000000}
            )
            print(result["result"])  # {"mean": 3.0}
            ```

        Note:
            This method requires admin privileges as it allows arbitrary code execution
            in the distributed Ray environment. The execution is isolated but should
            only be used by trusted administrators.
        """
        check_permissions(
            context=context,
            authorized_users=self.admin_users,
            resource_name=f"execute Python code in a Ray task",
        )
        user_id = context["user"]["id"]

        # Validate required parameters based on mode
        if mode == "pickle":
            if func_bytes is None:
                raise ValueError("func_bytes is required when mode='pickle'")
        elif mode == "source":
            if code is None:
                raise ValueError("code is required when mode='source'")
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'source' or 'pickle'")

        try:
            # Deserialize function before Ray execution
            if mode == "pickle":
                self.logger.debug(
                    f"User '{user_id}' is loading Python function '{function_name}' from pickle."
                )
                user_func = await asyncio.to_thread(cloudpickle.loads, func_bytes)
            else:
                self.logger.debug(
                    f"User '{user_id}' is loading Python function '{function_name}' from source."
                )
                user_func = await asyncio.to_thread(
                    self._load_func_from_source, code, function_name
                )

            if not callable(user_func):
                raise ValueError(
                    f"Object '{function_name}' is not callable: {user_func}"
                )

        except Exception as e:
            self.logger.error(f"Failed to load function: {e}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Configure Ray task with resource options (validates provided remote_options)
        remote_options = remote_options or {}
        try:
            configured_ray_task = ray_task.options(**remote_options)
        except ValueError as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Prepare arguments for Ray task
        args = args or []
        kwargs = kwargs or {}

        # Execute the function in a Ray task
        self.logger.info(
            f"User '{user_id}' is executing Python function '{function_name}' "
            f"in Ray task (remote_options={json.dumps(remote_options)})"
        )
        obj_ref = configured_ray_task.remote(user_func, args, kwargs)

        # If in SLURM mode, signal a resource request
        if self.ray_cluster.mode == "slurm":
            self.logger.info(
                "Notifying SLURM workers' autoscaling system of a change in cluster state."
            )
            await self.ray_cluster.monitor_cluster()

        try:
            result = await asyncio.wait_for(obj_ref, timeout=timeout)
        except asyncio.TimeoutError:
            return {
                "error": f"Function execution timed out after {timeout} seconds.",
                "traceback": "TimeoutError: Function execution exceeded maximum allowed time.",
            }
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Stream output to client
        if write_stdout and result.get("stdout"):
            for line in result["stdout"].splitlines():
                await write_stdout(line)
        if write_stderr and result.get("stderr"):
            for line in result["stderr"].splitlines():
                await write_stderr(line)

        return result


if __name__ == "__main__":

    class MockRayCluster:
        mode = "mock"

    executer = CodeExecutor(ray_cluster=MockRayCluster())

    # Load and execute a simple function from code
    simple_code = """
    def analyze():
        import time

        result = {
            "message": "Hello from BioEngine Worker!",
            "timestamp": time.time(),
            "python_version": "3.x"
        }

        return result
    """
    user_func = executer._load_func_from_source(simple_code, "analyze")
    assert callable(user_func)

    # Execute the function
    result = user_func()
    print(result)

    # Load and execute a parameterized function from code
    parameterized_code = """
    def analyze(name: str, multiplier: int):
    
        return {
            "greeting": f"Hello {name}!",
            "result": 42 * multiplier,
            "processed": True
        }
    """
    kwargs = {"name": "Alice", "multiplier": 3}

    user_func = executer._load_func_from_source(parameterized_code, "analyze")
    assert callable(user_func)

    # Execute the function with parameters
    result = user_func(**kwargs)
    print(result)
