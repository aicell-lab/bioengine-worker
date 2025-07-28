"""
End-to-end tests for BioEngine Worker CodeExecutor component.

This module tests the Python code execution functionality through the Hypha service API,
including execution of various types of simple test functions with different parameters,
return types, and computational patterns.
"""

import math

import cloudpickle
import pytest


@pytest.mark.asyncio
async def test_execute_python_code_simple_operations(bioengine_worker_service):
    """
    Test executing simple Python operations without external imports.

    This test validates:
    1. Basic Python code execution through the worker service
    2. Various data types and operations (math, strings, lists, dicts)
    3. Control flow and conditional logic
    4. Parameter passing with args only, kwargs only, and args+kwargs

    Steps:
    - Execute Python code with basic operations
    - Test different parameter passing methods
    - Verify correct results without external dependencies
    """
    # Define comprehensive function that tests multiple basic operations
    code = """
def process_data(numbers=None, texts=None, threshold=10, multiplier=1):
    # Handle defaults
    if numbers is None:
        numbers = []
    if texts is None:
        texts = []
    
    # Math operations
    math_result = {
        "sum": sum(numbers) * multiplier,
        "average": (sum(numbers) / len(numbers)) * multiplier if numbers else 0,
        "max": max(numbers) * multiplier if numbers else 0,
        "filtered": [x * multiplier for x in numbers if x > threshold]
    }
    
    # String operations
    string_result = {
        "combined": " ".join(texts),
        "lengths": [len(text) for text in texts],
        "uppercased": [text.upper() for text in texts],
        "total_chars": sum(len(text) for text in texts)
    }
    
    # List operations
    list_result = {
        "sorted_numbers": sorted(numbers),
        "even_numbers": [x for x in numbers if x % 2 == 0],
        "text_word_count": len(" ".join(texts).split()) if texts else 0
    }
    
    # Dictionary operations
    word_count = {}
    for text in texts:
        for word in text.lower().split():
            word_count[word] = word_count.get(word, 0) + 1
    
    # Conditional logic
    category = "small" if len(numbers) < 5 else "medium" if len(numbers) < 10 else "large"
    
    return {
        "math": math_result,
        "strings": string_result,
        "lists": list_result,
        "word_count": word_count,
        "category": category,
        "input_summary": {
            "number_count": len(numbers),
            "text_count": len(texts),
            "threshold": threshold,
            "multiplier": multiplier
        }
    }
"""

    # Test 1: Args only - passing positional arguments
    result = await bioengine_worker_service.execute_python_code(
        code=code,
        function_name="process_data",
        args=[[1, 5, 10, 15, 20], ["hello world", "python test"], 8, 2],
    )

    # Verify the results with args only
    assert result["result"]["math"]["sum"] == 102  # (1+5+10+15+20) * 2
    assert result["result"]["math"]["average"] == 20.4  # 10.2 * 2
    assert result["result"]["math"]["max"] == 40  # 20 * 2
    assert result["result"]["math"]["filtered"] == [20, 30, 40]  # [10, 15, 20] * 2

    assert result["result"]["strings"]["combined"] == "hello world python test"
    assert result["result"]["strings"]["lengths"] == [11, 11]
    assert result["result"]["strings"]["total_chars"] == 22

    assert result["result"]["lists"]["sorted_numbers"] == [1, 5, 10, 15, 20]
    assert result["result"]["lists"]["even_numbers"] == [10, 20]
    assert result["result"]["lists"]["text_word_count"] == 4

    assert result["result"]["word_count"] == {
        "hello": 1,
        "world": 1,
        "python": 1,
        "test": 1,
    }
    assert result["result"]["category"] == "medium"
    assert result["result"]["input_summary"]["threshold"] == 8
    assert result["result"]["input_summary"]["multiplier"] == 2
    assert "error" not in result or result["error"] is None

    # Test 2: Kwargs only - passing keyword arguments
    result = await bioengine_worker_service.execute_python_code(
        code=code,
        function_name="process_data",
        kwargs={
            "numbers": [2, 4, 6, 8],
            "texts": ["test data", "kwargs only"],
            "threshold": 5,
            "multiplier": 3,
        },
    )

    # Verify the results with kwargs only
    assert result["result"]["math"]["sum"] == 60  # (2+4+6+8) * 3
    assert result["result"]["math"]["average"] == 15.0  # 5.0 * 3
    assert result["result"]["math"]["max"] == 24  # 8 * 3
    assert result["result"]["math"]["filtered"] == [18, 24]  # [6, 8] * 3

    assert result["result"]["strings"]["combined"] == "test data kwargs only"
    assert result["result"]["strings"]["lengths"] == [9, 11]
    assert result["result"]["strings"]["total_chars"] == 20

    assert result["result"]["lists"]["sorted_numbers"] == [2, 4, 6, 8]
    assert result["result"]["lists"]["even_numbers"] == [2, 4, 6, 8]
    assert result["result"]["lists"]["text_word_count"] == 4

    assert result["result"]["word_count"] == {
        "test": 1,
        "data": 1,
        "kwargs": 1,
        "only": 1,
    }
    assert result["result"]["category"] == "small"
    assert result["result"]["input_summary"]["threshold"] == 5
    assert result["result"]["input_summary"]["multiplier"] == 3
    assert "error" not in result or result["error"] is None

    # Test 3: Args and kwargs combined
    result = await bioengine_worker_service.execute_python_code(
        code=code,
        function_name="process_data",
        args=[[3, 6, 9, 12, 15, 18], ["combined test"]],
        kwargs={"threshold": 10, "multiplier": 2},
    )

    # Verify the results with args and kwargs combined
    assert result["result"]["math"]["sum"] == 126  # (3+6+9+12+15+18) * 2
    assert result["result"]["math"]["average"] == 21.0  # 10.5 * 2
    assert result["result"]["math"]["max"] == 36  # 18 * 2
    assert result["result"]["math"]["filtered"] == [24, 30, 36]  # [12, 15, 18] * 2

    assert result["result"]["strings"]["combined"] == "combined test"
    assert result["result"]["strings"]["lengths"] == [13]
    assert result["result"]["strings"]["total_chars"] == 13

    assert result["result"]["lists"]["sorted_numbers"] == [3, 6, 9, 12, 15, 18]
    assert result["result"]["lists"]["even_numbers"] == [6, 12, 18]
    assert result["result"]["lists"]["text_word_count"] == 2

    assert result["result"]["word_count"] == {"combined": 1, "test": 1}
    assert result["result"]["category"] == "medium"
    assert result["result"]["input_summary"]["threshold"] == 10
    assert result["result"]["input_summary"]["multiplier"] == 2
    assert "error" not in result or result["error"] is None

    # Test 4: No args or kwargs (using defaults)
    result = await bioengine_worker_service.execute_python_code(
        code=code, function_name="process_data"
    )

    # Verify the results with defaults
    assert result["result"]["math"]["sum"] == 0
    assert result["result"]["math"]["average"] == 0
    assert result["result"]["math"]["max"] == 0
    assert result["result"]["math"]["filtered"] == []

    assert result["result"]["strings"]["combined"] == ""
    assert result["result"]["strings"]["lengths"] == []
    assert result["result"]["strings"]["total_chars"] == 0

    assert result["result"]["lists"]["sorted_numbers"] == []
    assert result["result"]["lists"]["even_numbers"] == []
    assert result["result"]["lists"]["text_word_count"] == 0

    assert result["result"]["word_count"] == {}
    assert result["result"]["category"] == "small"
    assert result["result"]["input_summary"]["threshold"] == 10
    assert result["result"]["input_summary"]["multiplier"] == 1
    assert "error" not in result or result["error"] is None


@pytest.mark.asyncio
async def test_execute_python_code_with_standard_libraries(
    bioengine_worker_service
):
    """
    Test executing functions that use standard Python library imports.

    This test validates:
    1. Import statements in remote execution environment
    2. Standard library module access (math, datetime, json, collections)
    3. Function execution with external dependencies
    4. Module availability in Ray workers

    Steps:
    - Execute code that imports and uses standard library modules
    - Test multiple library functions in one execution
    - Verify proper module initialization and functionality
    """
    # Define function that uses multiple standard library imports
    code = """
import math
import datetime
import json
import random
from collections import Counter

def analyze_with_libraries(numbers, texts, seed=42):
    # Set seed for reproducible results
    random.seed(seed)
    
    # Math operations
    math_results = {
        "sqrt_sum": math.sqrt(sum(numbers)) if numbers else 0,
        "log_sum": math.log10(sum(numbers)) if sum(numbers) > 0 else None,
        "sin_value": math.sin(math.pi / 4),  # Should be ~0.707
        "factorial_5": math.factorial(5),
        "gcd_example": math.gcd(48, 18)
    }
    
    # Datetime operations
    now = datetime.datetime.now()
    datetime_results = {
        "current_year": now.year,
        "formatted_time": now.strftime("%Y-%m-%d %H:%M"),
        "timestamp": now.timestamp(),
        "days_since_epoch": (now - datetime.datetime(1970, 1, 1)).days
    }
    
    # Collections operations
    word_list = []
    for text in texts:
        word_list.extend(text.lower().split())
    
    counter = Counter(word_list)
    collections_results = {
        "word_counts": dict(counter),
        "most_common": counter.most_common(2),
        "total_words": sum(counter.values()),
        "unique_words": len(counter)
    }
    
    # JSON operations
    data_to_serialize = {
        "numbers": numbers,
        "texts": texts,
        "summary": {"count": len(numbers), "text_count": len(texts)}
    }
    json_string = json.dumps(data_to_serialize)
    parsed_back = json.loads(json_string)
    
    # Random operations with fixed seed
    random_results = {
        "random_int": random.randint(1, 100),
        "random_choice": random.choice(["A", "B", "C"]),
        "shuffled_numbers": random.sample(numbers, min(3, len(numbers))) if numbers else []
    }
    
    return {
        "math": math_results,
        "datetime": datetime_results,
        "collections": collections_results,
        "json_length": len(json_string),
        "json_valid": parsed_back == data_to_serialize,
        "random": random_results,
        "execution_summary": {
            "libraries_used": ["math", "datetime", "json", "random", "collections"],
            "input_numbers": len(numbers),
            "input_texts": len(texts)
        }
    }
"""

    # Test the comprehensive function with standard libraries
    result = await bioengine_worker_service.execute_python_code(
        code=code,
        function_name="analyze_with_libraries",
        args=[[4, 9, 16, 25], ["hello world", "python programming", "test data"], 42],
    )

    # Verify math operations
    assert result["result"]["math"]["sqrt_sum"] == math.sqrt(54)  # sqrt(4+9+16+25)
    assert abs(result["result"]["math"]["sin_value"] - math.sin(math.pi / 4)) < 0.0001
    assert result["result"]["math"]["factorial_5"] == 120
    assert result["result"]["math"]["gcd_example"] == 6

    # Verify datetime operations
    assert result["result"]["datetime"]["current_year"] >= 2020
    assert result["result"]["datetime"]["days_since_epoch"] > 0
    assert isinstance(result["result"]["datetime"]["formatted_time"], str)

    # Verify collections operations
    assert result["result"]["collections"]["word_counts"]["hello"] == 1
    assert result["result"]["collections"]["word_counts"]["python"] == 1
    assert result["result"]["collections"]["total_words"] == 6
    assert result["result"]["collections"]["unique_words"] == 6

    # Verify JSON operations
    assert result["result"]["json_valid"] == True
    assert result["result"]["json_length"] > 0

    # Verify random operations (should be reproducible with seed=42)
    assert 1 <= result["result"]["random"]["random_int"] <= 100
    assert result["result"]["random"]["random_choice"] in ["A", "B", "C"]
    assert len(result["result"]["random"]["shuffled_numbers"]) == 3

    # Verify execution summary
    assert result["result"]["execution_summary"]["input_numbers"] == 4
    assert result["result"]["execution_summary"]["input_texts"] == 3
    assert "error" not in result or result["error"] is None


@pytest.mark.asyncio
async def test_execute_python_code_with_runtime_env_packages(
    bioengine_worker_service
):
    """
    Test executing functions that use runtime environment packages.

    This test validates:
    1. Runtime environment creation with specific pip packages
    2. Required packages (numpy, pandas, scipy, requests, Pillow) availability
    3. Package functionality in remote workers with runtime environment
    4. All specified packages must be available and functional

    Steps:
    - Create runtime environment with required packages
    - Test functionality of all required packages
    - Verify all packages are available (no ImportErrors allowed)
    - Validate package versions and basic operations
    """
    # Define function that uses all required runtime environment packages
    code = """
def test_runtime_packages():
    results = {
        "package_tests": {},
        "package_versions": {}
    }
    
    # Test numpy - REQUIRED
    import numpy as np
    results["package_versions"]["numpy"] = np.__version__
    
    # Test basic numpy functionality
    arr = np.array([1, 2, 3, 4, 5])
    results["package_tests"]["numpy"] = {
        "array_creation": arr.tolist(),
        "array_sum": np.sum(arr).item(),
        "array_mean": np.mean(arr).item(),
        "array_std": np.std(arr).item(),
        "array_shape": arr.shape,
        "linspace": np.linspace(0, 10, 5).tolist(),
        "zeros": np.zeros(3).tolist(),
        "ones": np.ones(3).tolist(),
        "matrix_multiply": np.dot(arr, arr).item()
    }
    
    # Test pandas - REQUIRED
    import pandas as pd
    results["package_versions"]["pandas"] = pd.__version__
    
    # Test basic pandas functionality
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": ["a", "b", "c", "d"],
        "C": [1.1, 2.2, 3.3, 4.4]
    })
    results["package_tests"]["pandas"] = {
        "dataframe_shape": df.shape,
        "column_names": df.columns.tolist(),
        "sum_A": df["A"].sum(),
        "mean_C": df["C"].mean(),
        "to_dict": df.to_dict("records"),
        "groupby_test": df.groupby("B")["A"].sum().to_dict(),
        "describe": df["C"].describe().to_dict()
    }
    
    # Test scipy - REQUIRED
    import scipy
    from scipy import stats
    results["package_versions"]["scipy"] = scipy.__version__
    
    # Test basic scipy functionality
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results["package_tests"]["scipy"] = {
        "mean": stats.describe(data).mean,
        "variance": stats.describe(data).variance,
        "norm_pdf": stats.norm.pdf(0, 0, 1),  # Standard normal at x=0
        "binom_pmf": stats.binom.pmf(3, 10, 0.3),  # Binomial probability
        "ttest": stats.ttest_1samp(data, 5.0).pvalue,
        "pearson_corr": stats.pearsonr([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]).correlation
    }
    
    # Test requests - REQUIRED
    import requests
    results["package_versions"]["requests"] = requests.__version__
    
    # Test requests functionality (without making actual network calls)
    results["package_tests"]["requests"] = {
        "import_successful": True,
        "has_get_method": hasattr(requests, "get"),
        "has_post_method": hasattr(requests, "post"),
        "has_session_class": hasattr(requests, "Session"),
        "available_methods": [m for m in dir(requests) if not m.startswith("_")][:10]
    }
    
    # Test PIL/Pillow - REQUIRED
    from PIL import Image
    import PIL
    results["package_versions"]["PIL"] = PIL.__version__
    
    # Test basic PIL functionality
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='red')
    results["package_tests"]["PIL"] = {
        "import_successful": True,
        "image_size": test_image.size,
        "image_mode": test_image.mode,
        "has_image_class": hasattr(Image, "new"),
        "available_formats": list(Image.registered_extensions().keys())[:10],
        "can_create_image": True,
        "supported_modes": ["RGB", "RGBA", "L"] # Common modes
    }
    
    # Test all packages are working together
    combined_test = {
        "numpy_pandas_integration": len(pd.DataFrame(np.random.random((5, 3)))),
        "scipy_numpy_integration": float(stats.norm.pdf(np.array([0, 1, 2]), 0, 1)[0]),
        "all_packages_imported": True
    }
    results["package_tests"]["integration"] = combined_test
    
    # Summary - all packages must be available
    results["summary"] = {
        "required_packages": ["numpy", "pandas", "scipy", "requests", "PIL"],
        "all_available": True,
        "package_count": len(results["package_versions"]),
        "test_count": len(results["package_tests"])
    }
    
    return results
"""

    # Create runtime environment with required packages
    remote_options = {
        "runtime_env": {
            "pip": [
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "scipy>=1.7.0",
                "requests>=2.25.0",
                "Pillow>=8.0.0",
            ]
        }
    }

    # Execute with runtime environment - all packages must be available
    result = await bioengine_worker_service.execute_python_code(
        code=code,
        function_name="test_runtime_packages",
        args=[],
        remote_options=remote_options,
    )

    # Verify no execution errors
    assert (
        "error" not in result or result["error"] is None
    ), f"Execution failed: {result.get('error')}"

    # All required packages must be available
    required_packages = ["numpy", "pandas", "scipy", "requests", "PIL"]

    for package in required_packages:
        assert (
            package in result["result"]["package_versions"]
        ), f"Package {package} not available"
        assert (
            package in result["result"]["package_tests"]
        ), f"Package {package} tests not executed"
        print(
            f"✓ {package} {result['result']['package_versions'][package]} is available and working"
        )

    # Verify numpy functionality
    numpy_tests = result["result"]["package_tests"]["numpy"]
    assert numpy_tests["array_creation"] == [1, 2, 3, 4, 5]
    assert numpy_tests["array_sum"] == 15
    assert numpy_tests["array_mean"] == 3.0
    assert len(numpy_tests["linspace"]) == 5
    assert numpy_tests["zeros"] == [0.0, 0.0, 0.0]
    assert numpy_tests["ones"] == [1.0, 1.0, 1.0]
    assert (
        numpy_tests["matrix_multiply"] == 55
    )  # dot product of [1,2,3,4,5] with itself

    # Verify pandas functionality
    pandas_tests = result["result"]["package_tests"]["pandas"]
    assert pandas_tests["dataframe_shape"] == [4, 3]
    assert pandas_tests["column_names"] == ["A", "B", "C"]
    assert pandas_tests["sum_A"] == 10
    assert pandas_tests["mean_C"] == 2.75
    assert len(pandas_tests["to_dict"]) == 4
    assert "mean" in pandas_tests["describe"]

    # Verify scipy functionality
    scipy_tests = result["result"]["package_tests"]["scipy"]
    assert scipy_tests["mean"] == 5.5
    assert (
        abs(scipy_tests["norm_pdf"] - 0.3989422804014327) < 0.0001
    )  # Standard normal PDF at 0
    assert 0 <= scipy_tests["binom_pmf"] <= 1  # Valid probability
    assert abs(scipy_tests["pearson_corr"] - 1.0) < 0.0001  # Perfect correlation

    # Verify requests functionality
    requests_tests = result["result"]["package_tests"]["requests"]
    assert requests_tests["import_successful"] == True
    assert requests_tests["has_get_method"] == True
    assert requests_tests["has_post_method"] == True
    assert requests_tests["has_session_class"] == True
    assert len(requests_tests["available_methods"]) >= 5

    # Verify PIL functionality
    pil_tests = result["result"]["package_tests"]["PIL"]
    assert pil_tests["import_successful"] == True
    assert pil_tests["image_size"] == [100, 100]
    assert pil_tests["image_mode"] == "RGB"
    assert pil_tests["can_create_image"] == True
    assert len(pil_tests["available_formats"]) >= 5

    # Verify integration tests
    integration_tests = result["result"]["package_tests"]["integration"]
    assert integration_tests["numpy_pandas_integration"] == 5  # 5 rows in dataframe
    assert integration_tests["all_packages_imported"] == True
    assert 0 <= integration_tests["scipy_numpy_integration"] <= 1  # Valid probability

    # Verify summary
    summary = result["result"]["summary"]
    assert summary["all_available"] == True
    assert summary["package_count"] == 5
    assert summary["test_count"] == 6  # 5 packages + integration tests
    assert set(summary["required_packages"]) == set(required_packages)

    print("✓ All required packages are available and functional in runtime environment")


@pytest.mark.asyncio
async def test_execute_python_code_error_handling(bioengine_worker_service):
    """
    Test error handling in Python code execution.

    This test validates:
    1. Proper error reporting and worker stability
    2. Safe error handling within functions
    3. Exception propagation from remote execution
    4. Recovery from execution failures

    Steps:
    - Execute functions with built-in error handling
    - Test deliberate error conditions
    - Verify worker remains stable after errors
    - Check proper error message formatting
    """
    # Define functions that handle errors gracefully
    code = """
def safe_division(a, b):
    try:
        result = a / b
        return {"success": True, "result": result, "error": None}
    except ZeroDivisionError:
        return {"success": False, "result": None, "error": "Cannot divide by zero"}
    except TypeError as e:
        return {"success": False, "result": None, "error": f"Type error: {str(e)}"}

def safe_list_access(items, index):
    try:
        return {"success": True, "value": items[index], "error": None}
    except IndexError:
        return {"success": False, "value": None, "error": f"Index {index} out of range"}
    except TypeError as e:
        return {"success": False, "value": None, "error": f"Invalid type: {str(e)}"}

def error_recovery_test():
    results = []
    test_cases = [
        lambda: 10 / 2,           # Success
        lambda: 10 / 0,           # Division by zero
        lambda: "hello"[100],     # Index error
        lambda: int("invalid"),   # Value error
        lambda: 5 + 3            # Success
    ]
    
    for i, test_case in enumerate(test_cases):
        try:
            result = test_case()
            results.append({"test": i, "success": True, "result": result, "error": None})
        except Exception as e:
            results.append({"test": i, "success": False, "result": None, "error": str(e)})
    
    return results

def cause_specific_error(error_type):
    # This function will actually cause errors for testing error propagation
    if error_type == "zero_division":
        return 1 / 0
    elif error_type == "type_error":
        return "string" + 5
    elif error_type == "index_error":
        return [1, 2, 3][10]
    elif error_type == "key_error":
        return {"a": 1}["b"]
    else:
        return "no_error"
"""

    # Test safe division - success case
    result = await bioengine_worker_service.execute_python_code(
        code=code, function_name="safe_division", args=[10, 2]
    )
    assert result["result"]["success"] == True
    assert result["result"]["result"] == 5.0

    # Test safe division - zero division (handled gracefully)
    result = await bioengine_worker_service.execute_python_code(
        code=code, function_name="safe_division", args=[10, 0]
    )
    assert result["result"]["success"] == False
    assert "divide by zero" in result["result"]["error"].lower()

    # Test safe list access - success case
    result = await bioengine_worker_service.execute_python_code(
        code=code, function_name="safe_list_access", args=[[1, 2, 3, 4, 5], 2]
    )
    assert result["result"]["success"] == True
    assert result["result"]["value"] == 3

    # Test safe list access - index error (handled gracefully)
    result = await bioengine_worker_service.execute_python_code(
        code=code, function_name="safe_list_access", args=[[1, 2, 3], 10]
    )
    assert result["result"]["success"] == False
    assert "out of range" in result["result"]["error"]

    # Test error recovery
    result = await bioengine_worker_service.execute_python_code(
        code=code, function_name="error_recovery_test", args=[]
    )

    recovery_results = result["result"]
    assert len(recovery_results) == 5

    # Check that some succeeded and some failed
    successes = [r for r in recovery_results if r["success"]]
    failures = [r for r in recovery_results if not r["success"]]

    assert len(successes) >= 2  # At least the first and last should succeed
    assert len(failures) >= 2  # At least the error cases should fail

    # Test that worker is still functional after mixed success/failure
    result = await bioengine_worker_service.execute_python_code(
        code=code, function_name="safe_division", args=[20, 4]
    )
    assert result["result"]["success"] == True
    assert result["result"]["result"] == 5.0

    # Test actual error propagation (these should return errors in the response)
    error_cases = ["zero_division", "type_error", "index_error", "key_error"]

    for error_type in error_cases:
        result = await bioengine_worker_service.execute_python_code(
            code=code, function_name="cause_specific_error", args=[error_type]
        )
        # These should fail and return error information
        assert "error" in result and result["error"] is not None

    # Test successful case to ensure worker is still working
    result = await bioengine_worker_service.execute_python_code(
        code=code, function_name="cause_specific_error", args=["success"]
    )
    assert result["result"] == "no_error"
    assert "error" not in result or result["error"] is None


@pytest.mark.asyncio
async def test_execute_python_code_with_pickle_mode(bioengine_worker_service):
    """
    Test executing pre-serialized Python functions using pickle mode.

    This test validates:
    1. Function serialization with cloudpickle
    2. Remote execution of pickled functions
    3. Parameter passing to pickled functions
    4. Complex data structures and closures in pickled functions

    Steps:
    - Create and serialize functions with cloudpickle
    - Execute pickled functions remotely with various parameters
    - Test functions with closures and complex return types
    - Verify results match expected behavior
    """
    # Define a simple function to pickle
    def simple_calculation(x, y, operation="add"):
        operations = {
            "add": lambda a, b: a + b,
            "multiply": lambda a, b: a * b,
            "power": lambda a, b: a**b,
            "divide": lambda a, b: a / b if b != 0 else None,
        }

        if operation not in operations:
            return {"error": f"Unknown operation: {operation}"}

        result = operations[operation](x, y)
        return {
            "operation": operation,
            "inputs": {"x": x, "y": y},
            "result": result,
            "type": type(result).__name__,
        }

    # Serialize the function with cloudpickle
    func_bytes = cloudpickle.dumps(simple_calculation)

    # Test with addition
    result = await bioengine_worker_service.execute_python_code(
        mode="pickle",
        func_bytes=func_bytes,
        function_name="simple_calculation",
        args=[10, 5],
        kwargs={"operation": "add"},
    )

    assert "error" not in result or result["error"] is None
    assert result["result"]["operation"] == "add"
    assert result["result"]["inputs"]["x"] == 10
    assert result["result"]["inputs"]["y"] == 5
    assert result["result"]["result"] == 15
    assert result["result"]["type"] == "int"

    # Test with multiplication
    result = await bioengine_worker_service.execute_python_code(
        mode="pickle",
        func_bytes=func_bytes,
        function_name="simple_calculation",
        args=[7, 3],
        kwargs={"operation": "multiply"},
    )

    assert result["result"]["operation"] == "multiply"
    assert result["result"]["result"] == 21

    # Test with power operation
    result = await bioengine_worker_service.execute_python_code(
        mode="pickle",
        func_bytes=func_bytes,
        function_name="simple_calculation",
        args=[2, 8],
        kwargs={"operation": "power"},
    )

    assert result["result"]["operation"] == "power"
    assert result["result"]["result"] == 256

    # Define a more complex function with closure
    def create_data_processor(multiplier):
        def process_data(data_list):
            import statistics

            # Apply multiplier to all values
            transformed = [x * multiplier for x in data_list]

            # Calculate statistics
            stats = {
                "original": {
                    "mean": statistics.mean(data_list),
                    "median": statistics.median(data_list),
                    "stdev": statistics.stdev(data_list) if len(data_list) > 1 else 0,
                },
                "transformed": {
                    "mean": statistics.mean(transformed),
                    "median": statistics.median(transformed),
                    "stdev": (
                        statistics.stdev(transformed) if len(transformed) > 1 else 0
                    ),
                },
                "multiplier": multiplier,
                "count": len(data_list),
            }

            return stats

        return process_data

    # Create a processor with multiplier=3
    processor = create_data_processor(3)
    processor_bytes = cloudpickle.dumps(processor)

    # Test the complex function with closure
    test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = await bioengine_worker_service.execute_python_code(
        mode="pickle",
        func_bytes=processor_bytes,
        function_name="data_processor",
        args=[test_data],
    )

    assert "error" not in result or result["error"] is None
    assert result["result"]["multiplier"] == 3
    assert result["result"]["count"] == 10
    assert result["result"]["original"]["mean"] == 5.5  # Mean of 1-10
    assert result["result"]["transformed"]["mean"] == 16.5  # Mean of 3-30 (3*1 to 3*10)
    assert result["result"]["original"]["median"] == 5.5
    assert result["result"]["transformed"]["median"] == 16.5

    # Test function with external dependencies
    def numpy_computation():
        import numpy as np

        # Create test arrays
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([2, 4, 6, 8, 10])

        return {
            "dot_product": np.dot(a, b).item(),
            "element_wise_sum": (a + b).tolist(),
            "element_wise_product": (a * b).tolist(),
            "matrix_result": np.outer(a, b)[:2, :2].tolist(),  # 2x2 subset
            "numpy_version": np.__version__,
        }

    numpy_func_bytes = cloudpickle.dumps(numpy_computation)

    # Execute with runtime environment for numpy
    result = await bioengine_worker_service.execute_python_code(
        mode="pickle",
        func_bytes=numpy_func_bytes,
        function_name="numpy_computation",
        args=[],
        remote_options={"runtime_env": {"pip": ["numpy>=1.21.0"]}},
    )

    assert "error" not in result or result["error"] is None
    assert result["result"]["dot_product"] == 110  # 1*2 + 2*4 + 3*6 + 4*8 + 5*10
    assert result["result"]["element_wise_sum"] == [3, 6, 9, 12, 15]
    assert result["result"]["element_wise_product"] == [2, 8, 18, 32, 50]
    assert len(result["result"]["matrix_result"]) == 2
    assert len(result["result"]["matrix_result"][0]) == 2
    assert "numpy_version" in result["result"]


@pytest.mark.asyncio
async def test_execute_python_code_with_stdout_stderr_callbacks(
    bioengine_worker_service
):
    """
    Test Python code execution with stdout and stderr callback functions.

    This test validates:
    1. Proper capture and streaming of stdout output
    2. Proper capture and streaming of stderr output
    3. Callback function invocation for each output line
    4. Mixed stdout/stderr output handling

    Steps:
    - Define functions that produce stdout and stderr output
    - Collect output via callback functions
    - Verify all output lines are captured correctly
    - Test both print statements and error output
    """
    # Storage for captured output
    captured_stdout = []
    captured_stderr = []

    # Define callback functions
    async def stdout_callback(line):
        captured_stdout.append(line)

    async def stderr_callback(line):
        captured_stderr.append(line)

    # Define function that produces both stdout and stderr output
    code = """
import sys

def produce_output():
    # Print to stdout
    print("Starting computation...")
    print("Processing data: [1, 2, 3, 4, 5]")
    
    # Print to stderr
    print("Warning: This is a test warning", file=sys.stderr)
    print("Debug: Processing step 1", file=sys.stderr)
    
    # More stdout
    for i in range(1, 4):
        print(f"Step {i}: Computing result...")
    
    # More stderr
    print("Warning: Memory usage is high", file=sys.stderr)
    
    # Final stdout
    print("Computation completed successfully!")
    
    # Return result
    return {
        "status": "completed",
        "steps_processed": 3,
        "warnings_generated": 2
    }
"""

    # Clear previous captures
    captured_stdout.clear()
    captured_stderr.clear()

    # Execute with stdout/stderr callbacks
    result = await bioengine_worker_service.execute_python_code(
        code=code,
        function_name="produce_output",
        args=[],
        write_stdout=stdout_callback,
        write_stderr=stderr_callback,
    )

    # Verify execution was successful
    assert "error" not in result or result["error"] is None
    assert result["result"]["status"] == "completed"
    assert result["result"]["steps_processed"] == 3
    assert result["result"]["warnings_generated"] == 2

    # Verify stdout capture
    assert len(captured_stdout) >= 5  # At least 5 stdout lines expected

    # Check specific stdout content
    stdout_content = "\n".join(captured_stdout)
    assert "Starting computation..." in stdout_content
    assert "Processing data: [1, 2, 3, 4, 5]" in stdout_content
    assert "Step 1: Computing result..." in stdout_content
    assert "Step 2: Computing result..." in stdout_content
    assert "Step 3: Computing result..." in stdout_content
    assert "Computation completed successfully!" in stdout_content

    # Verify stderr capture
    assert len(captured_stderr) >= 3  # At least 3 stderr lines expected

    # Check specific stderr content
    stderr_content = "\n".join(captured_stderr)
    assert "Warning: This is a test warning" in stderr_content
    assert "Debug: Processing step 1" in stderr_content
    assert "Warning: Memory usage is high" in stderr_content

    # Test function with only stdout output
    simple_code = """
def simple_output():
    print("Line 1: Hello World")
    print("Line 2: Testing stdout")
    print("Line 3: Final message")
    return "done"
"""

    # Clear and test stdout-only
    captured_stdout.clear()
    captured_stderr.clear()

    result = await bioengine_worker_service.execute_python_code(
        code=simple_code,
        function_name="simple_output",
        args=[],
        write_stdout=stdout_callback,
        write_stderr=stderr_callback,
    )

    assert result["result"] == "done"
    assert len(captured_stdout) == 3
    assert captured_stdout[0] == "Line 1: Hello World"
    assert captured_stdout[1] == "Line 2: Testing stdout"
    assert captured_stdout[2] == "Line 3: Final message"
    assert len(captured_stderr) == 0  # No stderr output expected

    # Test function with error output but successful execution
    error_code = """
import sys
import warnings

def output_with_warnings():
    # Generate some warnings that go to stderr
    warnings.warn("This is a test warning", UserWarning)
    
    # Print debug info to stderr
    print("Debug: Function started", file=sys.stderr)
    print("Debug: Processing data", file=sys.stderr)
    
    # Print progress to stdout
    print("Progress: 50%")
    print("Progress: 100%")
    
    # More debug to stderr
    print("Debug: Function completed", file=sys.stderr)
    
    return {"warnings": 1, "progress_updates": 2}
"""

    # Clear and test mixed output
    captured_stdout.clear()
    captured_stderr.clear()

    result = await bioengine_worker_service.execute_python_code(
        code=error_code,
        function_name="output_with_warnings",
        args=[],
        write_stdout=stdout_callback,
        write_stderr=stderr_callback,
    )

    assert "error" not in result or result["error"] is None
    assert result["result"]["warnings"] == 1
    assert result["result"]["progress_updates"] == 2

    # Verify stdout captured progress updates
    stdout_content = "\n".join(captured_stdout)
    assert "Progress: 50%" in stdout_content
    assert "Progress: 100%" in stdout_content

    # Verify stderr captured debug and warning messages
    stderr_content = "\n".join(captured_stderr)
    assert "Debug: Function started" in stderr_content
    assert "Debug: Processing data" in stderr_content
    assert "Debug: Function completed" in stderr_content
    # Warning messages from warnings module should also appear
    assert len(captured_stderr) >= 3  # At least the debug messages

    # Test execution without callbacks (should still work)
    captured_stdout.clear()
    captured_stderr.clear()

    result = await bioengine_worker_service.execute_python_code(
        code=simple_code,
        function_name="simple_output",
        args=[],
        # No write_stdout or write_stderr callbacks
    )

    assert result["result"] == "done"
    # Output should still be in the result even without callbacks
    assert "stdout" in result
    assert "stderr" in result
    assert "Line 1: Hello World" in result["stdout"]
    assert "Line 2: Testing stdout" in result["stdout"]
    assert "Line 3: Final message" in result["stdout"]
