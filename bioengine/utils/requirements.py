import re
from pathlib import Path
from typing import List, Optional

split_re = re.compile(r"(==|>=|<=|~=|>|<)")


def get_pip_requirements(select: Optional[List[str]] = None) -> List[str]:
    requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
    requirements = requirements_path.read_text().splitlines()

    if select is None:
        # If select is None, return all requirements except those starting with "ray"
        filtered_requirements = [
            requirement
            for requirement in requirements
            if requirement and not requirement.startswith("ray")
        ]
    else:
        # Otherwise, filter based on the select list
        filtered_requirements = [
            requirement
            for requirement in requirements
            if requirement
            and not requirement.startswith("ray")
            and split_re.split(requirement, maxsplit=1)[0] in select
        ]

    return filtered_requirements


def update_requirements(
    requirements: List[str], select: Optional[List[str]] = None
) -> List[str]:
    """
    Update the provided list of pip requirements with the missing BioEngine requirements.
    If a requirement is already present, it will not be overwritten.
    If `select` is provided, only the requirements that match the names in `select` will be added.
    """
    bioengine_requirements = get_pip_requirements(select)

    for bioengine_requirement in bioengine_requirements:
        exists = False
        for requirement in requirements:
            if (
                split_re.split(bioengine_requirement, maxsplit=1)[0]
                == split_re.split(requirement, maxsplit=1)[0]
            ):
                exists = True
                break

        if not exists:
            requirements.append(bioengine_requirement)

    return requirements
