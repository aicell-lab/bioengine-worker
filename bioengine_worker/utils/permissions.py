from typing import Dict, Union, List


def create_context(user_id: str = None, user_email: str = None) -> Dict[str, List[str]]:
    """
    Get context for admin users.

    Returns:
        Dict: Context dictionary containing admin user information
    """
    return {
        "user": {
            "id": user_id if user_id else "anonymous",
            "email": user_email if user_email else "anonymous@example.com",
        }
    }


def check_permissions(
    context: Dict[str, str],
    authorized_users: Union[List[str], str],
    resource_name: str,
) -> None:
    """
    Check if the user in the context is authorized to access the deployment.

    Validates user permissions against the authorized users list for specific
    deployment operations.

    Args:
        context: Request context containing user information
        authorized_users: List of authorized user IDs/emails or single user string
        resource_name: Name of the resource being accessed for logging

    Returns:
        bool: True if user is authorized

    Raises:
        PermissionError: If user is not authorized to access the resource
    """
    if context is None or "user" not in context:
        raise PermissionError("Context is missing user information")
    user = context["user"]
    if isinstance(authorized_users, str):
        authorized_users = [authorized_users]
    if (
        "*" not in authorized_users
        and user["id"] not in authorized_users
        and user["email"] not in authorized_users
    ):
        raise PermissionError(
            f"User {user['id']} is not authorized to {resource_name}."
        )
