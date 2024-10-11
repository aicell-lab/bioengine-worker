from functools import wraps
import logging

class NotAdminError(Exception):
    """Custom exception for non-admin access."""
    pass

class AdminChecker:
    """Class to handle admin authentication."""
    
    def __init__(self, admin_emails):
        self.admin_emails = admin_emails

    def get_admin_email(self, context):
        """Extract email from context and check if it's an admin."""
        try:
            email = context['user']['email']
        except KeyError:
            raise KeyError("Email not provided in context")
        
        if email in self.admin_emails:
            return email  # Return email if admin
        else:
            raise NotAdminError(f"Access denied for user with email: {email}")
        
    def verify_context(self, context) -> bool:
        """Verify the context to check if the user is an admin."""
        email = self.get_admin_email(context)
        logging.info(f"Admin context for {email}!")
        return True

    def context_verification(self):
        """Decorator to verify the context for admin services."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, context=None, **kwargs):
                if self.verify_context(context):
                    return await func(*args, context=context, **kwargs)
                else:
                    raise NotAdminError("Unauthorized access")
            return wrapper
        return decorator
    
