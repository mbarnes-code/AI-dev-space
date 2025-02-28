class BaseMCPError(Exception):
    """Base class for all MCP-related errors."""
    pass

class ConfigurationError(BaseMCPError):
    """Raised when there is a configuration-related error."""
    pass

class ConnectionError(BaseMCPError):
    """Raised when there is a connection-related error."""
    pass

class ToolError(BaseMCPError):
    """Raised when there is an error related to tool operations."""
    pass

class DatabaseConnectionError(BaseMCPError):
    """Raised when there is a database connection-related error."""
    pass

class AuthenticationError(BaseMCPError):
    """Raised when there is an authentication-related error."""
    pass

class AuthorizationError(BaseMCPError):
    """Raised when there is an authorization-related error."""
    pass

class ValidationError(BaseMCPError):
    """Raised when there is a validation-related error."""
    pass

class APIError(BaseMCPError):
    """Raised when there is an error related to API operations."""
    pass

class FileNotFoundError(BaseMCPError):
    """Raised when a required file is not found."""
    pass

class TimeoutError(BaseMCPError):
    """Raised when an operation times out."""
    pass

class ServiceUnavailableError(BaseMCPError):
    """Raised when a service is unavailable."""
    pass
