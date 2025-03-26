from datetime import datetime, timezone
from typing import Dict


def format_time(
    timestamp: int, tz: timezone = timezone.utc
) -> Dict:
    """Format a timestamp into human-readable format with duration

    Args:
        timestamp: Unix timestamp
        tz: Timezone to use for formatting

    Returns:
        Dictionary with formatted time information
    """
    current_time = datetime.now(tz)
    start_time = datetime.fromtimestamp(timestamp, tz)

    duration = current_time - start_time
    days = duration.days
    seconds = duration.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    duration_parts = []
    if days > 0:
        duration_parts.append(f"{days}d")
    if hours > 0:
        duration_parts.append(f"{hours}h")
    if minutes > 0:
        duration_parts.append(f"{minutes}m")
    if remaining_seconds > 0:
        duration_parts.append(f"{remaining_seconds}s")

    return {
        "start_time": start_time.strftime("%Y/%m/%d %H:%M:%S") + f" ({tz})",
        "duration_since": " ".join(duration_parts) if duration_parts else "0s",
    }