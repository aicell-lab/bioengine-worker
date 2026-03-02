import asyncio
import logging
from typing import Dict, Optional

import httpx


async def _get(
    url: str,
    params: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    """Perform a single async GET request with a 10-second timeout."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response


async def fetch_centroid_coordinates(
    country: str,
    region: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[float]]:
    """
    Fetch the centroid coordinates (latitude and longitude) for a given location
    from the Nominatim geocoding API.

    Args:
        country: The name of the country (required).
        region: The name of the region (e.g., state, province, city).
        logger: Optional logger instance.
    Returns:
        A dictionary with 'latitude' and 'longitude' keys (values may be None on failure).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Build query string from most specific to least specific
    query_parts = []
    if region:
        query_parts.append(region)
    query_parts.append(country)

    query = ", ".join(query_parts)

    latitude = None
    longitude = None
    try:
        response = await _get(
            url="https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
        )
        data = response.json()
        if data:
            latitude = float(data[0]["lat"])
            longitude = float(data[0]["lon"])
            logger.info(
                f"Coordinates fetched for '{query}': ({latitude:.4f}, {longitude:.4f})"
            )
    except Exception as e:
        logger.warning(
            f"Failed to calculate centroid coordinates for location '{query}': {e}"
        )

    return {"latitude": latitude, "longitude": longitude}


async def fetch_geolocation(
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[str]]:
    """
    Fetch geo location information from ipapi.co.

    Attempts to retrieve geographical location data based on the current IP address.
    Coordinates (latitude/longitude) are NOT fetched here; call
    fetch_centroid_coordinates separately when country information is available.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    geo_info: Dict[str, Optional[str]] = {
        "region": None,
        "country_name": None,
        "country_code": None,
        "continent_code": None,
        "latitude": None,
        "longitude": None,
        "timezone": None,
    }

    try:
        response = await _get(url="https://ipapi.co/json/")
        data = response.json()

        geo_info.update(
            {
                "region": data.get("region"),
                "country_name": data.get("country_name"),
                "country_code": data.get("country_code") or data.get("country"),
                "continent_code": data.get("continent_code"),
                "timezone": data.get("timezone"),
            }
        )
        logger.info(
            f"Geographic location detected: {geo_info['region']}, "
            f"{geo_info['country_name']}, "
            f"{geo_info['continent_code']} "
            f"(Timezone: {geo_info['timezone']})"
        )
    except Exception as e:
        logger.error(f"Failed to fetch geo location information: {e}")

    return geo_info


if __name__ == "__main__":
    geo_info = asyncio.run(fetch_geolocation())
    print(f"Result: {geo_info}")
