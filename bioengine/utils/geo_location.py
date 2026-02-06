import logging
from typing import Dict, Optional

from .network import get_url_with_retry


def calculate_region_centroid_coordinates(
    region: Optional[str] = None,
    country: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, float]]:
    """
    Calculate the centroid coordinates (latitude and longitude) for a given location.

    Args:
        region (Optional[str]): The name of the region (e.g., state, province, city).
        country (Optional[str]): The name of the country.
    Returns:
        Optional[Dict[str, float]]: A dictionary with 'latitude' and 'longitude' keys,
        or None if the location is not found.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Build query string from most specific to least specific
    query_parts = []
    if region:
        query_parts.append(region)
    if country:
        query_parts.append(country)

    if not query_parts:
        logger.warning("No location information provided for centroid calculation")
        return None

    query = ", ".join(query_parts)

    latitude = None
    longitude = None
    try:
        response = get_url_with_retry(
            url="https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            raise_for_status=True,
            logger=logger,
        )
        data = response.json()
        if data:
            latitude = float(data[0]["lat"])
            longitude = float(data[0]["lon"])
    except Exception as e:
        logger.warning(
            f"Failed to calculate centroid coordinates for location: {query}. Error: {e}"
        )

    return {"latitude": latitude, "longitude": longitude}


def run_geolocation(logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """
    Fetch geo location information from ipapi.co.

    Attempts to retrieve geographical location data based on the current IP address.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    geo_info = {
        "region": None,
        "country_name": None,
        "country_code": None,
        "continent_code": None,
        "latitude": None,
        "longitude": None,
        "timezone": None,
    }

    try:
        response = get_url_with_retry(
            url="https://ipapi.co/json/", raise_for_status=True, logger=logger
        )
        data = response.json()

        geo_info = {
            "region": data.get("region"),
            "country_name": data.get("country_name"),
            "country_code": data.get("country_code") or data.get("country"),
            "continent_code": data.get("continent_code"),
            "timezone": data.get("timezone"),
        }
    except Exception as e:
        logger.error(f"Failed to fetch geo location information: {e}")

    # Calculate coordinates for the region
    coordinates = calculate_region_centroid_coordinates(
        region=geo_info.get("region"),
        country=geo_info.get("country_name"),
        logger=logger,
    )
    geo_info.update(coordinates)

    msg = (
        f"Geographic location detected: {geo_info['region']}, "
        f"{geo_info['country_name']}, "
        f"{geo_info['continent_code']} "
        f"(Timezone: {geo_info['timezone']})"
    )

    if coordinates["latitude"] and coordinates["longitude"]:
        msg += f" at ({geo_info['latitude']:.4f}, {geo_info['longitude']:.4f})"
    else:
        msg += " at (latitude/longitude not available)"

    logger.info(msg)

    return geo_info


if __name__ == "__main__":
    geo_info = run_geolocation()
    print(f"Result: {geo_info}")
