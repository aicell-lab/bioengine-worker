import logging
import time
from typing import Dict, Optional

import httpx


def calculate_region_centroid_coordinates(
    region: Optional[str] = None,
    country: Optional[str] = None,
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
    # Build query string from most specific to least specific
    query_parts = []
    if region:
        query_parts.append(region)
    if country:
        query_parts.append(country)

    if not query_parts:
        print("⚠️ No location information provided for centroid calculation")
        return None

    query = ", ".join(query_parts)

    max_retries = 4
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": query, "format": "json", "limit": 1},
                )
                response.raise_for_status()
                data = response.json()
                if data:
                    latitude = float(data[0]["lat"])
                    longitude = float(data[0]["lon"])
                    return {"latitude": latitude, "longitude": longitude}
                else:
                    # No data found, no need to retry
                    break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                print(
                    f"⚠️ Attempt {attempt + 1}/{max_retries} failed for location '{query}': {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                print(
                    f"⚠️ Failed to calculate centroid for location '{query}' after {max_retries} attempts: {e}"
                )
    return None


def fetch_geo_location(logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """
    Fetch geo location information from ipapi.co.

    Attempts to retrieve geographical location data based on the current IP address.
    """

    geo_info = {
        "region": None,
        "country_name": None,
        "country_code": None,
        "continent_code": None,
        "latitude": None,
        "longitude": None,
        "timezone": None,
    }

    max_retries = 4
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get("https://ipapi.co/json/")
                response.raise_for_status()
                data = response.json()

                geo_info = {
                    "region": data.get("region"),
                    "country_name": data.get("country_name"),
                    "country_code": data.get("country_code") or data.get("country"),
                    "continent_code": data.get("continent_code"),
                    "timezone": data.get("timezone"),
                }
                break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                msg = f"Attempt {attempt + 1}/{max_retries} failed to fetch geo location: {e}. Retrying in {wait_time}s..."
                if logger:
                    logger.warning(msg)
                else:
                    print(f"⚠️ {msg}")
                time.sleep(wait_time)
            else:
                msg = f"Failed to fetch geo location information after {max_retries} attempts: {e}"
                if logger:
                    logger.error(msg)
                else:
                    print(f"⚠️ {msg}")

    # Calculate coordinates for the region
    coordinates = calculate_region_centroid_coordinates(
        region=geo_info.get("region"),
        country=geo_info.get("country_name"),
    )

    if coordinates:
        geo_info["latitude"] = coordinates["latitude"]
        geo_info["longitude"] = coordinates["longitude"]
    else:
        geo_info["latitude"] = None
        geo_info["longitude"] = None

    msg = (
        f"Geo location detected: {geo_info['region']}, "
        f"{geo_info['country_name']}, "
        f"{geo_info['continent_code']} "
        f"(Timezone: {geo_info['timezone']})"
    )

    if coordinates:
        msg += f" at ({geo_info['latitude']:.4f}, {geo_info['longitude']:.4f})"

    if logger:
        logger.info(msg)
    else:
        print(f"🌍 {msg}")

    return geo_info


if __name__ == "__main__":
    geo_info = fetch_geo_location()
    print(f"Result: {geo_info}")
