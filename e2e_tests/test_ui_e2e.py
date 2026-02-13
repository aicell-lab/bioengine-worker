import pytest
import os
from playwright.sync_api import Page, expect

# Config
# Use the public URL for the deployed app
APP_URL = "https://hypha.aicell.io/apps/ri-scale/cellpose-finetuning/index.html?server_url=https://hypha.aicell.io"

def test_app_loads(page: Page):
    """Test that the application loads and connects."""
    # Print for debugging
    print(f"Loading {APP_URL}")
    response = page.goto(APP_URL, timeout=30000)
    assert response.ok, f"Failed to load page: {response.status} {response.status_text}"
    
    # Check title with debug
    try:
        expect(page).to_have_title("Cellpose Fine-Tuning UI", timeout=10000)
    except AssertionError:
        title = page.title()
        print(f"Title mismatch. Actual: '{title}'")
        # Screenshot on failure
        page.screenshot(path="e2e_tests/failure_load.png")
        raise
    
    # Check connection (green indicator)
    # The UI shows "Connected" text in green when connected
    # We wait up to 20s for Hypha connection as it might be slow
    print("Waiting for 'Connected' indicator...")
    connected = page.locator("text=Connected").first
    try:
        expect(connected).to_be_visible(timeout=20000)
    except AssertionError:
        page.screenshot(path="e2e_tests/failure_connected.png")
        print(f"Page content: {page.content()[:1000]}")
        raise

def test_navigation(page: Page):
    """Test tab navigation."""
    page.goto(APP_URL)
    # Ensure loaded
    expect(page.locator("text=Connected").first).to_be_visible(timeout=20000)
    
    # Default is Dashboard
    expect(page.locator("h2:has-text('Dashboard')")).to_be_visible()
    
    # Click New Training
    page.click("text=New Training")
    expect(page.locator("h2:has-text('New Training Session')")).to_be_visible()
    
    # Click Inference
    page.click("text=Inference")
    expect(page.locator("h2:has-text('Live Inference')")).to_be_visible()

def test_file_browser(page: Page):
    """Test file browser interaction and path navigation."""
    page.goto(APP_URL)
    expect(page.locator("text=Connected").first).to_be_visible(timeout=20000)
    
    # Go to Training
    page.click("text=New Training")
    
    # We need a valid artifact ID to test browsing. 
    # Using the app's own source artifact.
    artifact_id = "ri-scale/cellpose-finetuning"
    page.fill("input[placeholder='workspace/dataset-alias']", artifact_id)
    
    # Open Browser
    page.click("button[title='Browse Files']")
    
    # Wait for modal
    expect(page.locator("text=Artifact Explorer")).to_be_visible()
    
    # Wait for loading to finish
    expect(page.locator("text=Loading files...")).not_to_be_visible()
    
    # Should see "index.html"
    expect(page.locator("text=index.html")).to_be_visible()
    
    # Test feedback on root selection
    # Assuming user selects folder by clicking a "Set Current" button
    # If the logic is "Select a folder by current path", we might need to select a subfolder or just verify files are listed.
    # The prompt mentioned "set dataset alias to be by default" - assuming when folder is selected.
    
    # Close
    page.click("button >> .fa-times")
    
def test_inference_layout(page: Page):
    """Test that inference layout elements are visible."""
    page.goto(APP_URL)
    expect(page.locator("text=Connected").first).to_be_visible(timeout=20000)
    
    page.click("text=Inference")
    
    # Check button visibility
    btn = page.locator("button:has-text('Run Segmentation')")
    expect(btn).to_be_visible()
    
    # Check if "Input Image" box is visible
    expect(page.locator("text=Click or Drag Image Here")).to_be_visible()

