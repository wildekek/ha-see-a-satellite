"""Tests for setting up See A Satellite Tonight."""

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock

from freezegun.api import FrozenDateTimeFactory
import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from see_a_satellite import CatalogError, Satellite, dump_catalog

from custom_components.see_a_satellite.const import DOMAIN
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from .conftest import NOW


async def test_setup_and_unload(
    hass: HomeAssistant, init_integration: MockConfigEntry
) -> None:
    """The entry loads and unloads."""
    assert init_integration.state is ConfigEntryState.LOADED
    assert await hass.config_entries.async_unload(init_integration.entry_id)
    assert init_integration.state is ConfigEntryState.NOT_LOADED


async def test_catalog_unavailable(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_see_a_satellite: dict[str, AsyncMock],
) -> None:
    """Setup is retried while there is no catalog at all."""
    mock_see_a_satellite["fetch_site_catalog"].side_effect = CatalogError("boom")
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_catalog_restored_from_storage(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
    freezer: FrozenDateTimeFactory,
    mock_config_entry: MockConfigEntry,
    mock_see_a_satellite: dict[str, AsyncMock],
    satellites: list[Satellite],
) -> None:
    """A stored catalog is used when it can't be refreshed."""
    freezer.move_to(NOW)
    hass_storage[f"{DOMAIN}.brightest_catalog"] = {
        "version": 1,
        "key": f"{DOMAIN}.brightest_catalog",
        "data": {
            "updated": (NOW - timedelta(days=1)).isoformat(),
            "catalog": dump_catalog(satellites),
        },
    }
    mock_see_a_satellite["fetch_site_catalog"].side_effect = CatalogError("boom")
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.LOADED
    assert hass.states.get("sensor.home_next_pass").state == "2026-10-10T19:08:36+00:00"


async def test_fresh_stored_catalog_is_not_downloaded(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
    freezer: FrozenDateTimeFactory,
    mock_config_entry: MockConfigEntry,
    mock_see_a_satellite: dict[str, AsyncMock],
    satellites: list[Satellite],
) -> None:
    """A recently stored catalog avoids downloads after a restart."""
    freezer.move_to(NOW)
    hass_storage[f"{DOMAIN}.brightest_catalog"] = {
        "version": 1,
        "key": f"{DOMAIN}.brightest_catalog",
        "data": {
            "updated": (NOW - timedelta(hours=1)).isoformat(),
            "catalog": dump_catalog(satellites),
        },
    }
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.LOADED
    mock_see_a_satellite["fetch_site_catalog"].assert_not_awaited()


async def test_catalog_shared_and_stored(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
    freezer: FrozenDateTimeFactory,
    mock_see_a_satellite: dict[str, AsyncMock],
) -> None:
    """All locations share one catalog download, which is stored."""
    freezer.move_to(NOW)
    entries = [
        MockConfigEntry(
            domain=DOMAIN,
            title=title,
            minor_version=2,
            data={"latitude": lat, "longitude": 5.5, "elevation": 0},
        )
        for title, lat in (("Home", 52.0), ("Cabin", 60.0))
    ]
    for entry in entries:
        entry.add_to_hass(hass)
        await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert all(entry.state is ConfigEntryState.LOADED for entry in entries)
    mock_see_a_satellite["fetch_site_catalog"].assert_awaited_once()

    freezer.tick(timedelta(seconds=11))
    async_fire_time_changed(hass)
    await hass.async_block_till_done()
    stored = hass_storage[f"{DOMAIN}.brightest_catalog"]["data"]
    assert len(stored["catalog"]) == 22


async def test_temporary_catalog_number(
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    mock_config_entry: MockConfigEntry,
    mock_see_a_satellite: dict[str, AsyncMock],
    satellites: list[Satellite],
) -> None:
    """A just-launched satellite's temporary 9-digit number is predicted."""
    bluewalker = next(s for s in satellites if s.norad_id == 53807)
    twin = Satellite.from_omm(
        {**bluewalker.omm, "NORAD_CAT_ID": "799501674", "OBJECT_NAME": "Launch A"},
        bluewalker.magnitude,
        bluewalker.model,
    )
    satellites.insert(0, twin)
    freezer.move_to(NOW)
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.LOADED
    state = hass.states.get("sensor.home_next_pass")
    assert state.state == "2026-10-10T19:08:36+00:00"
    assert state.attributes["satellite"] == "Launch A"
    assert state.attributes["norad_id"] == 799501674
    assert state.attributes["satellites"] == ["Launch A", "BlueWalker 3"]


@pytest.mark.usefixtures("init_integration")
async def test_cloud_cover_failure_is_tolerated(
    hass: HomeAssistant,
    freezer: FrozenDateTimeFactory,
    mock_see_a_satellite: dict[str, AsyncMock],
) -> None:
    """Passes are still predicted without a cloud cover forecast."""
    mock_see_a_satellite["fetch_cloud_cover"].side_effect = CatalogError("boom")
    freezer.tick(timedelta(hours=2, seconds=1))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert mock_see_a_satellite["fetch_cloud_cover"].await_count == 2
    state = hass.states.get("sensor.home_next_pass")
    assert state.state != "unavailable"
    assert state.attributes["cloud_cover"] is None


async def test_migrate_brightest_entry(
    hass: HomeAssistant, mock_see_a_satellite: dict[str, AsyncMock]
) -> None:
    """Entries from before 1.2 lose the catalog choice."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home",
        unique_id="52.0907_5.1214_brightest",
        data={
            "latitude": 52.0907,
            "longitude": 5.1214,
            "elevation": 5,
            "catalog": "brightest",
        },
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.LOADED
    assert entry.minor_version == 2
    assert entry.unique_id == "52.0907_5.1214"
    assert entry.data == {"latitude": 52.0907, "longitude": 5.1214, "elevation": 5}


@pytest.mark.usefixtures("mock_see_a_satellite")
async def test_starlink_entry_not_migrated(hass: HomeAssistant) -> None:
    """Starlink train entries can't be migrated and must be removed."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Starlink",
        unique_id="52.0907_5.1214_starlink",
        data={
            "latitude": 52.0907,
            "longitude": 5.1214,
            "elevation": 5,
            "catalog": "starlink",
        },
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.MIGRATION_ERROR
