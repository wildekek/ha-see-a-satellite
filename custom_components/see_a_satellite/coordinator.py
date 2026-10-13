"""Data coordinator for the See A Satellite Tonight integration."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, override

from see_a_satellite import (
    CatalogError,
    CloudCover,
    Satellite,
    TransitGroup,
    cloud_cover_at,
    dump_catalog,
    fetch_cloud_cover,
    fetch_site_catalog,
    group_transits,
    parse_site_catalog,
    predict_transits,
)

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ELEVATION, CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util
from homeassistant.util.hass_dict import HassKey

from .const import (
    CATALOG,
    CATALOG_REFRESH,
    CONF_CLOUD_COVER,
    CONF_MAX_MAGNITUDE,
    DEFAULT_MAX_MAGNITUDE,
    DOMAIN,
    LOGGER,
    PREDICTION_REFRESH,
)

type SeeASatelliteConfigEntry = ConfigEntry[SeeASatelliteCoordinator]

DATA_CATALOG: HassKey[SatelliteCatalog] = HassKey(DOMAIN)
STORAGE_VERSION = 1

# The largest catalog number a TLE can hold ("Z9999" in Alpha-5).
MAX_SGP4_SATNUM = 339999


def _propagatable(satellite: Satellite) -> Satellite:
    """Return the satellite with a catalog number SGP4 accepts.

    CelesTrak gives freshly launched satellites temporary 9-digit numbers,
    which sgp4 rejects although the number is only a label to SGP4.
    """
    if satellite.norad_id <= MAX_SGP4_SATNUM:
        return satellite
    return replace(satellite, omm={**satellite.omm, "NORAD_CAT_ID": "0"})


class SatelliteCatalog:
    """The catalog shared by all locations, persisted across restarts.

    The website's author hosts the catalog, so it is downloaded once for all
    locations and kept in storage instead of on every restart.
    """

    def __init__(self, hass: HomeAssistant, name: str) -> None:
        """Initialize the catalog."""
        self.hass = hass
        self.name = name
        self.satellites: list[Satellite] | None = None
        self.updated: datetime | None = None
        self._lock = asyncio.Lock()
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{name}_catalog"
        )

    async def async_get(self) -> list[Satellite]:
        """Return the catalog, refreshing it when it is due."""
        async with self._lock:
            if self.satellites is None:
                await self._async_load()
            now = dt_util.utcnow()
            if self.updated is None or now - self.updated >= CATALOG_REFRESH:
                try:
                    await self._async_refresh()
                except CatalogError as err:
                    if self.satellites is None:
                        raise
                    # Elements a few days old still predict well.
                    LOGGER.warning("Unable to refresh the satellite catalog: %s", err)
            assert self.satellites is not None
            return self.satellites

    async def _async_load(self) -> None:
        if (stored := await self._store.async_load()) is None:
            return
        self.satellites = parse_site_catalog(stored["catalog"])
        self.updated = dt_util.parse_datetime(stored["updated"])

    async def _async_refresh(self) -> None:
        self.satellites = await fetch_site_catalog(
            async_get_clientsession(self.hass), self.name
        )
        self.updated = dt_util.utcnow()
        self._store.async_delay_save(self._data_to_save, 10)

    @callback
    def _data_to_save(self) -> dict[str, Any]:
        assert self.updated is not None
        return {
            "updated": self.updated.isoformat(),
            "catalog": dump_catalog(self.satellites or ()),
        }


@callback
def async_get_catalog(hass: HomeAssistant) -> SatelliteCatalog:
    """Return the shared catalog."""
    if DATA_CATALOG not in hass.data:
        hass.data[DATA_CATALOG] = SatelliteCatalog(hass, CATALOG)
    return hass.data[DATA_CATALOG]


@dataclass(slots=True)
class SatellitePass:
    """A visible pass (possibly several satellites at once) with its weather."""

    group: TransitGroup
    bright: bool
    cloud_cover: int | None

    @property
    def start(self) -> datetime:
        """When the first satellite becomes visible."""
        return self.group.start

    @property
    def end(self) -> datetime:
        """When the last satellite stops being visible."""
        return self.group.end


@dataclass(slots=True)
class SatelliteData:
    """Predicted passes for one location."""

    passes: list[SatellitePass]
    catalog_updated: datetime | None
    catalog_size: int

    def current_or_next(
        self, now: datetime, predicate: Callable[[SatellitePass], bool] | None = None
    ) -> SatellitePass | None:
        """Return the pass in progress, or else the next one."""
        for p in self.passes:
            if p.end > now and (predicate is None or predicate(p)):
                return p
        return None

    def current(self, now: datetime) -> SatellitePass | None:
        """Return the pass in progress, if any."""
        p = self.current_or_next(now)
        return p if p is not None and p.start <= now else None


class SeeASatelliteCoordinator(DataUpdateCoordinator[SatelliteData]):
    """Predicts the visible passes for a location."""

    config_entry: SeeASatelliteConfigEntry

    def __init__(self, hass: HomeAssistant, entry: SeeASatelliteConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=PREDICTION_REFRESH,
        )
        self.latitude: float = entry.data[CONF_LATITUDE]
        self.longitude: float = entry.data[CONF_LONGITUDE]
        self.elevation: float = entry.data[CONF_ELEVATION]
        self.catalog = async_get_catalog(hass)
        self.max_magnitude: float = entry.options.get(
            CONF_MAX_MAGNITUDE, DEFAULT_MAX_MAGNITUDE
        )
        self.use_cloud_cover: bool = entry.options.get(CONF_CLOUD_COVER, True)
        self._unsub_boundary: CALLBACK_TYPE | None = None

    @override
    async def _async_update_data(self) -> SatelliteData:
        try:
            satellites = await self.catalog.async_get()
        except CatalogError as err:
            raise UpdateFailed(f"Unable to fetch the satellite catalog: {err}") from err
        cloud_cover = await self._async_cloud_cover()
        groups = await self.hass.async_add_executor_job(self._predict, satellites)
        data = SatelliteData(
            passes=[
                SatellitePass(
                    group=g,
                    bright=g.bright,
                    cloud_cover=cloud_cover_at(cloud_cover, g.start),
                )
                for g in groups
            ],
            catalog_updated=self.catalog.updated,
            catalog_size=len(satellites),
        )
        self._schedule_boundary(data)
        return data

    def _predict(self, satellites: list[Satellite]) -> list[TransitGroup]:
        transits = predict_transits(
            [_propagatable(s) for s in satellites],
            self.latitude,
            self.longitude,
            self.elevation,
            max_magnitude=self.max_magnitude,
        )
        return group_transits(transits, catalog_size=len(satellites))

    async def _async_cloud_cover(self) -> CloudCover | None:
        if not self.use_cloud_cover:
            return None
        try:
            return await fetch_cloud_cover(
                async_get_clientsession(self.hass), self.latitude, self.longitude
            )
        except CatalogError as err:
            LOGGER.debug("Unable to fetch the cloud cover forecast: %s", err)
            return None

    @callback
    def _schedule_boundary(self, data: SatelliteData) -> None:
        """Wake listeners when a pass starts or ends so states stay current."""
        if self._unsub_boundary:
            self._unsub_boundary()
            self._unsub_boundary = None
        now = dt_util.utcnow()
        upcoming = [t for p in data.passes for t in (p.start, p.end) if t > now]
        if not upcoming:
            return
        self._unsub_boundary = async_track_point_in_utc_time(
            self.hass, self._handle_boundary, min(upcoming) + timedelta(seconds=1)
        )

    @callback
    def _handle_boundary(self, _now: datetime) -> None:
        self._unsub_boundary = None
        self.async_update_listeners()
        if self.data is not None:
            self._schedule_boundary(self.data)

    @override
    async def async_shutdown(self) -> None:
        """Cancel the pass boundary timer."""
        if self._unsub_boundary:
            self._unsub_boundary()
            self._unsub_boundary = None
        await super().async_shutdown()
