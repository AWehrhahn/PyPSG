# ---------------------------------------------------
# Planet Spectrum Generator Interface
# PSG multiple-scattering model: https://psg.gsfc.nasa.gov/helpmodel.php
# PSG databases: https://psg.gsfc.nasa.gov/helpatm.php
# PSG API driver: https://psg.gsfc.nasa.gov/helpapi.php
# ---------------------------------------------------
from io import StringIO
from os.path import dirname, join
import subprocess
from datetime import datetime

import re
import hashlib
from tempfile import NamedTemporaryFile
import numpy as np

from astropy.utils.data import (
    import_file_to_cache,
    download_file,
    clear_download_cache,
    is_url_in_cache,
)
from astropy import units as u

# Add Atmosphere scale units
ppb = u.def_unit(
    ["ppb", "ppbv"], 1e-9 * u.one, namespace=globals(), doc="Parts Per Billion"
)
ppm = u.def_unit(
    ["ppm", "ppmv", "ppv"],
    1e3 * ppb,
    namespace=globals(),
    doc="Parts Per Million Volume",
)
ppt = u.def_unit(
    ["ppt", "pptv"], 1e6 * ppb, namespace=globals(), doc="Parts Per Thousand Volume"
)
m2 = u.def_unit(
    ["m2", "m-2"], None, namespace=globals(), doc="Molecules per square meter"
)
diameter = u.def_unit(
    ["diameter"], None, namespace=globals(), doc="Diameter of the telescope"
)
diffrac = u.def_unit(
    ["diffrac"],
    None,
    namespace=globals(),
    doc="defined by the telescope diameter and center wavelength",
)

scl = u.def_unit(["scl"], None, namespace=globals(), doc="Relative Scale")
u.add_enabled_units([ppb, ppm, ppt, m2, scl, diameter, diffrac])

# Package Name for the Astropy cache
PKGNAME = "planet-spectrum-generator"


class PSG_Config:
    def __init__(self, config) -> None:
        # Store all the other atmosphere parameters
        self.other = {}
        for key, value in config.items():
            self.other[key] = value

    @staticmethod
    def get_value(config, key, func=None):
        try:
            value = config[key]
            if func is not None:
                value = func(value)
        except KeyError:
            value = None
        return value

    def get_quantity(self, config, key, unit):
        return self.get_value(config, key, lambda x: float(x) * unit)

    def get_bool(self, config, key, true_value="Y"):
        return self.get_value(config, key, lambda x: x == true_value)

    def get_list(self, config, key, func=None, array=False, sep=","):
        value = self.get_value(config, key, lambda x: x.split(","))
        if value is None:
            return None
        if func is not None:
            value = [func(v) for v in value]
        if array:
            value = np.array(value)
        return value

    @staticmethod
    def parse_units(unit, units, names):
        if unit is None:
            return None
        for u, n in zip(units, names):
            if unit == n:
                return u
        raise ValueError("Could not parse unit")

    @staticmethod
    def get_units(unit, units, names):
        if unit is None:
            return None, None
        for un, n in zip(units, names):
            if unit == un:
                return un, n

        for un, n in zip(units, names):
            try:
                unit.to(un)
                return un, n
            except u.core.UnitConversionError:
                continue
        raise ValueError("Could not determine units")

    def to_config(self):
        return self.other


class PSG_Object(PSG_Config):
    def __init__(self, config) -> None:
        #:str: Object type (e.g., Exoplanet, Planet, Asteroid, Moon, Comet, Object)
        self.object = config.get("OBJECT")
        #:str: Object name
        self.name = config.get("OBJECT-NAME")
        # Datetime
        if "OBJECT-DATE" in config.keys():
            match = re.match(
                r"(\d{4})/(\d{2})/(\d{2}) (\d{2}):(\d{2})", config["OBJECT-DATE"]
            )
            date = datetime(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
                int(match.group(5)),
            )
        else:
            date = None
        #:datetime: Date of the observation (yyyy/mm/dd hh:mm) in Universal time [UT]
        self.date = date
        #:Quantity: Diameter of the object [km]
        self.diameter = self.get_quantity(config, "OBJECT-DIAMETER", u.km)
        # Gravity
        gravity_unit = config.get("OBJECT-GRAVITY-UNIT")
        if gravity_unit == "g":  # Surface Gravity
            gravity_unit = u.m / u.s ** 2
        elif gravity_unit == "rho":  # Mean Density
            gravity_unit = u.g / u.cm ** 3
        elif gravity_unit == "kg":  # Total Mass
            gravity_unit = u.kg
        #:Quantity: Gravity/density/mass of the object
        self.gravity = self.get_quantity(config, "OBJECT-GRAVITY", gravity_unit)
        #:Quantity: Distance of the planet to the Sun [AU], and for exoplanets the semi-major axis [AU]
        self.star_distance = self.get_quantity(config, "OBJECT-STAR-DISTANCE", u.AU)
        #:Quantity: Velocity of the planet to the Sun [km/s], and for exoplanets the RV amplitude [km/s]
        self.star_velocity = self.get_quantity(config, "OBJECT-STAR-VELOCITY", (u.km / u.s))
        #:Quantity: Sub-solar east longitude [degrees]
        self.solar_longitude = self.get_quantity(config, "OBJECT-SOLAR-LONGITUDE", u.deg)
        #:Quantity: Sub-solar latitude [degrees]
        self.solar_latitude = self.get_quantity(config, "OBJECT-SOLAR-LATITUDE", u.deg)
        #:Quantity: Angular parameter (season/phase) that defines the position of the planet moving along its Keplerian orbit. For exoplanets, 0:Secondary transit, 180:Primary transit, 90/270:Opposition. For solar-system bodies, 0:'N spring equinox', 90:'N summer solstice', 180:'N autumn equinox', 270:'N winter solstice' [degrees]
        self.season = self.get_quantity(config, "OBJECT-SEASON", u.deg)
        #:Quantity: Orbital inclination [degree], mainly relevant for exoplanets. Zero is phase on, 90 is a transiting orbit
        self.inclination = self.get_quantity(config, "OBJECT-INCLINATION", u.deg)
        #:float: Orbital eccentricity, mainly relevant for exoplanets
        self.eccentricity = self.get_value(config, "OBJECT-ECCENTRICITY", float)
        #:Quantity: Orbital longitude of periapse [degrees]. It indicates the phase at which the planet reaches periapsis
        self.periapsis = self.get_quantity(config, "OBJECT-PERIAPSIS", u.deg)
        #:str: Stellar type of the parent star [O/B/A/F/G/K/M]
        self.star_type = config.get("OBJECT-STAR-TYPE")
        #:Quantity:  Temperature of the parent star [K]
        self.star_temperature = self.get_quantity(config, "OBJECT-STAR-TEMPERATURE", u.K)
        #:Quantity: Radius of the parent star [Rsun]
        self.star_radius = self.get_quantity(config, "OBJECT-STAR-RADIUS", u.Rsun)
        #:float: Metallicity of the parent star and object with respect to the Sun in log [dex]
        self.star_metallicity = self.get_value(config, "OBJECT-STAR-METALLICITY", float)
        #:Quantity: Sub-observer east longitude [degrees]
        self.obs_longitude = self.get_quantity(config, "OBJECT-OBS-LONGITUDE", u.deg)
        #:Quantity: Sub-observer latitude, for exoplanets inclination [degrees]
        self.obs_latitude = self.get_quantity(config, "OBJECT-OBS-LATITUDE", u.deg)
        #:Quantity: Relative velocity between the observer and the object [km/s]
        self.obs_velocity = self.get_quantity(config, "OBJECT-OBS-VELOCITY", u.km / u.s)
        #:Quantity: This field is computed by the geometry module - It is the apparent rotational period of the object as seen from the observer [days]
        self.period = self.get_quantity(config, "OBJECT-PERIOD", u.day)
        #:str: This field reports the orbital parameters for small bodies. It is only relevant for display purposes of the orbital diagram.
        self.orbit = config.get("OBJECT-ORBIT")

    @property
    def gravity_unit(self):
        """ Unit for the OBJECT-GRAVITY field, g:'Surface gravity [m/s2]', rho:'Mean density [g/cm3]', or kg:'Total mass [kg]' """
        return self.gravity.unit

    def to_config(self):
        gravity_unit_loc, gravity_unit = self.get_units(
            self.gravity_unit,
            [u.m / u.s ** 2, u.g / u.cm ** 3, u.kg],
            ["g", "rho", "kg"],
        )

        config = {
            "OBJECT": self.object,
            "OBJECT-NAME": self.name,
            "OBJECT-DATE": f"{self.date.year:04}/{self.date.month:02}/{self.date.day:02} {self.date.hour:02}:{self.date.minute:02}" if self.date is not None else None,
            "OBJECT-DIAMETER": self.diameter.to_value(u.km) if self.diameter is not None else None,
            "OBJECT-GRAVITY": self.gravity.to_value(gravity_unit_loc) if self.gravity is not None and gravity_unit_loc is not None else None,
            "OBJECT-GRAVITY-UNIT": gravity_unit,
            "OBJECT-STAR-DISTANCE": self.star_distance.to_value(u.AU) if self.star_distance is not None else None,
            "OBJECT-STAR-VELOCITY": self.star_velocity.to_value(u.km / u.s) if self.star_velocity is not None else None,
            "OBJECT-SOLAR-LONGITUDE": self.solar_longitude.to_value(u.deg) if self.solar_longitude is not None else None,
            "OBJECT-SOLAR-LATITUDE": self.solar_latitude.to_value(u.deg) if self.solar_latitude is not None else None,
            "OBJECT-SEASON": self.season.to_value(u.deg) if self.season is not None else None,
            "OBJECT-INCLINATION": self.inclination.to_value(u.deg) if self.inclination is not None else None,
            "OBJECT-ECCENTRICITY": self.eccentricity,
            "OBJECT-PERIAPSIS": self.periapsis.to_value(u.deg) if self.periapsis is not None else None,
            "OBJECT-STAR-TYPE": self.star_type,
            "OBJECT-STAR-TEMPERATURE": self.star_temperature.to_value(u.K) if self.star_temperature is not None else None,
            "OBJECT-STAR-RADIUS": self.star_radius.to_value(u.Rsun) if self.star_radius is not None else None,
            "OBJECT-STAR-METALLICITY": self.star_metallicity,
            "OBJECT-OBS-LONGITUDE": self.obs_longitude.to_value(u.deg) if self.obs_longitude is not None else None,
            "OBJECT-OBS-LATITUDE": self.obs_latitude.to_value(u.deg) if self.obs_latitude is not None else None,
            "OBJECT-OBS-VELOCITY": self.obs_velocity.to_value(u.km / u.s) if self.obs_velocity is not None else None,
            "OBJECT-PERIOD": self.period.to_value(u.day) if self.period is not None else None,
            "OBJECT-ORBIT": self.orbit,
        }
        config = {k: str(v) for k, v in config.items() if v is not None}
        return config


class PSG_Geometry(PSG_Config):
    def __init__(self, config) -> None:
        #:str: Type of observing geometry
        self.geometry = config.get("GEOMETRY")
        #:str: Reference geometry (e.g., ExoMars, Maunakea), default is user defined or 'User'
        self.ref = config.get("GEOMETRY-REF")

        offset_unit = config.get("GEOMETRY-OFFSET-UNIT")
        offset_unit = self.parse_units(offset_unit, [u.arcsec, u.arcmin, u.deg, u.km, u.one],
            ["arcsec", "arcmin", "degree", "km", "diameter"],)
        #:quantity: Vertical offset with respect to the sub-observer location
        self.offset_ns = self.get_quantity(config, "GEOMETRY-OFFSET-NS", offset_unit)
        #:quantity: Horizontal offset with respect to the sub-observer location
        self.offset_ew = self.get_quantity(config, "GEOMETRY-OFFSET-EW", offset_unit)

        altitude_unit = config.get("GEOMETRY-ALTITUDE-UNIT")
        altitude_unit = self.parse_units(altitude_unit, [u.AU, u.km, u.one, u.pc], ["AU", "km", "diameter", "pc"])
        #:quantity: Distance between the observer and the surface of the planet
        self.obs_altitude = self.get_quantity(config, "GEOMETRY-OBS-ALTITUDE", altitude_unit)
        #:quantity: The azimuth angle between the observational projected vector and the solar vector on the reference plane
        self.azimuth = self.get_quantity(config, "GEOMETRY-AZIMUTH", u.deg)
        #:float: Parameter for the selected geometry, for Nadir / Lookingup this field indicates the zenith angle [degrees], for limb / occultations this field indicates the atmospheric height [km] being sampled
        self.user_param = self.get_value(config, "GEOMETRY-USER-PARAM", float)
        #:str: For stellar occultations, this field indicates the type of the occultation star [O/B/A/F/G/K/M]
        self.stellar_type = config.get("GEOMETRY-STELLAR-TYPE")
        #:quantity: For stellar occultations, this field indicates the temperature [K] of the occultation star
        self.stellar_temperature = self.get_quantity(config, "GEOMETRY-STELLAR-TEMPERATURE", u.K)
        #:quantity: For stellar occultations, this field indicates the brightness [magnitude] of the occultation star
        self.stellar_magnitude = self.get_quantity(config, "GEOMETRY-STELLAR-MAGNITUDE", u.mag)
        #:str: This field is computed by the geometry module - It indicates the angle between the observer and the planetary surface
        self.obs_angle = config.get("GEOMETRY-OBS-ANGLE")
        #:str: This field is computed by the geometry module - It indicates the angle between the Sun and the planetary surface
        self.solar_angle = config.get("GEOMETRY-SOLAR-ANGLE")
        #:int: This field allows to divide the observable disk in finite rings so radiative-transfer calculations are performed with higher accuracy
        self.disk_angles = int(config.get("GEOMETRY-DISK-ANGLES"))
        #:quantity: This field is computed by the geometry module - It indicates the phase between the Sun and observer
        self.phase = self.get_quantity(config, "GEOMETRY-PHASE", u.deg)
        #:str: This field is computed by the geometry module - It indicates how much the beam fills the planetary area (1:maximum)
        self.planet_fraction = config.get("GEOMETRY-PLANET-FRACTION")
        #:float: This field is computed by the geometry module - It indicates how much the beam fills the parent star (1:maximum)
        self.star_fraction = self.get_value(config, "GEOMETRY-STAR-FRACTION", float)
        #:float: This field is computed by the geometry module - It indicates the projected distance between the beam and the parent star in arcsceconds
        self.star_distance = self.get_value(config, "GEOMETRY-STAR-DISTANCE", float)
        #:str: This field is computed by the geometry module - It indicates the rotational Doppler shift [km/s] affecting the spectra and the spread of rotational velocities [km/s] within the FOV
        self.rotation = config.get("GEOMETRY-ROTATION")
        #:str: This field is computed by the geometry module - It indicates the scaling factor between the integrated reflectance for the FOV with respect to the BRDF as computed using the geometry indidence/emission angles
        self.brdfscaler = config.get("GEOMETRY-BRDFSCALER")

    @property
    def offset_unit(self):
        """ Unit of the GEOMETRY-OFFSET field, arcsec / arcmin / degree / km / diameter """
        return self.offset_ns.unit

    @property
    def altitude_unit(self):
        """ Unit of the GEOMETRY-OBS-ALTITUDE field, AU / km / diameter and pc:'parsec' """
        return self.obs_altitude.unit

    def to_config(self):
        loc_offset_unit, offset_unit = self.get_units(
            self.offset_unit,
            [u.arcsec, u.arcmin, u.deg, u.km, u.one],
            ["arcsec", "arcmin", "degree", "km", "diameter"],
        )
        loc_altitude_unit, altitude_unit = self.get_units(
            self.altitude_unit,
            [u.AU, u.km, u.pc, u.one],
            ["AU", "km", "pc", "diameter"],
        )

        config = {
            "GEOMETRY": self.geometry,
            "GEOMETRY-REF": self.ref,
            "GEOMETRY-OFFSET-NS": self.offset_ns.to_value(loc_offset_unit) if self.offset_ns is not None and loc_offset_unit is not None else None,
            "GEOMETRY-OFFSET-EW": self.offset_ew.to_value(loc_offset_unit) if self.offset_ew is not None and loc_offset_unit is not None else None,
            "GEOMETRY-OFFSET-UNIT": offset_unit,
            "GEOMETRY-OBS-ALTITUDE": self.obs_altitude.to_value(loc_altitude_unit) if self.obs_altitude is not None and loc_altitude_unit is not None else None,
            "GEOMETRY-ALTITUDE-UNIT": altitude_unit,
            "GEOMETRY-AZIMUTH": self.azimuth.to_value(u.deg) if self.azimuth is not None else None,
            "GEOMETRY-USER-PARAM": self.user_param,
            "GEOMETRY-STELLAR-TYPE": self.stellar_type,
            "GEOMETRY-STELLAR-TEMPERATURE": self.stellar_temperature.to_value(u.K) if self.stellar_temperature is not None else None,
            "GEOMETRY-STELLAR-MAGNITUDE": self.stellar_magnitude.to_value(u.mag) if self.stellar_magnitude is not None else None,
            "GEOMETRY-OBS-ANGLE": self.obs_angle,
            "GEOMETRY-SOLAR-ANGLE": self.solar_angle,
            "GEOMETRY-DISK-ANGLES": self.disk_angles,
            "GEOMETRY-PHASE": self.phase.to_value(u.deg) if self.phase is not None else None,
            "GEOMETRY-PLANET-FRACTION": self.planet_fraction,
            "GEOMETRY-STAR-FRACTION": self.star_fraction,
            "GEOMETRY-STAR-DISTANCE": self.star_distance,
            "GEOMETRY-ROTATION": self.rotation,
            "GEOMETRY-BRDFSCALER": self.brdfscaler,
        }
        config = {k: str(v) for k, v in config.items() if v is not None}
        return config


class PSG_Atmosphere(PSG_Config):
    # from https://hitran.org/docs/molec-meta/
    hitran_molecule_id = {
        "H2O": 1,
        "CO2": 2,
        "O3": 3,
        "N2O": 4,
        "CO": 5,
        "CH4": 6,
        "O2": 7,
        "NO": 8,
        "SO2": 9,
        "NO2": 10,
        "NH3": 11,
        "HNO3": 12,
        "OH": 13,
        "HF": 14,
        "HCl": 15,
        "HBr": 16,
        "HI": 17,
        "ClO": 18,
        "OCS": 19,
        "H2CO": 20,
        "HOCl": 21,
        "N2": 22,
        "HCN": 23,
        "CH3Cl": 24,
        "H2O2": 25,
        "C2H2": 26,
        "C2H6": 27,
        "PH3": 28,
        "COF2": 29,
        "SF6": 30,
        "H2S": 31,
        "HCOOH": 32,
        "HO2": 33,
        "O": 34,
        "ClONO2": 35,
        "NO+": 36,
        "HOBr": 37,
        "C2H4": 38,
        "CH3OH": 39,
        "CH3Br": 40,
        "CH3CN": 41,
        "CF4": 42,
        "C4H2": 43,
        "HC3N": 44,
        "H2": 45,
        "CS": 46,
        "SO3": 47,
        "C2N2": 48,
        "COCl2": 49,
        "CS2": 53,
        "NF3": 55,
    }

    def __init__(self, config) -> None:
        # Handle the component molecules
        self._gas = self.get_list(config, "ATMOSPHERE-GAS")
        #:list(str): Sub-type of the gases, e.g. 'HIT[1], HIT[2]'
        self.type = self.get_list(config, "ATMOSPHERE-TYPE")
        abun = self.get_list(config, "ATMOSPHERE-ABUN", func=float)
        unit = self.get_list(config, "ATMOSPHERE-UNIT", func=u.Unit)
        #:list(quantity): Abundance of gases. The values can be assumed to be same across all altitudes/layers [%,ppmv,ppbv,pptv,m-2], or as a multiplier [scl] to the provided vertical profile
        self.abun = [a * u for a, u in zip(abun, unit)] if abun is not None and unit is not None else None

        # Handle the Aerosols
        self._aeros = self.get_list(config, "ATMOSPHERE-AEROS")
        #:list(str): Sub-type of the aerosols
        self.atype = self.get_list(config, "ATMOSPHERE-ATYPE")
        abun = self.get_list(config, "ATMOSPHERE-AABUN", func=float)
        unit = self.get_list(config, "ATMOSPHERE-AUNIT", func=u.Unit)
        #:list(quantity): Abundance of aerosols. The values can be assumed to be same across all altitudes/layers [%,ppm,ppb,ppt,Kg/m2], or as a multiplier [scaler] to the provided vertical profile
        self.aabun = [a * u for a, u in zip(abun, unit)] if abun is not None and unit is not None else None
        size = self.get_list(config, "ATMOSPHERE-ASIZE", func=float)
        unit = self.get_list(config, "ATMOSPHERE-ASUNI", func=float)
        #:list(quantity): Effective radius of the aerosol particles. The values can be assumed to be same across all layers [um, m, log(um)], or as a multiplier [scaler] to the provided size vertical profile
        self.asize = [a * u for a, u in zip(size, unit)] if size is not None and unit is not None else None

        # Handle the atmosphere layers
        #:list(str): Molecules quantified by the vertical profile
        self.layers_molecules = self.get_list(config, "ATMOSPHERE-LAYERS-MOLECULES")
        nlayers = self.get_value(config, "ATMOSPHERE-LAYERS", int)
        if nlayers is not None:
            try:
                layers = [config[f"ATMOSPHERE-LAYER-{i}"] for i in range(1, nlayers + 1)]
                layers = StringIO("\n".join(layers))
                layers = np.genfromtxt(layers, delimiter=",")
            except KeyError:
                layers = None
        else:
            layers = None
        #:array: Values for that specific layer: Pressure[bar], Temperature[K], gases[mol/mol], aerosols [kg/kg] - Optional fields: Altitude[km], X_size[m, aerosol X particle size]
        self.layer = layers
        #:str: Parameters defining the 3D General Circulation Model grid: num_lons, num_lats, num_alts, lon0, lat0, delta_lon, delta_lat, variables (csv)
        self.gcm_parameters = config.get("ATMOSPHERE-GCM-PARAMETERS")
        #:str: The structure of the atmosphere, None / Equilibrium:'Hydrostatic equilibrium' / Coma:'Cometary expanding coma'
        self.structure = config.get("ATMOSPHERE-STRUCTURE")
        #:float: For equilibrium atmospheres, this field defines the surface pressure; while for cometary coma, this field indicates the gas production rate
        self.pressure = self.get_value(config, "ATMOSPHERE-PRESSURE", float)
        #:str: The unit of the ATMOSPHERE-PRESSURE field, Pa:Pascal / bar / kbar / mbar / ubar / at / atm / torr / psi / gas:'molecules / second' / gasau:'molecules / second at rh=1AU'
        self.punit = config.get("ATMOSPHERE-PUNIT")
        #:quantity: For atmospheres without a defined P/T profile, this field indicates the temperature across all altitudes
        self.temperature = self.get_quantity(config, "ATMOSPHERE-TEMPERATURE", u.K)
        #:float: Molecular weight of the atmosphere [g/mol] or expansion velocity [m/s] for expanding atmospheres
        self.weight = self.get_value(config, "ATMOSPHERE-WEIGHT", float)
        #:str: Continuum processes to be included in the calculation
        self.continuum = config.get("ATMOSPHERE-CONTINUUM")
        #:str: For expanding cometary coma, this field indicates the photodissociation lifetime of the molecules [s]
        self.tau = config.get("ATMOSPHERE-TAU")
        #:int: When performing scattering aerosols calculations, this parameter indicates the number of n-stream pairs - Use 0 for extinction calculations only (e.g. transit, occultation)
        self.nmax = self.get_value(config, "ATMOSPHERE-NMAX", int)
        #:int: When performing scattering aerosols calculations, this parameter indicates the number of scattering Legendre polynomials used for describing the phase function - Use 0 for extinction calculations only (e.g. transit, occultation)
        self.lmax = self.get_value(config, "ATMOSPHERE-LMAX", int)
        #:str: Description establishing the source/reference for the vertical profile
        self.description = config.get("ATMOSPHERE-DESCRIPTION")

    def to_config(self):
        config = {
            "ATMOSPHERE-NGAS": self.ngas,
            "ATMOSPHERE-GAS": ",".join([str(v) for v in self.gas]) if self.gas is not None else None,
            "ATMOSPHERE-TYPE": ",".join([str(v) for v in self.type]) if self.type is not None else None,
            "ATMOSPHERE-ABUN": ",".join([str(v.value) for v in self.abun]) if self.abun is not None else None,
            "ATMOSPHERE-UNIT": ",".join([str(v.unit) for v in self.abun]) if self.abun is not None else None,
            "ATMOSPHERE-NAERO": self.naero,
            "ATMOSPHERE-AEROS": ",".join([str(v) for v in self.aeros]) if self.aeros is not None else None,
            "ATMOSPHERE-ATYPE": ",".join([str(v) for v in self.atype]) if self.atype is not None else None,
            "ATMOSPHERE-AABUN": ",".join([str(v.value) for v in self.aabun]) if self.aabun is not None else None,
            "ATMOSPHERE-AUNIT": ",".join([str(v.unit) for v in self.aabun]) if self.aabun is not None else None,
            "ATMOSPHERE-ASIZE": ",".join([str(v.value) for v in self.asize]) if self.asize is not None else None,
            "ATMOSPHERE-ASUNI": ",".join([str(v.unit) for v in self.asize]) if self.asize is not None else None,
            "ATMOSPHERE-LAYERS-MOLECULES": ",".join(
                [str(v) for v in self.layers_molecules]
            ) if self.layers_molecules is not None else None,
            "ATMOSPHERE-LAYERS": self.layers ,
            "ATMOSPHERE-STRUCTURE": self.structure,
            "ATMOSPHERE-PRESSURE": self.pressure,
            "ATMOSPHERE-PUNIT": self.punit,
            "ATMOSPHERE-TEMPERATURE": self.temperature.to_value(u.K) if self.temperature is not None else None,
            "ATMOSPHERE-WEIGHT": self.weight,
            "ATMOSPHERE-CONTINUUM": self.continuum,
            "ATMOSPHERE-TAU": self.tau,
            "ATMOSPHERE-NMAX": self.nmax,
            "ATMOSPHERE-LMAX": self.lmax,
            "ATMOSPHERE-DESCRIPTION": self.description,
            "ATMOSPHERE-GCM-PARAMETERS": self.gcm_parameters,
        }
        if self.layers is not None:
            for i in range(1, self.layers + 1):
                config[f"ATMOSPHERE-LAYER-{i}"] = np.array2string(
                    self.layer[i - 1], separator=",", max_line_width=np.inf
                )[1:-1]

        config = {k: str(v) for k, v in config.items() if v is not None}
        return config

    @property
    def gas(self):
        #:list(str): Name of the gases to include in the simulation, e.g 'H2O, CO2'. Only these will considered for the radiative transfer
        return self._gas

    @gas.setter
    def gas(self, value):
        self._gas = value
        self.abun = [1 * scl] * len(value)
        self.type = [f"HIT[{self.hitran_molecule_id[v]}]" for v in self.gas]

    @property
    def unit(self):
        #:list(unit): Unit of the ATMOSPHERE-ABUN field, % / ppmv / ppbv / pptv / m2:'molecules/m2' / scl:'scaler of profile'
        return [a.unit for a in self.abun]

    @property
    def ngas(self):
        #:int: Number of gases to include in the simulation, maximum 20
        return len(self.gas)

    @property
    def aeros(self):
        return self._aeros

    @aeros.setter
    def aeros(self, value):
        self._aeros = value
        self.aabun = [1 * scl] * len(value)
        self.atype = [""] * len(value)
        self.asize = [1 * scl] * len(value)

    @property
    def aunit(self):
        #:list(unit): Unit of the ATMOSPHERE-AABUN field, % / ppmv / ppbv / pptv / m2:'molecules/m2' / scl:'scaler of profile'
        return [a.unit for a in self.aabun]

    @property
    def asuni(self):
        #:list(init): Unit of the size of the aerosol particles
        if self.asize is None:
            return None
        return [a.unit for a in self.asize]

    @property
    def naero(self):
        #:int: Number of aerosols to include in the simulation, maximum 20
        if self.aeros is None:
            return None
        return len(self.aeros)

    @property
    def layers(self):
        return self.layer.shape[0]


class PSG_Surface(PSG_Config):
    def __init__(self, config) -> None:
        #:str: Type of scattering model describing the surface, and the model parameters
        self.model = self.get_value(config, "SURFACE-MODEL")
        #:quantity: Temperature of the surface [K]
        self.temperature = self.get_quantity(config, "SURFACE-TEMPERATURE", u.K)
        #:float: Albedo the surface [0:non-reflectance, 1:fully-reflective]
        self.albedo = self.get_value(config, "SURFACE-ALBEDO", float)
        #:float: Emissivity of the surface [0:non-emitting, 1:perfect-emitter]
        self.emissivity = self.get_value(config, "SURFACE-EMISSIVITY", float)
        #:float: For expanding cometary coma, this value indicates an scaling value for the dust in the coma
        self.gas_ratio = self.get_value(config, "SURFACE-GAS-RATIO", float)
        #:str: Unit of the dust abundance, [ratio] is dust/gas mass ratio, while [afrho] is the Afrho value and [lafrho] is log[Afrho/Q]
        self.gas_unit = config.get("SURFACE-GAS-UNIT")
        #:int: Number of components describing the surface properties [areal mixing]
        self.nsurf = self.get_value(config, "SURFACE-NSURF", int)
        #:str: Name of surface components to be included in the simulation
        self.surf = config.get("SURFACE-SURF")
        #:str: Sub-type of the surface components
        self.type = config.get("SURFACE-TYPE")
        #:str: Relative abundance of the surface components. For the remaining abundance, average surface albedo/emissivity will be considered
        self.abun = config.get("SURFACE-ABUN")
        #:Unit: Unit of the SURFACE-ABUN field, % / ppm / ppv
        self.unit = self.get_value(config, "SURFACE-UNIT", u.Unit)
        #:str: Thickness for each surface component [um]
        self.thick = self.get_quantity(config, "SURFACE-THICK", u.um)

    def to_config(self):
        config = {
            "SURFACE-MODEL": self.model,
            "SURFACE-TEMPERATURE": self.temperature.to_value(u.K)
            if self.temperature is not None
            else None,
            "SURFACE-ALBEDO": self.albedo,
            "SURFACE-EMISSIVITY": self.emissivity,
            "SURFACE-GAS-RATIO": self.gas_ratio,
            "SURFACE-GAS-UNIT": self.gas_unit,
            "SURFACE-NSURF": self.nsurf,
            "SURFACE-SURF": self.surf,
            "SURFACE-TYPE": self.type,
            "SURFACE-ABUN": self.abun,
            "SURFACE-UNIT": self.unit,
            "SURFACE-THICK": self.thick.to_value(u.um)
            if self.thick is not None
            else None,
        }
        config = {k: str(v) for k, v in config.items() if v is not None}
        return config


class PSG_Generator(PSG_Config):
    def __init__(self, config) -> None:
        # Unit of the GENERATOR-RANGE fields, um / nm / mm / An:'Angstrom' / cm:'Wavenumber [cm-1]' / MHz / GHz / kHz
        range_unit = config.get("GENERATOR-RANGEUNIT")
        range_unit = self.parse_units(range_unit, [u.um, u.nm, u.mm, u.AA, 1 / u.cm, u.MHz, u.GHz, u.kHz],
            ["um", "nm", "An", "cm", "MHz", "GHz", "kHz"],)
        #:quantity: Lower spectral range for the simulation
        self.range1 = self.get_quantity(config, "GENERATOR-RANGE1", range_unit)
        #:quantity: Upper spectral range for the simulation
        self.range2 = self.get_quantity(config, "GENERATOR-RANGE2", range_unit)

        resolution_unit = config.get("GENERATOR-RESOLUTIONUNIT")
        resolution_unit = self.parse_units(resolution_unit, [u.one, u.um, u.nm, u.mm, u.AA, 1/u.cm, u.MHz, u.GHz, u.kHz], ["RP", "um", "nm", "mm", "An", "cm", "MHz", "GHz", "kHz"])
        #:quantity: Spectral resolution for the simulation. PSG assumes that the sampling resolution is equal is to the instrumental resolution, yet radiative transfer resolutions are always performed at the necessary/higher resolutions in order to accurately describe the lineshapes
        self.resolution = self.get_quantity(config, "GENERATOR-RESOLUTION", resolution_unit)
        #:bool: Convolution kernel applied to the spectra, default is 'N'
        self.resolution_kernel = self.get_bool(config, "GENERATOR-RESOLUTIONKERNEL")
        #:bool: Flag indicating whether to include molecular signatures as generated with PUMAS or CEM [Y/N]
        self.gas_model = self.get_bool(config, "GENERATOR-GAS-MODEL")
        #:bool: Flag indicating whether to include continuum signatures as generated by the surface, the star (when in the field) and dust/nucleus (when synthesizing comets) [Y/N]
        self.cont_model = self.get_bool(config, "GENERATOR-CONT-MODEL")
        #:bool: Flag indicating whether to include stellar absorption signatures in the reflected sunlight / stellar spectra [Y/N]
        self.cont_stellar = self.get_bool(config, "GENERATOR-CONT-STELLAR")
        #:bool: Flag indicating whether we are synthesizing planetary spectra as observed with a ground-based telescope. This flag will ensure that the noise module properly includes telluric signatures
        self.trans_show = self.get_bool(config, "GENERATOR-TRANS-SHOW")
        #:bool: Flag indicating whether to show the spectra as observed and multiplied by the telluric transmittance [Y]
        self.trans_apply = self.get_bool(config, "GENERATOR-TRANS-APPLY")
        #:str: Keyword [SS-WW] indicating the site [SS] and water abundance [WW]. Values of SS are 00:'0m (sea level)', 01:'2,600m (8,500 feet)', 02:'4,200m (14,000 feet)', 03:'14,000m (46,000 feet)', 04:'35,000m (120,000 feet)'. Values of WW are 00:'10% tropical', 01:'30% tropical', 02:'70% tropical', 03:'100% tropical'
        self.trans = config.get("GENERATOR-TRANS")
        #:str: Radiation unit for the generated spectra, see full list of permitted keywords in the 'Modeling > Radiation Units' section
        self.rad_units = config.get("GENERATOR-RADUNITS")
        #:bool: Flag indicating whether to show the spectra employing a logarithmic scale
        self.lograd = self.get_bool(config, "GENERATOR-LOGRAD")
        #:str: Type of telescope, SINGLE:'single dish telescope or instrument', ARRAY:'Interferometric array', CORONA:'Coronagraph', AOTF or LIDAR
        self.telescope = config.get("GENERATOR-TELESCOPE")

        beam_unit = self.get_value(config, "GENERATOR-BEAM-UNIT", u.Unit)
        #:quantity: Full width half-maximum (FWHM) of the instrument's beam or field-of-view (FOV)
        self.beam = self.get_quantity(config, "GENERATOR-BEAM", beam_unit)
        #:quantity: Diameter of the main reflecting surface of the telescope or instrument [m]
        self.diam_tele = self.get_quantity(config, "GENERATOR-DIAMTELE", u.m)
        #:str: For interferometers, the number of telescopes; for coronagraphs, the instrument's contrast
        self.telescope1 = config.get("GENERATOR-TELESCOPE1")
        #:str: This field indicates the zodi-level (1.0:Ecliptic pole/minimum, 2.0:HST/JWST low values, 10.0:Normal values, 100.0:Close to ecliptic/Sun), or order number for the AOTF system. For coronagraphs, this field indicates allows two entries: the exozodi level and the local zodiacal dust level
        self.telescope2 = config.get("GENERATOR-TELESCOPE2")
        #:str: For coronagraphic observations, the inner working angle (IWA) in units of [L/D]
        self.telescope3 = config.get("GENERATOR-TELESCOPE3")
        #:str: Keyword identifying the noise model to consider, NO:'None', TRX:'Receiver temperature', RMS:'Constant noise in radiation units', BKG:'Constant noise with added background', NEP:'Power equivalent noise detector model', D*:'Detectability noise detector model', CCD:'Image sensor'
        self.noise = config.get("GENERATOR-NOISE")
        #:quantity: Exposure time per frame [sec]
        self.noise_time = self.get_quantity(config, "GENERATOR-NOISETIME", u.s)
        #:int: Number of exposures
        self.noise_frames = self.get_value(config, "GENERATOR-NOISEFRAMES", int)
        #:int: Total number of pixels that encompass the beam (GENERATOR-BEAM) and the spectral unit (GENERATOR-RESOLUTION)
        self.noise_pixels = self.get_value(config, "GENERATOR-NOISEPIXELS", int)
        #:str: First noise model parameter - For RMS, 1-sigma noise; for TRX, the receiver temperature; for BKG, the 1-sigma noise; for NEP, the sensitivity in W/sqrt(Hz); for DET, the sensitivity in cm.sqrt(Hz)/W; for CCD, the read noise [e-]
        self.noise1 = config.get("GENERATOR-NOISE1")
        #:str: Second noise model parameter - For RMS, not used; for TRX, the sideband g-factor; for BKG, the not used; for NEP, not used; for DET, the pixel size [um]; for CCD, the dark rate [e-/s]
        self.noise2 = config.get("GENERATOR-NOISE2")
        #:float: Total throughput of the telescope+instrument, from photons arriving to the main mirror to photons being quantified by the detector [0:none to 1:perfect]. The user can provide wavelength dependent values as neff@wavelength[um] (e.g., '0.087@2.28,0.089@2.30,0.092@2.31,0.094@2.32,...')
        self.noise_oeff = config.get("GENERATOR-NOISEOEFF")
        #:float: Emissivity of the telescope+instrument optics [0 to 1]
        self.noise_oemis = self.get_value(config, "GENERATOR-NOISEOEMIS", float)
        #:float: Temperature of the telescope+instrument optics [K]
        self.noise_otemp = self.get_quantity(config, "GENERATOR-NOISEOTEMP", u.K)
        #:str: Text describing if an instrument template was used to define the GENERATOR parameters
        self.instrument = config.get("GENERATOR-INSTRUMENT")
        #:float: Well depth [e-] of each pixel detector
        self.noise_well = self.get_value(config, "GENERATOR-NOISEWELL", float)
        #:float: Spatial binning applied to the GCM data when computing spectra. 1: Full resolution
        self.gcm_binning = self.get_value(config, "GENERATOR-GCM-BINNING", float)

    @property
    def range_unit(self):
        return self.range1.unit

    @property
    def resolution_unit(self):
        return self.resolution.unit

    @property
    def beam_unit(self):
        return self.beam.unit

    def to_config(self):
        range_unit_loc, range_unit = self.get_units(
            self.range_unit,
            [u.um, u.nm, u.mm, u.AA, 1 / u.cm, u.MHz, u.GHz, u.kHz],
            ["um", "nm", "An", "cm", "MHz", "GHz", "kHz"],
        )
        resolution_unit_loc, resolution_unit = self.get_units(
            self.resolution_unit,
            [u.one, u.um, u.nm, u.mm, u.AA, 1 / u.cm, u.MHz, u.GHz, u.kHz],
            ["RP", "um", "nm", "mm", "An", "cm", "MHz", "GHz", "kHz"],
        )
        beam_unit_loc, beam_unit = self.get_units(
            self.beam_unit,
            [u.arcsec, u.arcmin, u.deg, u.km, diameter, diffrac],
            ["arcsec", "arcmin", "degree", "km", "diameter", "diffrac"],
        )
        config = {
            "GENERATOR-RANGE1": self.range1.to_value(range_unit_loc) if self.range1 is not None and range_unit_loc is not None else None,
            "GENERATOR-RANGE2": self.range2.to_value(range_unit_loc) if self.range2 is not None and range_unit_loc is not None else None,
            "GENERATOR-RANGEUNIT": range_unit,
            "GENERATOR-RESOLUTION": self.resolution.to_value(resolution_unit_loc) if self.resolution is not None and resolution_unit_loc is not None else None,
            "GENERATOR-RESOLUTIONUNIT": resolution_unit,
            "GENERATOR-RESOLUTIONKERNEL": "Y" if self.resolution_kernel else "N" if self.resolution_kernel is not None else None,
            "GENERATOR-GAS-MODEL": "Y" if self.gas_model else "N" if self.gas_model is not None else None,
            "GENERATOR-CONT-MODEL": "Y" if self.cont_model else "N" if self.cont_model is not None else None,
            "GENERATOR-CONT-STELLAR": "Y" if self.cont_stellar else "N" if self.cont_stellar is not None else None,
            "GENERATOR-TRANS-SHOW": "Y" if self.trans_show else "N" if self.trans_show is not None else None,
            "GENERATOR-TRANS-APPLY": "Y" if self.trans_apply else "N" if self.trans_apply is not None else None,
            "GENERATOR-TRANS": self.trans,
            "GENERATOR-RADUNITS": self.rad_units,
            "GENERATOR-LOGRAD": "Y" if self.lograd else "N" if self.lograd is not None else None,
            "GENERATOR-TELESCOPE": self.telescope,
            "GENERATOR-BEAM": self.beam.to_value(beam_unit_loc) if self.beam is not None and beam_unit_loc is not None else None,
            "GENERATOR-BEAM-UNIT": beam_unit,
            "GENERATOR-DIAMTELE": self.diam_tele.to_value(u.m) if self.diam_tele is not None else None,
            "GENERATOR-TELESCOPE1": self.telescope1,
            "GENERATOR-TELESCOPE2": self.telescope2,
            "GENERATOR-TELESCOPE3": self.telescope3,
            "GENERATOR-NOISE": self.noise,
            "GENERATOR-NOISETIME": self.noise_time.to_value(u.s) if self.noise_time is not None else None,
            "GENERATOR-NOISEFRAMES": self.noise_frames,
            "GENERATOR-NOISEPIXELS": self.noise_pixels,
            "GENERATOR-NOISE1": self.noise1,
            "GENERATOR-NOISE2": self.noise2,
            "GENERATOR-NOISEOEFF": self.noise_oeff,
            "GENERATOR-NOISEOEMIS": self.noise_oemis,
            "GENERATOR-NOISEOTEMP": self.noise_otemp.to_value(u.K) if self.noise_otemp is not None else None,
            "GENERATOR-INSTRUMENT": self.instrument,
            "GENERATOR-NOISEWELL": self.noise_well,
            "GENERATOR-GCM-BINNING": self.gcm_binning,
        }
        config = {k: str(v) for k, v in config.items() if v is not None}
        return config


class PSG_Retrieval(PSG_Config):
    def __init__(self, config) -> None:
        #:float: The parameter Gamma (or Levenberg-Marquart parameter) is the extra regularization parameter (e.g., 0:Classic LM, 1:classic Rodgers' formalism, 10:Heavily tailored to the a-priori)
        self.gamma = self.get_value(config, "RETRIEVAL-GAMMA", float)
        #:str: Parameters for the nested sampling retrieval method
        self.nest = config.get("RETRIEVAL-NEST")

        range_unit = config.get("RETRIEVAL-RANGEUNIT")
        #:Unit: Spectral unit of the user-provided data for the retrieval, um / nm / mm / An:'Angstrom' / cm:'Wavenumber [cm-1]' / MHz / GHz / kHz
        self.range_unit = self.parse_units(
            range_unit,
            [u.um, u.nm, u.mm, u.AA, 1 / u.cm, u.MHz, u.GHz, u.kHz],
            ["um", "nm", "mm", "An", "cm", "MHz", "GHz", "kHz"],
        )

        resolution_unit = config.get("RETRIEVAL-RESOLUTIONUNIT")
        resolution_unit = self.parse_units(
            resolution_unit,
            [u.one, u.um, u.nm, u.mm, u.AA, 1 / u.cm, u.MHz, u.GHz, u.kHz],
            ["RP", "um", "nm", "mm", "An", "cm", "MHz", "GHz", "kHz"],
        )
        #:quantity: Instrument's spectral resolution [FWHM] of the user-provided data. This value is independent of the sampling rate of the data, and refers to the actual spectral resolution of the instrument
        self.resolution = self.get_quantity(config, "RETRIEVAL-RESOLUTION", resolution_unit)
        #:float: Scaling value to be applied to all fluxes of the user-provided data file
        self.flux_scaler = self.get_value(config, "RETRIEVAL-FLUXSCALER", float)
        #:str: Frequency/wavelength corrections (0th, 1st and 2nd orders) to be applied to the data
        self.freq_shift = config.get("RETRIEVAL-FREQSHIFT")
        #:str: Labels for the columns of the data file
        self.flux_labels = config.get("RETRIEVAL-FLUXLABELS")
        #:int: Polynomical degree of the instrument's gain function, -1:None, 0:Constant, 1:Sloped, 2:Quadratic, etc
        self.fit_gain = self.get_value(config, "RETRIEVAL-FITGAIN", int)
        #:bool: Flag indicating whether to preserve the photometric information of the data (zeroth order of gain fitting) [Y:Disable 0th order / N:Enable]
        self.fit_gain_photometric = self.get_bool(config, "RETRIEVAL-FITGAIN-PHOTOMETRIC")
        #:int: Polynomical degree of the residual offset, -1:None, 0:Constant, 1:Sloped, 2:Quadratic, etc
        self.remove_offset = self.get_value(config, "RETRIEVAL-REMOVEOFFSET", int)
        #:int: Maximum number of spectral fringes to be removed from the data
        self.remove_fringe = self.get_value(config, "RETRIEVAL-REMOVEFRINGE", int)
        #:bool: Flag indicating whether to fit the intensity of the solar/stellar features [Y/N]
        self.fit_stellar = self.get_bool(config, "RETRIEVAL-FITSTELLAR")
        #:bool: Flag indicating whether to refine the spectral calibration [Y/N]
        self.fit_freq = self.get_bool(config, "RETRIEVAL-FITFREQ")
        #:bool: Flag indicating whether to fit the spectral resolution [Y/N]
        self.fit_resolution = self.get_bool(config, "RETRIEVAL-FITRESOLUTION")
        #:bool: Flag indicating whether to fit the telluric features [Y/N]. This is done by perturbing the selected telluric column/water abundances
        self.fit_telluric = self.get_bool(config, "RETRIEVAL-FITTELLURIC")
        #:list: Name of the variables of the retrieval (comma separated)
        self.variables = self.get_list(config, "RETRIEVAL-VARIABLES")
        #:array: A-priori and resulting values of the retrieval parameters (comma separated)
        self.values = self.get_list(config, "RETRIEVAL-VALUES", func=float, array=True)
        #:array: Resulting 1-sigma uncertainties (corrected by chi-square) of the retrieval parameters (comma separated)
        self.sigmas = self.get_list(config, "RETRIEVAL-SIGMAS", func=float, array=True)
        #:array: Lower boundary permitted for each parameter (comma separated)
        self.min = self.get_list(config, "RETRIEVAL-MIN", func=float, array=True)
        #:array: Upper boundary permitted for each parameter (comma separated)
        self.max = self.get_list(config, "RETRIEVAL-MAX", func=float, array=True)
        #:list: Magnitude unit of the a-priori and boundary entries (comma separated)
        self.units = self.get_list(config, "RETRIEVAL-UNITS")
        #:str: Flag indicating the status of the retrieval suite (e.g., RUNNING, OK)
        self.status = self.get_bool(config, "RETRIEVAL-STATUS")

    @property
    def resolution_unit(self):
        if self.resolution is None:
            return None
        return self.resolution.unit

    @property
    def nvars(self):
        if self.variables is None:
            return None
        return len(self.variables)

    def to_config(self):
        range_unit_loc, range_unit = self.get_units(
            self.range_unit,
            [u.um, u.nm, u.mm, u.AA, 1 / u.cm, u.MHz, u.GHz, u.kHz],
            ["um", "nm", "mm", "An", "cm", "MHz", "GHz", "kHz"],
        )
        resolution_unit_loc, resolution_unit = self.get_units(
            self.resolution_unit,
            [u.one, u.um, u.nm, u.mm, u.AA, 1 / u.cm, u.MHz, u.GHz, u.kHz],
            ["RP", "um", "nm", "mm", "An", "cm", "MHz", "GHz", "kHz"],
        )

        config = {
            "RETRIEVAL-GAMMA": self.gamma,
            "RETRIEVAL-NEST": self.nest,
            "RETRIEVAL-RANGEUNIT": self.range_unit,
            "RETRIEVAL-RESOLUTION": self.resolution.to_value(resolution_unit_loc) if self.resolution is not None and resolution_unit_loc is not None else None,
            "RETRIEVAL-RESOLUTIONUNIT": resolution_unit,
            "RETRIEVAL-FLUXSCALER": self.flux_scaler,
            "RETRIEVAL-FREQSHIFT": self.freq_shift,
            "RETRIEVAL-FLUXLABELS": self.flux_labels,
            "RETRIEVAL-FITGAIN": self.fit_gain,
            "RETRIEVAL-FITGAIN-PHOTOMETRIC": "Y" if self.fit_gain_photometric else "N" if self.fit_gain_photometric is not None else None,
            "RETRIEVAL-REMOVEOFFSET": self.remove_offset,
            "RETRIEVAL-REMOVEFRINGE": self.remove_fringe,
            "RETRIEVAL-FITSTELLAR": "Y" if self.fit_stellar else "N" if self.fit_stellar is not None else None,
            "RETRIEVAL-FITFREQ": "Y" if self.fit_freq else "N" if self.fit_freq is not None else None,
            "RETRIEVAL-FITRESOLUTION": "Y" if self.fit_resolution else "N" if self.fit_resolution is not None else None,
            "RETRIEVAL-FITTELLURIC": "Y" if self.fit_telluric else "N" if self.fit_telluric is not None else None,
            "RETRIEVAL-NVARS": self.nvars,
            "RETRIEVAL-VARIABLES": ",".join(self.variables) if self.variables is not None else None,
            "RETRIEVAL-VALUES": ",".join([str(v) for v in self.values]) if self.values is not None else None,
            "RETRIEVAL-SIGMAS": ",".join([str(v) for v in self.sigmas]) if self.sigmas is not None else None,
            "RETRIEVAL-MIN": ",".join([str(v) for v in self.min]) if self.min is not None else None,
            "RETRIEVAL-MAX": ",".join([str(v) for v in self.max]) if self.max is not None else None,
            "RETRIEVAL-UNITS": ",".join(self.units) if self.units is not None else None,
            "RETRIEVAL-STATUS": self.status,
        }
        config = {k: str(v) for k, v in config.items() if v is not None}
        return config


class PSG_Package:
    """ Abstract Class for a PSG package """

    name = None

    def __init__(self, server, version=None) -> None:
        self.server = server
        if version is None:
            try:
                version = server.get_package_version(self.name)
            except:
                version = ""
        self.version = version

    def __str__(self):
        return f"{self.name.upper()} - version ({self.version})"

    def update(self):
        result = self.sever.update_package(self.name)
        self.version = server.get_package_version(self.name)
        return result

    def install(self):
        result = self.server.install_package(self.name)
        self.version = server.get_package_version(self.name)
        return result

    def remove(self):
        self.version = None
        return self.server.remove_package(self.name)

    def help(self):
        """ Return a formatted docstring """
        return self.__doc__



class Programs_Package(PSG_Package):
    """
    Operational software and programs required by PSG. This package includes all the necessary binaries
    to perform the radiative transfer calculations and retrievals, together with the PHP server interpreter modules. 
    This package cannot be removed (it is fundamental for the operation of PSG), and it should be constantly
    updated to reflect the upgrades performed to the suite.
    """

    name = "programs"


class Base_Package(PSG_Package):
    """
    This package includes the basic spectroscopic data for performing line-by-line calculations across a broad range of wavelengths and domains. 
    It includes:

    - HITRAN line-by-line database formatted for PSG (binary) with collissional information for CO2, H2, He when available.
    - HITRAN Collission-Induced-Absorption (CIA) database for many species due to various collisionally interacting atoms or molecules. 
      Some CIA spectra are given over an extended range of frequencies.
    - UV cross-sections database for a multiple species (ingested automatically by the RT modules when applicable).
    - Kurucz's stellar templates.
    - Scattering models for hundreds of optical constants (HRI, GSFC, Mie scattering), and for typical Mars aerosols 
      (dust, water-ice, T-max based on Wolff et al.)
    """

    name = "base"


class Surfaces_Package(PSG_Package):
    """
    The surface spectroscopic package includes the complete repository of optical constants and reflectances 
    from all databases handled by PSG (e.g., RELAB, USGS, JENA).
    """

    name = "surfaces"


class Atmospheres_Package(PSG_Package):
    """
    This package includes the climatological and atmospheric files needed by the 'atmosphere' module. 
    Specifically, it includes:

    - Atmospheric templates for most Solar-System bodies.
    - Earth MERRA2 auxiliary data (e.g., topography), Mars-MCD, and Mars-GEM databases.
    - Exoplanet basic templates.
    - Exoplanet Parmentier T/P model and Kempton equilibrium chemistry modules.
    - Exoplanet Terrestrial LAPS model (Turbet+2015).
    """

    name = "atmospheres"


class Ephm_Package(PSG_Package):
    """
    The ephemerides package includes orbital and geometric used by the 'geometry' module. 
    Specifically, it includes:

    - Ephemerides information for hundreds of Solar-System bodies (1960-2050).
    - Ephemerides information for dozens of planetary missions (e.g., Cassini, MAVEN).
    """

    name = "ephm"


class Telluric_Package(PSG_Package):
    """
    This package includes the database of telluric transmittances necessary when computing spectra as observed with ground-based observatories. 
    It includes:

    - Database of telluric transmittances pre-computed for 5 altitudes and 4 columns of water for each case.
    - The altitudes include that of Mauna-Kea/Hawaii (4200 m), Paranal/Chile (2600 m), SOFIA (14,000 m) and balloon observatories (35,000 m).
    - The water vapor column was established by scaling the tropical water profile by a factor of 0.1, 0.3 and 0.7 and 1.
    """

    name = "telluric"


class Xcross_Package(PSG_Package):
    """
    This package contains hundreds of absorption cross-sections for 
    complex volatiles as reported by the latest HITRAN release. 
    """

    name = "xcross"


class Lines_Package(PSG_Package):
    """
    This package contains line-by-line spectroscopic information from several databases.

    - GEISA database.
    - JPL Molecular spectroscopy database.
    - CDMS molecular spectroscopy database.
    - CFA/Harvard Kurucz atomic database.
    """

    name = "lines"


class Fluor_Package(PSG_Package):
    """
    This package contains non-LTE fluorescence efficiencies for dozens of species, 
    suitable when synthesizing cometary spectra in the UV/optical/IR range. 
    """

    name = "fluor"


class Exo_Package(PSG_Package):
    """
    This package contains molecular and atomic cross-sections applicable for exoplanetary modeling, 
    and it is based on the database employed by the open source 'Exo-Transmit' code (Kempton et al. 2017). 
    """

    name = "exo"


class Mass_Package(PSG_Package):
    """
    The mass spectrometry package provides access to the MS fragmentation pattern 
    database for >20,000 species computed based on the NIST Standard Reference Database Number 69 library.
    """

    name = "mass"


class Corrklow_Package(PSG_Package):
    """
    This package contains correlated-k tables for the main HITRAN species 
    (H2O, CO2, O3, N2O, CO, CH4, O2, SO2, NO2, NH3, HCl, OCS, H2CO, N2, HCN, C2H2, C2H4, PH3, H2S, C2H4, H2), 
    and for different collissional partners (e.g., CO2, H2, He) when available. The tables were computed with PUMAS 
    assuming wings of 25 cm-1 and a fine core of 1 cm-1 where maximum resolution calculations are applied.
    This is the 'low' resolution package applicable to synthesis of spectra with a resolving power lower/equal than 500. 
    """

    name = "corrklow"


class Corrkmed_Package(PSG_Package):
    """
    This package contains correlated-k tables for the main HITRAN species 
    (H2O, CO2, O3, N2O, CO, CH4, O2, SO2, NO2, NH3, HCl, OCS, H2CO, N2, HCN, C2H2, C2H4, PH3, H2S, C2H4, H2), 
    and for different collissional partners (e.g., CO2, H2, He) when available. The tables were computed with PUMAS 
    assuming wings of 25 cm-1 and a fine core of 1 cm-1 where maximum resolution calculations are applied.
    This is the 'med' resolution package applicable to synthesis of spectra with a resolving power greater than 500 and lower/equal to 5000.
    """

    name = "corrkmed"


class PSG:
    # Assign package names to package classes
    _packages = {
        "programs": Programs_Package,
        "base": Base_Package,
        "surfaces": Surfaces_Package,
        "atmospheres": Atmospheres_Package,
        "ephm": Ephm_Package,
        "telluric": Telluric_Package,
        "xcross": Xcross_Package,
        "lines": Lines_Package,
        "fluor": Fluor_Package,
        "exo": Exo_Package,
        "mass": Mass_Package,
        "corrklow": Corrklow_Package,
        "corrkmed": Corrkmed_Package,
    }

    def __init__(self, server=None, config=None) -> None:
        # self.server = 'https://psg.gsfc.nasa.gov'
        if server is None:
            server = "http://localhost:3000"
        self.server = server

        # Read configuration from file
        if config is None:
            config_file = join(dirname(__file__), "psg_cfg.txt")
            config = self.read_config(config_file)
        else:
            try:
                config = self.read_config(config)
            except FileNotFoundError:
                config = config
        self._config = config

        # Pass config to substructures
        self.object = PSG_Object(self._config)
        self.geometry = PSG_Geometry(self._config)
        self.atmosphere = PSG_Atmosphere(self._config)
        self.surface = PSG_Surface(self._config)
        self.generator = PSG_Generator(self._config)
        self.retrieval = PSG_Retrieval(self._config)

        # Load the individual packages for object oriented interface
        versions = self.get_package_version()
        self.packages = {
            name: cls(self, versions[name.upper()])
            for name, cls in self._packages.items() if name.upper() in versions.keys()
        }

    @property
    def config(self):
        return self.to_config()

    @staticmethod
    def read_config(config_file):
        with open(config_file, "r") as f:
            lines = f.read()

        matches = re.findall(r"<(.*?)>(.*)\n", lines)
        config = {k: v for k, v in matches}
        return config

    def to_config(self):
        self._config.update(self.object.to_config())
        self._config.update(self.geometry.to_config())
        self._config.update(self.atmosphere.to_config())
        self._config.update(self.surface.to_config())
        self._config.update(self.generator.to_config())
        self._config.update(self.retrieval.to_config())
        return self._config

    def write_config(self, config_file=None):
        config = self.to_config()
        lines = [f"<{k}>{v}\n" for k, v in config.items()]
        text = "".join(lines)
        if config_file is not None:
            with open(config_file, "w") as f:
                f.write(text)
                f.flush()
        return text

    @staticmethod
    def read_datafile(datafile):
        # Read the header
        # and split into the seperate parts
        with open(datafile, "r") as f:
            columns = None
            for i, line in enumerate(f):
                if not line.startswith("#"):
                    # Skip after the header
                    break
                if line.startswith("# WARNING"):
                    print(line[2:])
                    continue
                columns = line
            if columns is not None:
                columns = columns[2:-1].split(" ")
        # Read the data
        data = np.genfromtxt(datafile, names=columns)
        return data

    @staticmethod
    def download_file(
        server, config_text, psg_type="rad", wgeo="y", wephm="n", watm="n", cache=True
    ):
        hash = hashlib.sha256((psg_type + wephm + watm + config_text).encode("utf-8"))
        url = join(server, f"{hash.hexdigest()}.txt")

        if not is_url_in_cache(url, pkgname=PKGNAME) or not cache:
            with NamedTemporaryFile("w") as cf:
                cf.write(config_text)
                cf.flush()

                result = subprocess.run(
                    f"curl -s -d type={psg_type} -d wgeo={wgeo} -d wephm={wephm} -d watm={watm} --data-urlencode file@{cf.name} {server}/api.php",
                    capture_output=True,
                    shell=True,
                )
            text = result.stdout.decode()
            if text == "":
                raise RuntimeError("The PSG server did not return a result")
            with NamedTemporaryFile("w") as output:
                output.write(text)
                output.flush()
                import_file_to_cache(url, output.name, pkgname=PKGNAME)

        # Return the filename in the cache
        result = download_file(url, cache=True, pkgname=PKGNAME)
        return result

    @staticmethod
    def clear_cache():
        clear_download_cache(pkgname=PKGNAME)

    @staticmethod
    def run_curl_command(server, page="index.php", command=None):
        call = f"curl {server}/index.php"
        if command is not None:
            call = f"{call}?{command}"

        result = subprocess.run(call, capture_output=True, shell=True,)
        text = result.stdout.decode()
        return text

    def get_info(self):
        return self.run_curl_command(self.server)

    def get_installed_packages(self):
        text = self.get_info()
        lines = text.splitlines()
        packages = [l.split("-", 1)[0].strip().lower() for l in lines]
        return packages

    def get_package_version(self, package=None):
        text = self.get_info()
        if package is not None:
            match = re.match(
                package.upper() + r" - .*version \((\d{4}-\d{2}-\d{2})\)", text
            )
            version = match.group(1)
            return version
        else:
            match = re.findall(r"(\w*) - .*version \((\d{4}-\d{2}-\d{2})\)", text)
            version = {m[0]: m[1] for m in match}
            return version

    def install_package(self, package):
        text = self.run_curl_command(self.server, command=f"install={package}")
        # TODO: Check that the result is successful
        return text

    def update_package(self, package):
        text = self.run_curl_command(self.server, command=f"update={package}")
        # TODO: Check that the result is successful
        return text

    def remove_package(self, package):
        text = self.run_curl_command(self.server, command=f"remove={package}")
        # TODO: Check that the result is successful
        return text

    def update_all_packages(self):
        text = self.run_curl_command(self.server)
        lines = text.splitlines()
        lines = [l.split("-", 1) for l in lines]
        packages = [l[0].strip().lower() for l in lines if "Update available" in l[1]]
        for package in packages:
            self.update_package(package)
        return None

    def request(self, psg_type="rad", wgeo="y", wephm="n", watm="n"):
        # Create the configuration for the PSG
        config_text = self.write_config()
        # Get the filename from the cache
        output_name = self.download_file(
            self.server,
            config_text,
            psg_type=psg_type,
            wgeo=wgeo,
            wephm=wephm,
            watm=watm,
        )
        # Read the results from file
        data = self.read_datafile(output_name)
        return data
