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
    ["ppm", "ppmv"], 1e3 * ppb, namespace=globals(), doc="Parts Per Million Volume"
)
ppt = u.def_unit(
    ["ppt", "pptv"], 1e6 * ppb, namespace=globals(), doc="Parts Per Thousand Volume"
)
m2 = u.def_unit(
    ["m2", "m-2"], None, namespace=globals(), doc="Molecules per square meter"
)
scl = u.def_unit(["scl"], None, namespace=globals(), doc="Relative Scale")
u.add_enabled_units([ppb, ppm, ppt, m2, scl])

# Package Name for the Astropy cache
PKGNAME = "planet-spectrum-generator"


class PSG_Config:
    def __init__(self, config) -> None:
        # Store all the other atmosphere parameters
        self.other = {}
        for key, value in config.items():
            self.other[key] = value

    def to_config(self):
        return self.other

class PSG_Object(PSG_Config):
    def __init__(self, config) -> None:
        #:str: Object type (e.g., Exoplanet, Planet, Asteroid, Moon, Comet, Object)
        self.object = config["OBJECT"]
        #:str: Object name
        self.name = config["OBJECT-NAME"]
        # Datetime
        match = re.match(r"(\d{4})/(\d{2})/(\d{2}) (\d{2}):(\d{2})", config["OBJECT-DATE"])
        #:datetime: Date of the observation (yyyy/mm/dd hh:mm) in Universal time [UT]
        self.date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5)))
        #:Quantity: Diameter of the object [km]
        self.diameter = float(config["OBJECT-DIAMETER"]) * u.km
        # Gravity
        gravity_unit = config["OBJECT-GRAVITY-UNIT"]
        if gravity_unit == "g": # Surface Gravity
            gravity_unit = u.m/u.s**2
        elif gravity_unit == "rho": # Mean Density
            gravity_unit = u.g/u.cm**3
        elif gravity_unit == "kg": # Total Mass
            gravity_unit = u.kg
        #:Quantity: Gravity/density/mass of the object
        self.gravity = float(config["OBJECT-GRAVITY"]) * gravity_unit
        #:Quantity: Distance of the planet to the Sun [AU], and for exoplanets the semi-major axis [AU]
        self.star_distance = float(config["OBJECT-STAR-DISTANCE"]) * u.AU
        #:Quantity: Velocity of the planet to the Sun [km/s], and for exoplanets the RV amplitude [km/s]
        self.star_velocity = float(config["OBJECT-STAR-VELOCITY"]) * (u.km / u.s)
        #:Quantity: Sub-solar east longitude [degrees]
        self.solar_longitude = float(config["OBJECT-SOLAR-LONGITUDE"]) * u.deg
        #:Quantity: Sub-solar latitude [degrees]
        self.solar_latitude = float(config["OBJECT-SOLAR-LATITUDE"]) * u.deg
        #:Quantity: Angular parameter (season/phase) that defines the position of the planet moving along its Keplerian orbit. For exoplanets, 0:Secondary transit, 180:Primary transit, 90/270:Opposition. For solar-system bodies, 0:'N spring equinox', 90:'N summer solstice', 180:'N autumn equinox', 270:'N winter solstice' [degrees]
        self.season = float(config["OBJECT-SEASON"]) * u.deg
        #:Quantity: Orbital inclination [degree], mainly relevant for exoplanets. Zero is phase on, 90 is a transiting orbit
        self.inclination = float(config["OBJECT-INCLINATION"]) * u.deg
        #:float: Orbital eccentricity, mainly relevant for exoplanets
        self.eccentricity = float(config["OBJECT-ECCENTRICITY"])
        #:Quantity: Orbital longitude of periapse [degrees]. It indicates the phase at which the planet reaches periapsis
        self.periapsis = float(config["OBJECT-PERIAPSIS"]) * u.deg
        #:str: Stellar type of the parent star [O/B/A/F/G/K/M]
        self.star_type = config["OBJECT-STAR-TYPE"]
        #:Quantity:  Temperature of the parent star [K]
        self.star_temperature = float(config["OBJECT-STAR-TEMPERATURE"]) * u.K
        #:Quantity: Radius of the parent star [Rsun]
        self.star_radius = float(config["OBJECT-STAR-RADIUS"]) * u.Rsun
        #:float: Metallicity of the parent star and object with respect to the Sun in log [dex]
        self.star_metallicity = float(config["OBJECT-STAR-METALLICITY"])
        #:Quantity: Sub-observer east longitude [degrees]
        self.obs_longitude = float(config["OBJECT-OBS-LONGITUDE"]) * u.deg
        #:Quantity: Sub-observer latitude, for exoplanets inclination [degrees]
        self.obs_latitude = float(config["OBJECT-OBS-LATITUDE"]) * u.deg
        #:Quantity: Relative velocity between the observer and the object [km/s]
        self.obs_velocity = float(config["OBJECT-OBS-VELOCITY"]) * (u.km/u.s)
        #:Quantity: This field is computed by the geometry module - It is the apparent rotational period of the object as seen from the observer [days]
        self.period = float(config["OBJECT-PERIOD"]) * u.day
        #:str: This field reports the orbital parameters for small bodies. It is only relevant for display purposes of the orbital diagram.
        self.orbit = config.get("OBJECT-ORBIT", "")

    @property
    def gravity_unit(self):
        """ Unit for the OBJECT-GRAVITY field, g:'Surface gravity [m/s2]', rho:'Mean density [g/cm3]', or kg:'Total mass [kg]' """
        return self.gravity.unit

    def to_config(self):
        try:
            gravity = self.gravity.to_value(u.m/u.s**2)
            gravity_unit = "g"
        except u.core.UnitConversionError:
            try:
                gravity = self.gravity.to_value(u.g/u.cm**3)
                gravity_unit = "rho"
            except u.core.UnitConversionError:
                try:
                    gravity = self.gravity.to_value(u.kg)
                    gravity_unit = "kg"
                except u.core.UnitConversionError:
                    raise ValueError("Can not convert the gravity units to PSG units")

        config = {
            "OBJECT" : self.object,
            "OBJECT-NAME": self.name,
            "OBJECT-DATE": f"{self.date.year:04}/{self.date.month:02}/{self.date.day:02} {self.date.hour:02}:{self.date.minute:02}",
            "OBJECT-DIAMETER": self.diameter.to_value(u.km),
            "OBJECT-GRAVITY": gravity,
            "OBJECT-GRAVITY-UNIT": gravity_unit,
            "OBJECT-STAR-DISTANCE": self.star_distance.to_value(u.AU),
            "OBJECT-STAR-VELOCITY": self.star_velocity.to_value(u.km/u.s),
            "OBJECT-SOLAR-LONGITUDE": self.solar_longitude.to_value(u.deg),
            "OBJECT-SOLAR-LATITUDE": self.solar_latitude.to_value(u.deg),
            "OBJECT-SEASON": self.season.to_value(u.deg),
            "OBJECT-INCLINATION": self.inclination.to_value(u.deg),
            "OBJECT-ECCENTRICITY": self.eccentricity,
            "OBJECT-PERIAPSIS": self.periapsis.to_value(u.deg),
            "OBJECT-STAR-TYPE": self.star_type,
            "OBJECT-STAR-TEMPERATURE": self.star_temperature.to_value(u.K),
            "OBJECT-STAR-RADIUS": self.star_radius.to_value(u.Rsun),
            "OBJECT-STAR-METALLICITY": self.star_metallicity,
            "OBJECT-OBS-LONGITUDE": self.obs_longitude.to_value(u.deg),
            "OBJECT-OBS-LATITUDE": self.obs_latitude.to_value(u.deg),
            "OBJECT-OBS-VELOCITY": self.obs_velocity.to_value(u.km/u.s),
            "OBJECT-PERIOD": self.period.to_value(u.day),
            "OBJECT-ORBIT": self.orbit
        }
        config = {k:str(v) for k, v in config.items()}
        return config

class PSG_Geometry(PSG_Config):
    def __init__(self, config) -> None:
        #:str: Type of observing geometry
        self.geometry = config["GEOMETRY"]
        #:str: Reference geometry (e.g., ExoMars, Maunakea), default is user defined or 'User'
        self.ref = config["GEOMETRY-REF"]

        offset_unit = config["GEOMETRY-OFFSET-UNIT"]
        if offset_unit == "arcsec": 
            offset_unit = u.arcsec
        elif offset_unit == "arcmin":
            offset_unit = u.arcmin
        elif offset_unit == "degree":
            offset_unit = u.deg
        elif offset_unit == "km":
            offset_unit = u.km
        elif offset_unit == "diameter":
            offset_unit = u.one
        else:
            raise ValueError("GEOMETRY-OFFSET-UNIT not recognized")
        #:quantity: Vertical offset with respect to the sub-observer location
        self.offset_ns = float(config["GEOMETRY-OFFSET-NS"]) * offset_unit
        #:quantity: Horizontal offset with respect to the sub-observer location
        self.offset_ew = float(config["GEOMETRY-OFFSET-EW"]) * offset_unit
        
        altitude_unit = config["GEOMETRY-ALTITUDE-UNIT"]
        if altitude_unit == "AU": 
            altitude_unit = u.AU
        elif altitude_unit == "km":
            altitude_unit = u.km
        elif altitude_unit == "diameter":
            altitude_unit = u.one
        elif altitude_unit == "pc":
            altitude_unit = u.pc
        else:
            raise ValueError("GEOMETRY-ALTITUDE-UNIT not recognized")
        #:quantity: Distance between the observer and the surface of the planet
        self.obs_altitude = float(config["GEOMETRY-OBS-ALTITUDE"]) * altitude_unit
        #:quantity: The azimuth angle between the observational projected vector and the solar vector on the reference plane
        self.azimuth = float(config["GEOMETRY-AZIMUTH"]) * u.deg
        #:float: Parameter for the selected geometry, for Nadir / Lookingup this field indicates the zenith angle [degrees], for limb / occultations this field indicates the atmospheric height [km] being sampled
        self.user_param = float(config["GEOMETRY-USER-PARAM"])
        #:str: For stellar occultations, this field indicates the type of the occultation star [O/B/A/F/G/K/M]
        self.stellar_type = config["GEOMETRY-STELLAR-TYPE"]
        #:quantity: For stellar occultations, this field indicates the temperature [K] of the occultation star
        self.stellar_temperature = float(config["GEOMETRY-STELLAR-TEMPERATURE"]) * u.K
        #:quantity: For stellar occultations, this field indicates the brightness [magnitude] of the occultation star
        self.stellar_magnitude = float(config["GEOMETRY-STELLAR-MAGNITUDE"]) * u.mag
        #:str: This field is computed by the geometry module - It indicates the angle between the observer and the planetary surface
        self.obs_angle = config["GEOMETRY-OBS-ANGLE"]
        #:str: This field is computed by the geometry module - It indicates the angle between the Sun and the planetary surface
        self.solar_angle = config["GEOMETRY-SOLAR-ANGLE"]
        #:int: This field allows to divide the observable disk in finite rings so radiative-transfer calculations are performed with higher accuracy
        self.disk_angles = int(config["GEOMETRY-DISK-ANGLES"])
        #:quantity: This field is computed by the geometry module - It indicates the phase between the Sun and observer
        self.phase = float(config["GEOMETRY-PHASE"]) * u.deg
        #:str: This field is computed by the geometry module - It indicates how much the beam fills the planetary area (1:maximum)
        self.planet_fraction = config["GEOMETRY-PLANET-FRACTION"]
        #:float: This field is computed by the geometry module - It indicates how much the beam fills the parent star (1:maximum)
        self.star_fraction = float(config["GEOMETRY-STAR-FRACTION"])
        #:float: This field is computed by the geometry module - It indicates the projected distance between the beam and the parent star in arcsceconds
        self.star_distance = float(config["GEOMETRY-STAR-DISTANCE"])
        #:str: This field is computed by the geometry module - It indicates the rotational Doppler shift [km/s] affecting the spectra and the spread of rotational velocities [km/s] within the FOV
        self.rotation = config["GEOMETRY-ROTATION"]
        #:str: This field is computed by the geometry module - It indicates the scaling factor between the integrated reflectance for the FOV with respect to the BRDF as computed using the geometry indidence/emission angles
        self.brdfscaler = config["GEOMETRY-BRDFSCALER"]

    @property
    def offset_unit(self):
        """ Unit of the GEOMETRY-OFFSET field, arcsec / arcmin / degree / km / diameter """
        return self.offset_ns.unit

    @property
    def altitude_unit(self):
        """ Unit of the GEOMETRY-OBS-ALTITUDE field, AU / km / diameter and pc:'parsec' """
        return self.obs_altitude.unit

    def to_config(self):
        if self.offset_unit == u.arcsec:
            offset_unit = "arcsec"
            loc_offset_unit = u.arcsec
        elif self.offset_unit == u.arcmin:
            offset_unit = "arcmin"
            loc_offset_unit = u.arcmin
        elif self.offset_unit == u.deg:
            offset_unit = "degree"
            loc_offset_unit = u.deg
        else:
            try:
                self.offset_unit.to(u.arcsec)
                offset_unit = "arcsec"
                loc_offset_unit = u.arcsec
            except u.core.UnitConversionError:
                try:
                    self.offset_unit.to(u.km)
                    offset_unit = "km"
                    loc_offset_unit = u.km
                except u.core.UnitConversionError:
                    try:
                        self.offset_unit.to(u.one)
                        offset_unit = "diameter"
                        loc_offset_unit = u.one
                    except u.core.UnitConversionError:
                        raise ValueError("Could not determine the offset unit")

        if self.altitude_unit == u.AU:
            altitude_unit = "AU"
            loc_altitude_unit = u.AU
        elif self.altitude_unit == u.km:
            altitude_unit = "km"
            loc_altitude_unit = u.km
        elif self.altitude_unit == u.pc:
            altitude_unit = "pc"
            loc_altitude_unit = u.pc
        else:
            try:
                self.altitude_unit.to(u.km)
                altitude_unit = "km"
                loc_altitude_unit = u.km
            except u.core.UnitConversionError:
                try:
                    self.altitude_unit.to(u.one)
                    altitude_unit = "diameter"
                    loc_altitude_unit = u.one
                except u.core.UnitConversionError:
                    raise ValueError("Could not recognize altitude units")

        config = {
            "GEOMETRY": self.geometry,
            "GEOMETRY-REF": self.ref,
            "GEOMETRY-OFFSET-NS" : self.offset_ns.to_value(loc_offset_unit),
            "GEOMETRY-OFFSET-EW" : self.offset_ew.to_value(loc_offset_unit),
            "GEOMETRY-OFFSET-UNIT": offset_unit,
            "GEOMETRY-OBS-ALTITUDE": self.obs_altitude.to_value(loc_altitude_unit),
            "GEOMETRY-ALTITUDE-UNIT": altitude_unit,
            "GEOMETRY-AZIMUTH": self.azimuth.to_value(u.deg),
            "GEOMETRY-USER-PARAM": self.user_param,
            "GEOMETRY-STELLAR-TYPE": self.stellar_type,
            "GEOMETRY-STELLAR-TEMPERATURE": self.stellar_temperature.to_value(u.K),
            "GEOMETRY-STELLAR-MAGNITUDE": self.stellar_magnitude.to_value(u.mag),
            "GEOMETRY-OBS-ANGLE": self.obs_angle,
            "GEOMETRY-SOLAR-ANGLE": self.solar_angle,
            "GEOMETRY-DISK-ANGLES": self.disk_angles,
            "GEOMETRY-PHASE": self.phase.to_value(u.deg),
            "GEOMETRY-PLANET-FRACTION": self.planet_fraction,
            "GEOMETRY-STAR-FRACTION": self.star_fraction,
            "GEOMETRY-STAR-DISTANCE": self.star_distance,
            "GEOMETRY-ROTATION": self.rotation,
            "GEOMETRY-BRDFSCALER": self.brdfscaler
        }
        config = {k: str(v) for k, v in config.items()}
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
        self._gas = (
            config["ATMOSPHERE-GAS"].split(",")
            if "ATMOSPHERE-GAS" in config.keys()
            else []
        )
        #:list(str): Sub-type of the gases, e.g. 'HIT[1], HIT[2]'
        self.type = (
            config["ATMOSPHERE-TYPE"].split(",")
            if "ATMOSPHERE-TYPE" in config.keys()
            else []
        )
        abun = (
            [float(v) for v in config["ATMOSPHERE-ABUN"].split(",")]
            if "ATMOSPHERE-ABUN" in config.keys()
            else []
        )
        unit = (
            [u.Unit(v) for v in config["ATMOSPHERE-UNIT"].split(",")]
            if "ATMOSPHERE-UNIT" in config.keys()
            else []
        )
        #:list(quantity): Abundance of gases. The values can be assumed to be same across all altitudes/layers [%,ppmv,ppbv,pptv,m-2], or as a multiplier [scl] to the provided vertical profile
        self.abun = [a * u for a, u in zip(abun, unit)]

        # Handle the Aerosols
        self._aeros = (
            config["ATMOSPHERE-AEROS"].split(",")
            if "ATMOSPHERE-AEROS" in config.keys()
            else []
        )
        #:list(str): Sub-type of the aerosols
        self.atype = (
            config["ATMOSPHERE-ATYPE"].split(",")
            if "ATMOSPHERE-ATYPE" in config.keys()
            else []
        )
        abun = (
            [float(v) for v in config["ATMOSPHERE-AABUN"].split(",")]
            if "ATMOSPHERE-AABUN" in config.keys()
            else []
        )
        unit = (
            [u.Unit(v) for v in config["ATMOSPHERE-AUNIT"].split(",")]
            if "ATMOSPHERE-AUNIT" in config.keys()
            else []
        )
        #:list(quantity): Abundance of aerosols. The values can be assumed to be same across all altitudes/layers [%,ppm,ppb,ppt,Kg/m2], or as a multiplier [scaler] to the provided vertical profile
        self.aabun = [a * u for a, u in zip(abun, unit)]
        size = (
            [float(v) for v in config["ATMOSPHERE-ASIZE"].split(",")]
            if "ATMOSPHERE-ASIZE" in config.keys()
            else []
        )
        unit = (
            [u.Unit(v) for v in config["ATMOSPHERE-ASUNI"].split(",")]
            if "ATMOSPHERE-ASUNI" in config.keys()
            else []
        )
        #:list(quantity): Effective radius of the aerosol particles. The values can be assumed to be same across all layers [um, m, log(um)], or as a multiplier [scaler] to the provided size vertical profile
        self.asize = [a * u for a, u in zip(size, unit)]

        # Handle the atmosphere layers
        #:list(str): Molecules quantified by the vertical profile
        self.layers_molecules = (
            config["ATMOSPHERE-LAYERS-MOLECULES"].split(",")
            if "ATMOSPHERE-LAYERS-MOLECULES" in config.keys()
            else []
        )
        nlayers = int(config.get("ATMOSPHERE-LAYERS", 0))
        layers = [config[f"ATMOSPHERE-LAYER-{i}"] for i in range(1, nlayers + 1)]
        layers = StringIO("\n".join(layers))
        #:array: Values for that specific layer: Pressure[bar], Temperature[K], gases[mol/mol], aerosols [kg/kg] - Optional fields: Altitude[km], X_size[m, aerosol X particle size]
        self.layer = np.genfromtxt(layers, delimiter=",")
        #:str: Parameters defining the 3D General Circulation Model grid: num_lons, num_lats, num_alts, lon0, lat0, delta_lon, delta_lat, variables (csv)
        self.gcm_parameters = config.get("ATMOSPHERE-GCM-PARAMETERS", "")
        #:str: The structure of the atmosphere, None / Equilibrium:'Hydrostatic equilibrium' / Coma:'Cometary expanding coma'
        self.structure = config["ATMOSPHERE-STRUCTURE"]
        #:float: For equilibrium atmospheres, this field defines the surface pressure; while for cometary coma, this field indicates the gas production rate
        self.pressure = float(config["ATMOSPHERE-PRESSURE"])
        #:str: The unit of the ATMOSPHERE-PRESSURE field, Pa:Pascal / bar / kbar / mbar / ubar / at / atm / torr / psi / gas:'molecules / second' / gasau:'molecules / second at rh=1AU'
        self.punit = config["ATMOSPHERE-PUNIT"]
        #:float: For atmospheres without a defined P/T profile, this field indicates the temperature across all altitudes
        self.temperature = float(config.get("ATMOSPHERE-TEMPERATURE", 0))
        #:float: Molecular weight of the atmosphere [g/mol] or expansion velocity [m/s] for expanding atmospheres
        self.weight = float(config["ATMOSPHERE-WEIGHT"])
        #:str: Continuum processes to be included in the calculation
        self.continuum = config.get("ATMOSPHERE-CONTINUUM", "")
        #:str: For expanding cometary coma, this field indicates the photodissociation lifetime of the molecules [s]
        self.tau = config.get("ATMOSPHERE-TAU", "")
        #:int: When performing scattering aerosols calculations, this parameter indicates the number of n-stream pairs - Use 0 for extinction calculations only (e.g. transit, occultation)
        self.nmax = int(config.get("ATMOSPHERE-NMAX", 0))
        #:int: When performing scattering aerosols calculations, this parameter indicates the number of scattering Legendre polynomials used for describing the phase function - Use 0 for extinction calculations only (e.g. transit, occultation)
        self.lmax = int(config.get("ATMOSPHERE-LMAX", 0))
        #:str: Description establishing the source/reference for the vertical profile
        self.description = config["ATMOSPHERE-DESCRIPTION"]

    def to_config(self):
        config = {
            "ATMOSPHERE-NGAS": self.ngas,
            "ATMOSPHERE-GAS": ",".join([str(v) for v in self.gas]),
            "ATMOSPHERE-TYPE": ",".join([str(v) for v in self.type]),
            "ATMOSPHERE-ABUN": ",".join([str(v.value) for v in self.abun]),
            "ATMOSPHERE-UNIT": ",".join([str(v.unit) for v in self.abun]),
            "ATMOSPHERE-NAERO": self.naero,
            "ATMOSPHERE-AEROS": ",".join([str(v) for v in self.aeros]),
            "ATMOSPHERE-ATYPE": ",".join([str(v) for v in self.atype]),
            "ATMOSPHERE-AABUN": ",".join([str(v.value) for v in self.aabun]),
            "ATMOSPHERE-AUNIT": ",".join([str(v.unit) for v in self.aabun]),
            "ATMOSPHERE-ASIZE": ",".join([str(v.value) for v in self.asize]),
            "ATMOSPHERE-ASUNI": ",".join([str(v.unit) for v in self.asize]),
            "ATMOSPHERE-LAYERS-MOLECULES": ",".join(
                [str(v) for v in self.layers_molecules]
            ),
            "ATMOSPHERE-LAYERS": self.layers,
            "ATMOSPHERE-STRUCTURE": self.structure,
            "ATMOSPHERE-PRESSURE": self.pressure,
            "ATMOSPHERE-PUNIT": self.punit,
            "ATMOSPHERE-TEMPERATURE": self.temperature,
            "ATMOSPHERE-WEIGHT": self.weight,
            "ATMOSPHERE-CONTINUUM": self.continuum,
            "ATMOSPHERE-TAU": self.tau,
            "ATMOSPHERE-NMAX": self.nmax,
            "ATMOSPHERE-LMAX": self.lmax,
            "ATMOSPHERE-DESCRIPTION": self.description,
            "ATMOSPHERE-GCM-PARAMETERS": self.gcm_parameters,
        }
        for i in range(1, self.layers + 1):
            config[f"ATMOSPHERE-LAYER-{i}"] = np.array2string(
                self.layer[i - 1], separator=",", max_line_width=np.inf
            )[1:-1]

        config = {k:str(v) for k,v in config.items()}
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
        return [a.unit for a in self.asize]

    @property
    def naero(self):
        #:int: Number of aerosols to include in the simulation, maximum 20
        return len(self.aeros)

    @property
    def layers(self):
        return self.layer.shape[0]


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

        # Load the individual packages for object oriented interface
        versions = self.get_package_version()
        self.packages = {
            name: cls(self, versions.get(name, ""))
            for name, cls in self._packages.items()
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
