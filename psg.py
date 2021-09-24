# ---------------------------------------------------
# Planet Spectrum Generator Interface
# PSG multiple-scattering model: https://psg.gsfc.nasa.gov/helpmodel.php
# PSG databases: https://psg.gsfc.nasa.gov/helpatm.php
# PSG API driver: https://psg.gsfc.nasa.gov/helpapi.php
# ---------------------------------------------------
from io import StringIO
from os.path import dirname, join
import subprocess

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
from astropy import units

# Add Atmosphere scale units
ppb = units.def_unit(
    ["ppb", "ppbv"], 1e-9 * units.one, namespace=globals(), doc="Parts Per Billion"
)
ppm = units.def_unit(
    ["ppm", "ppmv"], 1e3 * ppb, namespace=globals(), doc="Parts Per Million Volume"
)
ppt = units.def_unit(
    ["ppt", "pptv"], 1e6 * ppb, namespace=globals(), doc="Parts Per Thousand Volume"
)
m2 = units.def_unit(
    ["m2", "m-2"], None, namespace=globals(), doc="Molecules per square meter"
)
scl = units.def_unit(["scl"], None, namespace=globals(), doc="Relative Scale")
units.add_enabled_units([ppb, ppm, ppt, m2, scl])

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
            [units.Unit(v) for v in config["ATMOSPHERE-UNIT"].split(",")]
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
            [units.Unit(v) for v in config["ATMOSPHERE-AUNIT"].split(",")]
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
            [units.Unit(v) for v in config["ATMOSPHERE-ASUNI"].split(",")]
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

        # Store all the other atmosphere parameters
        self.other = {}
        for key, value in config.items():
            match = re.match(r"ATMOSPHERE-(.*?)(-\d+)?$", key)
            if match is not None and match[1].lower().replace("-", "_") not in dir(
                self
            ):
                self.other[key] = value

    def to_config(self):
        config = {
            "ATMOSPHERE-NGAS": str(self.ngas),
            "ATMOSPHERE-GAS": ",".join([str(v) for v in self.gas]),
            "ATMOSPHERE-TYPE": ",".join([str(v) for v in self.type]),
            "ATMOSPHERE-ABUN": ",".join([str(v.value) for v in self.abun]),
            "ATMOSPHERE-UNIT": ",".join([str(v.unit) for v in self.abun]),
            "ATMOSPHERE-NAERO": str(self.naero),
            "ATMOSPHERE-AEROS": ",".join([str(v) for v in self.aeros]),
            "ATMOSPHERE-ATYPE": ",".join([str(v) for v in self.atype]),
            "ATMOSPHERE-AABUN": ",".join([str(v.value) for v in self.aabun]),
            "ATMOSPHERE-AUNIT": ",".join([str(v.unit) for v in self.aabun]),
            "ATMOSPHERE-ASIZE": ",".join([str(v.value) for v in self.asize]),
            "ATMOSPHERE-ASUNI": ",".join([str(v.unit) for v in self.asize]),
            "ATMOSPHERE-LAYERS-MOLECULES": ",".join(
                [str(v) for v in self.layers_molecules]
            ),
            "ATMOSPHERE-LAYERS": str(self.layers),
            "ATMOSPHERE-STRUCTURE": str(self.structure),
            "ATMOSPHERE-PRESSURE": str(self.pressure),
            "ATMOSPHERE-PUNIT": str(self.punit),
            "ATMOSPHERE-TEMPERATURE": str(self.temperature),
            "ATMOSPHERE-WEIGHT": str(self.weight),
            "ATMOSPHERE-CONTINUUM": str(self.continuum),
            "ATMOSPHERE-TAU": str(self.tau),
            "ATMOSPHERE-NMAX": str(self.nmax),
            "ATMOSPHERE-LMAX": str(self.lmax),
            "ATMOSPHERE-DESCRIPTION": str(self.description),
            "ATMOSPHERE-GCM-PARAMETERS": str(self.gcm_parameters),
        }
        for i in range(1, self.layers + 1):
            config[f"ATMOSPHERE-LAYER-{i}"] = np.array2string(
                self.layer[i - 1], separator=",", max_line_width=np.inf
            )[1:-1]

        other = self.other.copy()
        other.update(config)
        return other

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
        self.config = config

        # Pass config to substructures
        self.atmosphere = PSG_Atmosphere(self.config)

        # Load the individual packages for object oriented interface
        versions = self.get_package_version()
        self.packages = {
            name: cls(self, versions.get(name, ""))
            for name, cls in self._packages.items()
        }

    @staticmethod
    def read_config(config_file):
        with open(config_file, "r") as f:
            lines = f.read()

        matches = re.findall(r"<(.*?)>(.*)\n", lines)
        config = {k: v for k, v in matches}
        return config

    def to_config(self):
        self.config.update(self.atmosphere.to_config())
        return self.config

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
