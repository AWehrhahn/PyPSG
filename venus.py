# ---------------------------------------------------
# Sensivity study of Venus with CRIRES+
# Villanueva, NASA-GSFC, June/2021
# PSG multiple-scattering model: https://psg.gsfc.nasa.gov/helpmodel.php
# PSG databases: https://psg.gsfc.nasa.gov/helpatm.php
# PSG API driver: https://psg.gsfc.nasa.gov/helpapi.php
# ---------------------------------------------------
from io import StringIO
import os
from os.path import dirname, join
from urllib.parse import urlparse
from tqdm import tqdm
import subprocess

import os
import shutil
import re
import hashlib
from tempfile import NamedTemporaryFile
import numpy as np
import matplotlib.pyplot as plt

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
ppm = units.def_unit(["ppm", "ppmv"], 1e3 * ppb, namespace=globals(), doc="Parts Per Million Volume")
ppt = units.def_unit(["ppt", "pptv"], 1e6 * ppb, namespace=globals(), doc="Parts Per Thousand Volume")
m2 = units.def_unit(["m2", "m-2"], None, namespace=globals(), doc="Molecules per square meter")
scl = units.def_unit(["scl"], None, namespace=globals(), doc="Relative Scale")
units.add_enabled_units([ppb, ppm, ppt, m2, scl])

# Package Name for the Astropy cache
PKGNAME = "crires-venus"

class PSG_Base:
    def __init__(self, config) -> None:
        # Store all the other atmosphere parameters
        self.other = {}
        for key, value in config.items():
            self.other[key] = value

    def to_config(self):
        return self.other

class PSG_Atmosphere(PSG_Base):
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
        self._gas = config["ATMOSPHERE-GAS"].split(",")
        #:list(str): Sub-type of the gases, e.g. 'HIT[1], HIT[2]'
        self.type = config["ATMOSPHERE-TYPE"].split(",")
        abun = [float(v) for v in config["ATMOSPHERE-ABUN"].split(",")]
        unit = [units.Unit(v) for v in config["ATMOSPHERE-UNIT"].split(",")]
        #:list(quantity): Abundance of gases. The values can be assumed to be same across all altitudes/layers [%,ppmv,ppbv,pptv,m-2], or as a multiplier [scl] to the provided vertical profile
        self.abun = [a * u for a, u in zip(abun, unit)]

        # Handle the Aerosols
        self._aeros = config["ATMOSPHERE-AEROS"].split(",")
        #:list(str): Sub-type of the aerosols
        self.atype = config["ATMOSPHERE-ATYPE"].split(",")
        abun = [float(v) for v in config["ATMOSPHERE-AABUN"].split(",")]
        unit = [units.Unit(v) for v in config["ATMOSPHERE-AUNIT"].split(",")]
        #:list(quantity): Abundance of aerosols. The values can be assumed to be same across all altitudes/layers [%,ppm,ppb,ppt,Kg/m2], or as a multiplier [scaler] to the provided vertical profile
        self.aabun = [a * u for a, u in zip(abun, unit)]
        size = [float(v) for v in config["ATMOSPHERE-ASIZE"].split(",")]
        unit = [units.Unit(v) for v in config["ATMOSPHERE-ASUNI"].split(",")]
        #:list(quantity): Effective radius of the aerosol particles. The values can be assumed to be same across all layers [um, m, log(um)], or as a multiplier [scaler] to the provided size vertical profile
        self.asize = [a * u for a, u in zip(size, unit)]

        # Handle the atmosphere layers
        #:list(str): Molecules quantified by the vertical profile
        self.layers_molecules = config["ATMOSPHERE-LAYERS-MOLECULES"].split(",")
        nlayers = int(config["ATMOSPHERE-LAYERS"])
        layers = [config[f"ATMOSPHERE-LAYER-{i}"] for i in range(1, nlayers+1)]
        layers = StringIO("\n".join(layers))
        #:array: Values for that specific layer: Pressure[bar], Temperature[K], gases[mol/mol], aerosols [kg/kg] - Optional fields: Altitude[km], X_size[m, aerosol X particle size]
        self.layer = np.genfromtxt(layers, delimiter=",")
        #:str: Parameters defining the 3D General Circulation Model grid: num_lons, num_lats, num_alts, lon0, lat0, delta_lon, delta_lat, variables (csv)
        self.gcm_parameters = config.get("ATMOSPHERE-GCM-PARAMETERS")

        #:str: The structure of the atmosphere, None / Equilibrium:'Hydrostatic equilibrium' / Coma:'Cometary expanding coma'
        self.structure = config["ATMOSPHERE-STRUCTURE"]
        #:float: For equilibrium atmospheres, this field defines the surface pressure; while for cometary coma, this field indicates the gas production rate
        self.pressure = float(config["ATMOSPHERE-PRESSURE"])
        #:str: The unit of the ATMOSPHERE-PRESSURE field, Pa:Pascal / bar / kbar / mbar / ubar / at / atm / torr / psi / gas:'molecules / second' / gasau:'molecules / second at rh=1AU'
        self.punit = config["ATMOSPHERE-PUNIT"]
        #:float: For atmospheres without a defined P/T profile, this field indicates the temperature across all altitudes
        self.temperature = float(config["ATMOSPHERE-TEMPERATURE"])
        #:float: Molecular weight of the atmosphere [g/mol] or expansion velocity [m/s] for expanding atmospheres
        self.weight = float(config["ATMOSPHERE-WEIGHT"])
        #:str: Continuum processes to be included in the calculation
        self.continuum = config["ATMOSPHERE-CONTINUUM"]
        #:str: For expanding cometary coma, this field indicates the photodissociation lifetime of the molecules [s]
        self.tau = config["ATMOSPHERE-TAU"]
        #:int: When performing scattering aerosols calculations, this parameter indicates the number of n-stream pairs - Use 0 for extinction calculations only (e.g. transit, occultation)
        self.nmax = int(config["ATMOSPHERE-NMAX"])
        #:int: When performing scattering aerosols calculations, this parameter indicates the number of scattering Legendre polynomials used for describing the phase function - Use 0 for extinction calculations only (e.g. transit, occultation)
        self.lmax = int(config["ATMOSPHERE-LMAX"])
        #:str: Description establishing the source/reference for the vertical profile
        self.description = config["ATMOSPHERE-DESCRIPTION"]

        # Store all the other atmosphere parameters
        self.other = {}
        for key, value in config.items():
            match = re.match(r"ATMOSPHERE-(.*?)(-\d+)?$", key)
            if match is not None and match[1].lower().replace("-","_") not in dir(self):
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
            "ATMOSPHERE-LAYERS-MOLECULES": ",".join([str(v) for v in self.layers_molecules]),
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
        }
        for i in range(1, self.layers+1):
            config[f"ATMOSPHERE-LAYER-{i}"] = np.array2string(self.layer[i-1], separator=",", max_line_width=np.inf)[1:-1]
        if self.gcm_parameters is not None:
            config["ATMOSPHERE-GCM-PARAMETERS"] = self.gcm_parameters
        
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

class PSG:
    def __init__(self) -> None:
        # self.server = 'https://psg.gsfc.nasa.gov'
        self.server = "http://localhost:3000"

        # Read configuration from file
        config_file = join(dirname(__file__), "psg_cfg.txt")
        self.config = self.read_config(config_file)

        self.atmosphere = PSG_Atmosphere(self.config)

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
            with NamedTemporaryFile("w") as output:
                output.write(result.stdout.decode())
                output.flush()
                import_file_to_cache(url, output.name, pkgname=PKGNAME)

        # Return the filename in the cache
        result = download_file(url, cache=True, pkgname=PKGNAME)
        return result

    @staticmethod
    def clear_cache():
        clear_download_cache(pkgname=PKGNAME)

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


# Toggle Plotting
plot = True

# Local directory
localdir = dirname(__file__)
spectra_dir = join(localdir, "spectra")
figures_dir = join(localdir, "figures")
psg_cfg_file = join(localdir, "psg_cfg.txt")
cfg_file = join(localdir, "config.txt")

# Create directories if necessary
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(spectra_dir, exist_ok=True)

lam1 = 0.9  # Lowest wavelength [um]
lam2 = 5.3  # Maximum wavelength [um]
dfrq = 500.0  # Spectral range per window [cm-1]

f1 = int(1e4 / lam2)
fm = int(1e4 / lam1)

mols = ["ref", "PH3", "ClO", "H2S", "HCl", "HF", "HCN", "HNO3", "NH3", "C2H4", "C2H6"]
ppbs = ["1", "10", "100"]


psg = PSG()
psg.atmosphere.nmax = 21
psg.atmosphere.lmax = 84


def set_config(psg, mol, night, ppb):
    if night == "day":
        psg.config["GEOMETRY-OFFSET-EW"] = "-0.3"
    else:
        psg.config["GEOMETRY-OFFSET-EW"] = "0.3"
    if mol == "ref":
        psg.atmosphere.gas = "CO2,N2,CO,O2,SO2,H2O,O3".split(",")
    else:
        psg.atmosphere.gas = f"CO2,N2,CO,O2,SO2,H2O,O3,{mol}".split(",")
        psg.atmosphere.abun[7] = ppb * units.Unit("ppb")
    return psg


def request_spectrum(psg, mol, night, ppb, f1, fm):
    n_segments = np.ceil((fm - f1) / dfrq) + 1
    fs = f1 + np.arange(n_segments) * dfrq
    fs[-1] = fm
    fs = fs.astype(int)

    spec = []
    for fq, f2 in tqdm(
        zip(fs[:-1], fs[1:]), total=n_segments - 1, desc="Segment", leave=False
    ):
        psg.config["GENERATOR-RANGE1"] = "%.1f" % fq
        psg.config["GENERATOR-RANGE2"] = "%.1f" % f2

        file = f"{mol}-{night}-{fq}-{ppb}.txt"
        file = join(spectra_dir, file)
        data = psg.request(psg_type="rad")
        spec += [data]
    spec = np.concatenate(spec)
    return spec


# Calculate reference
reference = {}
for night in tqdm(["day", "night"], desc="Day/Night", leave=False):
    mol, ppb = "ref", 1
    psg = set_config(psg, mol, night, ppb)
    reference[night] = request_spectrum(psg, mol, night, ppb, f1, fm)

# Calculate spectra
for imol, mol in tqdm(enumerate(mols), total=len(mols), desc="Species", leave=False):
    for night in tqdm(["day", "night"], desc="Day/Night", leave=False):
        data = {}
        for ppb in tqdm(ppbs, desc="PPB", leave=False):
            psg = set_config(psg, mol, night, ppb)
            spec = request_spectrum(psg, mol, night, ppb, f1, fm)
            data[ppb] = spec

        # Convert to Angstrom
        # spec[:, 0] = 1e8 / spec[:, 0]
        if plot:
            spec = data["100"]
            ref = reference[night]

            mt = np.median(ref[:, 1])
            dspec = spec[:, 1] - ref[:, 1]
            imin = np.argmin(dspec / mt)
            if dspec[imin] == 0.0:
                continue
            ind = (np.abs(ref[:, 0] - ref[imin, 0]) < 20).nonzero()[0]
            mw = np.median(ref[ind, 1])
            snr = -dspec[imin] / mw
            abun = 3.0 * (0.02 / snr) * 100.0
            print(f"{mol:7s} {night:7s} {-dspec[imin]/mw:8.2f} {mw:8.2f} {abun:.1e}")

            pl, ax = plt.subplots(2, 1, figsize=(10, 8))

            for ppb in ppbs[::-1]:
                ax[0].plot(
                    1e8 / data[ppb][:, 0],
                    data[ppb][:, 1],
                    label=f"Flux {ppb} ppbv",
                    linewidth=0.1,
                )
            ax[0].set_yscale("log")
            ax[0].set_ylabel("Spectral irradiance [Jy] within 1 arcsec")
            ax[0].set_ylim([1, 2e3])
            ax[0].set_xlim([1e8 / fm, 1e8 / f1])
            ax[0].set_title(
                f"{night} - {mol} - Best at {ref[imin, 0]:.3f} cm-1 ({1e8 / ref[imin, 0]:.3f} Å)"
            )
            ax[0].legend()

            ax2 = ax[0].twinx()
            ax2.plot(1e8 / ref[:, 0], dspec / mt)
            ax2.set_xlabel("Wavelength [Å]")
            ax2.set_ylabel(f"{mol} absorption w.r.t. median flux")

            for ppb in ppbs[::-1]:
                ax[1].plot(
                    1e8 / data[ppb][ind, 0],
                    data[ppb][ind, 1],
                    label=f"{mol} {ppb} ppbv",
                )
            ax[1].plot(1e8 / ref[ind, 0], ref[ind, 1], label="Reference")
            ax[1].set_xlabel("Wavelength [Å]")
            ax[1].set_ylabel("Spectral irradiance [Jy] within 1 arcsec")
            ax[1].set_xlim([1e8 / ref[ind[-1], 0], 1e8 / ref[ind[0], 0]])
            ax[1].legend()

            plt.tight_layout()

            fname = f"{mol}-{night}.png"
            plt.savefig(join(figures_dir, fname))
