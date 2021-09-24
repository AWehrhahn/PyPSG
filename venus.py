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
from tqdm import tqdm
import subprocess

import os
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

from psg import PSG

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
