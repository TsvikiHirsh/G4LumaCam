"""
Nanoguide simulations: an AmBe source beside a fibre-structured scintillator viewed end-on.

The nanoguide is a square guide (10 mm x 10 mm x 100 mm) of scintillating fibres (EJ-200
type PVT) in PMMA, the fibres along its length. In the simulation the length runs along z,
the optical axis: Geant4 moves every guided scintillation photon from its birth (x, y) to
the viewed end face (``/lumacam/nanoguide``), which the lens images as its object plane
(z = 0 in the SimPhotons files). ``Config.ambe_nanoguide`` sets this up; the functions here
give the source geometry and sort the simulated events by their separated light clusters.

Coordinates: x and y are the end-face coordinates of the SimPhotons files (mm, guide
centred on the axis). The source azimuth is measured in the same x-y plane, from +x
towards +y; the viewed end face is at Geant4 z = length and the far end at z = 0.
"""
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# AmBe neutron spectrum shipped with Geant4 10.6 (examples/advanced/underground_physics/
# ambe_spectrum.mac, "estimated neutron spectrum from Jaime's paper"): point-wise values
# (E in MeV, relative intensity), linearly interpolated, mean energy 3.8 MeV. The ISO
# 8529-1 241Am-Be spectrum (mean 4.2 MeV) can replace it through ``spectrum=``.
AMBE_SPECTRUM: List[Tuple[float, float]] = [
    (0.0, 10000), (0.2, 11000), (0.4, 12000), (0.6, 13000), (0.8, 12000),
    (1.0, 10000), (1.2, 9000), (1.4, 11000), (1.6, 10000), (1.8, 11000),
    (2.0, 9000), (2.2, 14000), (2.4, 11000), (2.6, 12000), (2.8, 15000),
    (3.0, 17000), (3.2, 16000), (3.4, 15000), (3.6, 16500), (3.8, 17000),
    (4.0, 17000), (4.2, 14000), (4.4, 13500), (4.6, 14500), (4.8, 14000),
    (5.0, 13500), (5.2, 13000), (5.4, 12500), (5.6, 11000), (5.8, 10000),
    (6.0, 9800), (6.2, 8000), (6.4, 7700), (6.6, 7000), (6.8, 6800),
    (7.0, 6000), (7.2, 5000), (7.4, 4000), (7.6, 3600), (7.8, 3000),
    (8.0, 2000), (8.2, 1900), (8.4, 1750), (8.6, 1800), (8.8, 1825),
    (9.0, 1600), (9.2, 1400), (9.4, 1200), (9.6, 1100), (9.8, 1000),
    (10.0, 800),
]

# Gamma ray from the first excited state of 12C, emitted by AmBe (about 0.6 per neutron)
AMBE_GAMMA_MEV = 4.438

# Focus of Lens(kind="microscope") on the end face (z = 0): traced point sources emitting
# into the fibre exit cone give a 30 um rms spot (under one 39 um pixel) and a 1.005 image
# scale; at 11.0 the spot is 22 um and the scale 1.012. Use trace_rays(zfine=MICROSCOPE_ZFINE).
MICROSCOPE_ZFINE = 11.5


def load_spectrum(spectrum: Union[str, Path, Sequence[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Spectrum points (E in MeV, intensity) from a list or a two-column text file."""
    if isinstance(spectrum, (str, Path)):
        data = np.loadtxt(spectrum, ndmin=2)
        return [(float(e), float(w)) for e, w in data[:, :2]]
    return [(float(e), float(w)) for e, w in spectrum]


def source_window(azimuth: float, distance: float, depth: Optional[float] = None,
                  length: float = 100.0, width: float = 10.0, margin: float = 0.5) -> dict:
    """
    Point-source position and the emission window that covers the whole nanoguide.

    Geant4's isotropic GPS source emits along -(sin t cos p, sin t sin p, cos t), so with
    p around the source azimuth and t around 90 deg the neutrons head for the guide. The
    window is uniform in cos t and p, which keeps the emission isotropic inside it.

    Args:
        azimuth: source direction seen from the guide axis, deg (x-y plane, from +x).
        distance: source to guide axis, mm (perpendicular to the axis).
        depth: source position along the guide, mm from the viewed end face
            (default: the middle of the guide).
        length, width: guide length and transverse full width, mm.
        margin: extra coverage around the guide, mm.

    Returns:
        dict with the Geant4 position (x, y, z in mm), the window
        (theta_min, theta_max, phi_min, phi_max in deg) and ``fraction``, the solid
        angle of the window over 4 pi (source neutrons = simulated events / fraction).
    """
    if depth is None:
        depth = length / 2
    r = width / math.sqrt(2) + margin  # circle around the square cross-section
    if distance <= r:
        raise ValueError(f"The source must be outside the guide: distance {distance} mm <= {r:.2f} mm")
    zs = length - depth  # Geant4 z of the source (far end at z = 0)
    rho = (distance - r, distance + r)  # transverse path to the near and far side
    # cot t = (zs - z) / rho for the hits at z in [-margin, length + margin]
    cot_max = max((zs + margin) / p for p in rho)
    cot_min = min((zs - length - margin) / p for p in rho)
    theta_min = math.degrees(math.atan2(1.0, cot_max))
    theta_max = math.degrees(math.atan2(1.0, cot_min))
    dphi = math.degrees(math.asin(r / distance))
    phi_min, phi_max = azimuth - dphi, azimuth + dphi
    fraction = ((math.cos(math.radians(theta_min)) - math.cos(math.radians(theta_max)))
                * math.radians(phi_max - phi_min) / (4 * math.pi))
    a = math.radians(azimuth)
    return {
        "x": distance * math.cos(a), "y": distance * math.sin(a), "z": zs,
        "theta_min": theta_min, "theta_max": theta_max,
        "phi_min": phi_min, "phi_max": phi_max,
        "fraction": fraction,
    }


def light_clusters(df: pd.DataFrame, merge: float = 1.0, min_photons: int = 1) -> pd.DataFrame:
    """
    Light clusters on the end face, per event.

    Every photon belongs to the charged particle that emitted it (``parent_id``). The
    particles of one event whose photon centroids lie within ``merge`` mm of each other
    (chained) form one cluster, so a recoil proton and its delta electrons count once.

    Args:
        df: SimPhotons (or traced photons joined with them) with the columns neutron_id,
            parent_id, parentName, parentEnergy, px, py, pz, x, y, toa.
        merge: clustering distance on the end face, mm.
        min_photons: drop clusters with fewer photons.

    Returns:
        One row per cluster: neutron_id, cluster, x, y (photon centroid, mm), n_photons,
        t (earliest photon, ns), particles (emitting particle types, '+'-joined),
        energy (largest emitting-particle energy, MeV), px, py, pz (birth of that
        particle, mm).
    """
    parents = (df.groupby(["neutron_id", "parent_id"], sort=True)
                 .agg(particle=("parentName", "first"), energy=("parentEnergy", "first"),
                      px=("px", "first"), py=("py", "first"), pz=("pz", "first"),
                      x=("x", "mean"), y=("y", "mean"), n_photons=("x", "size"),
                      t=("toa", "min"))
                 .reset_index())
    labels = np.zeros(len(parents), dtype=int)
    nid = parents["neutron_id"].to_numpy()
    xy = parents[["x", "y"]].to_numpy()
    starts = np.flatnonzero(np.r_[True, nid[1:] != nid[:-1]])
    ends = np.r_[starts[1:], len(parents)]
    for s, e in zip(starts, ends):
        # single-linkage clustering of this event's emitting particles
        lab = np.arange(e - s)
        if e - s > 1:
            d = np.hypot(*(xy[s:e, None, :] - xy[None, s:e, :]).transpose(2, 0, 1))
            for i, j in zip(*np.nonzero(np.triu(d < merge, 1))):
                lab[lab == lab[j]] = lab[i]
        labels[s:e] = np.unique(lab, return_inverse=True)[1]
    parents["cluster"] = labels
    parents["wx"] = parents["x"] * parents["n_photons"]
    parents["wy"] = parents["y"] * parents["n_photons"]

    g = parents.groupby(["neutron_id", "cluster"], sort=True)
    clusters = g.agg(wx=("wx", "sum"), wy=("wy", "sum"), n_photons=("n_photons", "sum"), t=("t", "min"),
                     particles=("particle", lambda p: "+".join(sorted(set(p)))))
    clusters["x"] = clusters.pop("wx") / clusters["n_photons"]
    clusters["y"] = clusters.pop("wy") / clusters["n_photons"]
    lead = parents.loc[g["energy"].idxmax().to_numpy(), ["energy", "px", "py", "pz"]]
    clusters[["energy", "px", "py", "pz"]] = lead.to_numpy()
    clusters = clusters.reset_index()[["neutron_id", "cluster", "x", "y", "n_photons", "t",
                                       "particles", "energy", "px", "py", "pz"]]
    return clusters[clusters["n_photons"] >= min_photons].reset_index(drop=True)


def cluster_pairs(clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Every pair of light clusters within one event (events with two or more clusters).

    Args:
        clusters: output of ``light_clusters``.

    Returns:
        One row per pair: neutron_id, distance (end-face separation, mm), dt (time
        difference of the earliest photons, ns), n_photons_1/2, particles_1/2 and
        depth (z distance between the emitting particles' births, mm), the pair with
        the brighter cluster first.
    """
    rows = []
    for nid, g in clusters.groupby("neutron_id"):
        if len(g) < 2:
            continue
        g = g.sort_values("n_photons", ascending=False).reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.loc[i], g.loc[j]
                rows.append({
                    "neutron_id": nid,
                    "distance": math.hypot(a["x"] - b["x"], a["y"] - b["y"]),
                    "dt": b["t"] - a["t"],
                    "n_photons_1": a["n_photons"], "n_photons_2": b["n_photons"],
                    "particles_1": a["particles"], "particles_2": b["particles"],
                    "depth": abs(a["pz"] - b["pz"]),
                })
    return pd.DataFrame(rows, columns=["neutron_id", "distance", "dt", "n_photons_1", "n_photons_2",
                                       "particles_1", "particles_2", "depth"])
