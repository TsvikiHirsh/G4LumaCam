"""Nanoguide simulation: the AmBe source window must cover every direction that hits the
guide, the macro must switch the guide on, the cluster analysis must separate two recoils,
and (when the executable is built) Geant4 must put the guided photons on the end face,
inside the fibre exit cone, with recoils pointing away from the source."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))
from lumacam.nanoguide import cluster_pairs, light_clusters, source_window
from lumacam.simulate import Config

EXE = os.path.join(HERE, 'build', 'lumacam')
NA = np.sqrt(1.58 ** 2 - 1.49 ** 2)


def hits_box(origin, d, half=(5.0, 5.0), zrange=(0.0, 100.0)):
    """Slab test: does the ray origin + t d (t > 0) cross the guide box?"""
    lo = np.array([-half[0], -half[1], zrange[0]])
    hi = np.array([half[0], half[1], zrange[1]])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo - origin) / d
        t2 = (hi - origin) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    return (tmax >= np.maximum(tmin, 0))


@pytest.mark.parametrize('azimuth,distance,depth', [(30, 50, None), (200, 25, 10), (-90, 120, 95)])
def test_window_covers_guide(azimuth, distance, depth):
    w = source_window(azimuth, distance, depth)
    origin = np.array([w['x'], w['y'], w['z']])
    rng = np.random.default_rng(1)
    # isotropic over 4 pi in GPS's convention d = -(sin t cos p, sin t sin p, cos t)
    cost = rng.uniform(-1, 1, 2_000_000)
    phi = rng.uniform(-np.pi, np.pi, cost.size)
    sint = np.sqrt(1 - cost ** 2)
    d = -np.c_[sint * np.cos(phi), sint * np.sin(phi), cost]
    hit = hits_box(origin, d)
    theta = np.degrees(np.arccos(cost))
    dphi = (np.degrees(phi) - azimuth + 180) % 360 - 180
    inside = ((theta >= w['theta_min']) & (theta <= w['theta_max'])
              & (np.abs(dphi) <= (w['phi_max'] - w['phi_min']) / 2))
    assert hit.sum() > 1000
    assert not (hit & ~inside).any(), 'directions that hit the guide fall outside the window'
    assert abs(inside.mean() - w['fraction']) < 3 * np.sqrt(w['fraction'] / cost.size) + 1e-4


def test_config_macro(tmp_path):
    c = Config.ambe_nanoguide(azimuth=30, distance=50, num_events=1000, source_strength=1e6)
    c.write(str(tmp_path / 'macro.mac'))
    mac = (tmp_path / 'macro.mac').read_text()
    for line in ['/gps/pos/type Point', '/gps/ene/type Arb', '/gps/hist/inter Lin',
                 '/lumacam/scintMaterial Nanoguide', '/lumacam/scintSize 1.0 cm',
                 '/lumacam/scintThickness 10.0 cm', '/lumacam/nanoguide true']:
        assert line in mac
    assert '/gps/pos/halfx' not in mac
    w = source_window(30, 50)
    assert c.solid_angle_fraction == pytest.approx(w['fraction'], rel=1e-5)
    # events spread over the time the source needs to send them into the window
    assert c.tmax == pytest.approx(1000 / (1e6 * w['fraction']) * 1e9)
    g = Config.ambe_nanoguide(azimuth=30, distance=50, particle='gamma')
    assert g.particle == 'gamma' and g.energy == pytest.approx(4.438)


def test_clusters_and_pairs():
    rng = np.random.default_rng(2)

    def photons(nid, pid, name, x, y, n, t, e=2.0):
        return pd.DataFrame(dict(neutron_id=nid, parent_id=pid, parentName=name, parentEnergy=e,
                                 px=x, py=y, pz=50.0, x=x + rng.normal(0, 0.05, n),
                                 y=y + rng.normal(0, 0.05, n), toa=t + rng.exponential(3, n)))

    df = pd.concat([
        photons(0, 2, 'proton', -4.5, 0.0, 200, 10.0, e=3.0),  # first n-p scatter
        photons(0, 7, 'e-', -4.2, 0.1, 30, 10.0, e=0.01),      # its delta electron
        photons(0, 3, 'proton', 4.5, 0.0, 120, 10.4, e=1.0),   # second n-p scatter, 9 mm away
        photons(1, 2, 'proton', 1.0, 1.0, 5, 50.0),            # dim single recoil
    ], ignore_index=True)
    cl = light_clusters(df, merge=1.0, min_photons=10)
    assert sorted(cl.groupby('neutron_id').size().to_dict().items()) == [(0, 2)]
    first = cl[cl.n_photons == 230].iloc[0]
    assert first.particles == 'e-+proton' and first.energy == 3.0 and first.px == -4.5
    pr = cluster_pairs(cl)
    assert len(pr) == 1
    assert pr.distance.iloc[0] == pytest.approx(9.0, abs=0.05)
    assert pr.dt.iloc[0] == pytest.approx(0.4, abs=0.2)


@pytest.mark.skipif(not os.path.exists(EXE), reason='build/lumacam not built')
def test_geant4_end_face(tmp_path):
    c = Config.ambe_nanoguide(azimuth=30, distance=50, num_events=2000)
    c.write(str(tmp_path / 'body.mac'))
    (tmp_path / 'macro.mac').write_text('/random/setSeeds 111 222\n' + (tmp_path / 'body.mac').read_text())
    subprocess.run([EXE, 'macro.mac'], cwd=tmp_path, check=True, capture_output=True)
    df = pd.read_csv(tmp_path / 'SimPhotons' / 'sim_data_0.csv')
    assert len(df) > 1000
    assert (df.z == 0).all()
    assert df.x.abs().max() <= 5 and df.y.abs().max() <= 5
    assert np.hypot(df.dx, df.dy).max() <= NA + 1e-4
    # photons arrive after their trip along the fibre
    assert ((df.toa - df.pulse_time_ns) > (100 - df.pz) * 1.58 / 299.792458).all()
    # recoil protons point away from the source (azimuth 30 deg -> 210 deg)
    p = df[df.parentName == 'proton'].groupby(['neutron_id', 'parent_id']).agg(
        px=('px', 'first'), py=('py', 'first'), x=('x', 'mean'), y=('y', 'mean'))
    p = p[np.hypot(p.x - p.px, p.y - p.py) > 0.05]
    mean = np.exp(1j * np.arctan2(p.y - p.py, p.x - p.px)).mean()
    assert abs(mean) > 0.7
    assert abs((np.degrees(np.angle(mean)) - 210 + 180) % 360 - 180) < 20
