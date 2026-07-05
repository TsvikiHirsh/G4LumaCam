"""
EMPIR Pipeline Parameter Settings

Edit this file to customize parameters for different processing modes.
Users can modify these settings without touching the main Analysis code.
"""

# Calibrated detector model (v0.6): Bayesian optimum of the
# gaussian_probabilistic model against PTB air45 per-event data
# (Sigma-chi2 = 0.52 +/- 0.02 over four observables x in/out of focus).
# Afterpulse satellites at the literature rate (Mahon et al. 2024,
# NIM-A 1059 168816); set ap_prob=0 to disable them.
BEST_DETECTOR_MODEL = {
    "detector_model": "gaussian_probabilistic",
    "zfine": 12.6,
    "zscan": 10.0,
    "fnumber": 0.95,
    "blob": 0.405,                    # intensifier PSF sigma (px)
    "decay_time": 16.5,               # P47 phosphor decay (ns)
    "deadtime": 600.0,                # TPX3 per-pixel dead time (ns)
    "gain": 10000.0,
    "model_params": {
        "n_secondaries": 9,           # detected pixels per photon gain spot
        "photon_keep_fraction": 0.241,  # effective optical yield (~QE)
        "ap_prob": 0.012,             # afterpulses per photon (~2% of events)
        "ap_rmax": 5.5,               # satellite displacement radius (px)
        "ap_secondaries": 8,          # pixels per satellite mini-cluster
    },
}

DEFAULT_PARAMS = {
    "in_focus": {
        "pixel2photon": {
            "dSpace": 2,
            "dTime": 100e-09,
            "nPxMin": 8,
            "nPxMax": 100,
            "TDC1": True
        },
        "photon2event": {
            "dSpace_px": 0.001,
            "dTime_s": 5e-08,
            "durationMax_s": 5e-07,
            "dTime_ext": 5
        },
        "event2image": {
            "size_x": 512,
            "size_y": 512,
            "nPhotons_min": 1,
            "nPhotons_max": 1,
            "psd_min": 0,
            "time_extTrigger": "reference",
            "time_res_s": 1.5625e-9,
            "time_limit": 640
        },
    },
    "out_of_focus": {
        "pixel2photon": {
            "dSpace": 2,
            "dTime": 5e-08,
            "nPxMin": 2,
            "nPxMax": 12,
            "TDC1": True
        },
        "photon2event": {
            "dSpace_px": 60,
            "dTime_s": 10e-08,
            "durationMax_s": 10e-07,
            "dTime_ext": 5,
            # anchor multi-photon events to the largest (parent) cluster —
            # robust against intensifier-afterpulse satellites
            "position_mode": "largest"
        },
        "event2image": {
            "size_x": 512,
            "size_y": 512,
            "nPhotons_min": 2,
            "nPhotons_max": 9999,
            "psd_min": 0,
            "time_extTrigger": "reference",
            "time_res_s": 1.5625e-9,
            "time_limit": 640
        },
    },
    "fast_neutrons": {
        "pixel2photon": {
            "dSpace": 2,
            "dTime": 5e-08,
            "nPxMin": 2,
            "TDC1": True
        },
        "photon2event": {
            "dSpace_px": 2,
            "dTime_s": 10e-08,
            "durationMax_s": 10e-07,
            "dTime_ext": 5
        },
        "event2image": {
            "size_x": 512,
            "size_y": 512,
            "nPhotons_min": 2,
            "nPhotons_max": 9999,
            "psd_min": 0,
            "time_extTrigger": "reference",
            "time_res_s": 1.5625e-9,
            "time_limit": 640
        },
    },
    "hitmap": {
        "pixel2photon": {
            "dSpace": 0.001,
            "dTime": 1e-9,
            "nPxMin": 1,
            "TDC1": True
        },
        "photon2event": {
            "dSpace_px": 0.001,
            "dTime_s": 0,
            "durationMax_s": 0,
            "dTime_ext": 5
        },
        "event2image": {
            "size_x": 256,
            "size_y": 256,
            "nPhotons_min": 1,
            "nPhotons_max": 9999,
            "psd_min": 0,
            "time_extTrigger": "reference",
            "time_res_s": 1.5625e-9,
            "time_limit": 640
        },
    },
}