"""
EMPIR Pipeline Parameter Settings

Edit this file to customize parameters for different processing modes.
Users can modify these settings without touching the main Analysis code.
"""

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