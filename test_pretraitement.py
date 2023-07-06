from pathlib import Path
import main

if __name__ == "__main__":
    a = Path("/var/dev/frmi_bpd/datasetFSL/Control/.nii")
    print(a.name)
    main.sclice_timing_correction(a)
    main.BET(a)
    main.run_mcflirt(a)
    main.spacial_smoothing(a)
    main.intensity_normalization(a)