# ELEC5705 Project

This directory collects the starting materials for the course project based on the paper:

- `paper/A_46_mu_textW_13_b_6.4_MS_s_SAR_ADC_With_Background_Mismatch_and_Offset_Calibration.pdf`

It also contains the presentation that was already given on this paper:

- `presentation/paper_presentation_sectioned.tex`
- `presentation/paper_presentation_sectioned.pdf`
- `presentation/Figures/`

Current project topic:

- A 46 uW 13-bit 6.4 MS/s SAR ADC with background mismatch and offset calibration

Next steps can build from here:

- extract the architecture and main circuit blocks from the paper
- define which blocks to reproduce or model for the project
- set up a project report and implementation plan

Implementation scaffold:

- `rtl/`: real digital RTL blocks for the calibration path
- `model/`: behavioral models for the ADC, comparator offset, and DAC mismatch
- `sim/`: testbenches and local run scripts
- `notes/`: project planning and architecture notes
- `results/`: simulation outputs and generated artifacts
- `report/`: LaTeX report source, PDF, and plot data
- `synthesis/`: synthesized digital-calibration artifacts and block views
- `tools/`: helper scripts for regenerating report data and local model evaluation
