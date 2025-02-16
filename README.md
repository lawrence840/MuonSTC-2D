# MuonSTC-2D: 2D Muography algorithm from USTC
2D Muography for Average Density Reconstruction Using Cosmic Muons

**MuonSTC-2D** is a muon 2D transmission imaging code developed by the USTC team. By acquiring muon data from open sky and objects in experimental or simulated scenarios, a 2D density map of the object can be obtained. We applied it to image the teaching and administrative building in the east campus and achieved excellent results.

---

## Citation

If you use or refer to **MuonSTC**, please cite the following paper:  
> **He, Z. Y., Pan, Z. W., Liu, Y. L., Wang, Z., Lin, Z. B., Chen, Z., ... & Ye, B. J. (2024). Feasibility and optimization study of a two-dimensional density reconstruction method for large-object muography. *Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 1061*, 169138.**  
>
> *[Link to the paper](https://doi.org/10.1016/j.nima.2024.169138))*


# Usage:
The files `BF2.txt` and `sky2.txt` contain the muon data received by two layers of detectors under BF and opensky situations. The data includes the positions \((x_1, y_1, z_1)\) and \((x_2, y_2, z_2)\) as well as the energies \(E_1\) and \(E_2\). (For experimental data, if energy information is unavailable, substitute values greater than the energy threshold.) The file format is:  
```
x1 y1 z1 x2 y2 z2 E1 E2
```
Replace these two files with your own simulated or experimental data.

The files `density-calculate.txt` and `length-calculate.txt` provide the thickness and average density angular distribution of the object from the detector's perspective. If only a 2D density map reconstruction is required, the thickness distribution alone is sufficient. However, if you need to calculate the Mean Squared Error (MSE) between the reconstructed density and the theoretical density, you must also input the theoretical density distribution.

The files `output_power.txt` and `output_quad.txt` are used for interpolation solutions of two complex formulas when calculating opacity from the muon's minimum energy.

To use the code, adjust the following parameters:
- Minimum energy \(E_m\)
- Angular separation \(d\)
- Detector azimuthal offset \(\Phi\)
- Detector placement angle (`normal_vector_x`)
- Imaging range (\(\theta_{\text{Min}}\), \(\theta_{\text{Max}}\), \(\phi_{\text{Min}}\), \(\phi_{\text{Max}}\))

Then, make corresponding adjustments to the matrix blocks in the subsequent code.

If you wish to input your own muon differential flux, you need to modify the formula in `In[4]: Minimum Energy Calculation`.
