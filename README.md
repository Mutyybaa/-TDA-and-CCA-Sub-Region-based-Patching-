#  Sub-Region-Based Patching for Brain Tumor Segmentation

## Description
This project presents a novel **sub-region-based patching technique** for brain tumor segmentation from 3D MRI scans. Unlike traditional patching methods (fixed, random, or overlapping), this approach intelligently selects patches from both the **whole tumor (WT-Patch)** and **core tumor (TC-Patch)** regions—**without requiring ground truth masks**—making it suitable for inference and efficient model training.  

The method addresses key challenges in medical image segmentation such as:
- High computational cost of 3D MRI data  
- Class imbalance between tumor and non-tumor pixels  
- Redundant background information in overlapping patching  
- Nonlinear increase in patch count due to overlap  


---

##  Dataset Information
- **Dataset Name:** BraTS 2020 (Brain Tumor Segmentation Challenge)  
- **Data Type:** 3D MRI scans (T1, T1CE, T2, FLAIR modalities)  
- **Labels:** Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)  
- **Source:** [MICCAI BraTS 2020 Challenge](https://www.med.upenn.edu/cbica/brats2020/)  
- **Preprocessing Steps:**
  - Skull stripping and intensity normalization  
  - Bias field correction  
  - Resampling to isotropic voxel spacing  

---

##  Code Information
- **2D_CCA_Based_Patching:** `2D_Patch_Extraction.py` — runs 2D patching 
- **2D_TDA_Based_Patching:** `2D__TDA_pipeline.py` — implements the proposed TDA-based patch selection  
- **3D_TC_Based_Patching:** `TC_Based_Patching.py` — implements TC-based patch selection  
- **3D_WT_Based_Patching:** `WT_Based_Patching.py` — implements WT-based patch selection  
- **3D_Image_Reconstruction:** `Patch_Coordinates.py` — 3D MRI reconstruction from patch
---

## ⚙️ Usage Instructions

### 1. Clone the repository
```bash
git clone https://github.com/username/subregion-patching-brain-tumor.git
cd subregion-patching-brain-tumor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset setup
Download the **BraTS 2020** dataset and place it in the `data/` directory:
```
data/
 ├── BraTS2020_Training/
 └── BraTS2020_Validation/
```

### 4. Run patch generation
```bash
python WT_Based_Patching.py --input data/BraTS2020_Training --output patches/
```


---

## Requirements
- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy ≥ 1.23  
- nibabel ≥ 3.2  
- scikit-image ≥ 0.19  
- matplotlib ≥ 3.5  
- tqdm  



---

##  Methodology
1. **Patch Extraction:**  
   - Sub-region-based selection of WT-Patches and TC-Patches without ground truth masks.  
   - Adaptive overlap to minimize background redundancy.  

2. **Preprocessing:**  
   - Normalization and standardization of MRI modalities.  
   - Cropping and resizing for uniform patch dimensions.  

3. **Model Training:**  
   - U-Net architecture with Dice loss optimization.    

4. **Evaluation:**  
   - Dice coefficient and IoU metrics across tumor subregions.  
   - Comparison with traditional patching techniques.  

---

##  Citations
If you use this dataset or code in your research, please cite:
```
@article{asghar2023tumor,
  title={Tumor-centered patching for enhanced medical image segmentation},
  author={Asghar, Mutyyba and Shahid, Ahmad Raza and Jamil, Akhtar and Aftab, Kiran and Enam, Syed Ather},
  journal={arXiv preprint arXiv:2308.12168},
  year={2023}
}
```
Also cite the original BraTS dataset:
```
@inproceedings{menze2015multimodal,
  title={The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)},
  author={Menze, Bjoern H. et al.},
  booktitle={IEEE Transactions on Medical Imaging},
  year={2015}
}
```

---

##  License 
- **License:**  you are free to use, modify, and distribute this code with attribution for research purpose only.  

