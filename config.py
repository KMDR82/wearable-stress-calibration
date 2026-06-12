
from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
   
    wesad_root: str = "/kaggle/input/wesad/WESAD"
   
    second_stress_root: str = ""  # boşsa cross-dataset adımı atlanır

    out_dir: str = "/kaggle/working/outputs"
    ckpt_dir: str = "/kaggle/working/checkpoints"

    
    wrist_fs: int = 64            
    chest_fs: int = 70            
    window_sec: int = 60          
    window_stride_sec: int = 60   
    wrist_channels: List[str] = field(default_factory=lambda: ["BVP", "EDA", "TEMP", "ACC"])

   
    keep_labels: tuple = (1, 2, 3)
    stress_label: int = 2
    nonstress_labels: tuple = (1, 3)

    
    normalization: str = "robust_per_subject"

   
    ssl_mask_ratio: float = 0.5
    ssl_epochs: int = 50
    ssl_batch_size: int = 256
    ssl_lr: float = 1e-3
    encoder_width: int = 64
    embed_dim: int = 128

    
    finetune_epochs: int = 30
    clf_batch_size: int = 128
    clf_lr: float = 3e-4
    label_fractions: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 1.0])

   
    ece_bins: int = 15
    dca_thresholds: List[float] = field(default_factory=lambda: [round(0.01 + 0.01 * i, 4) for i in range(60)])  # 0.01..0.60

    
    run_loso: bool = True          
    run_site_shift: bool = True    
    run_cross_dataset: bool = False  

   
    ablate_ssl_on_off: bool = True        
    ablate_calibration_on_off: bool = True  
    ablate_mask_ratios: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    ablate_modality_subsets: List[tuple] = field(default_factory=lambda: [("BVP",), ("BVP", "EDA"), ("BVP", "EDA", "TEMP", "ACC")])
    ablate_loso_vs_random: bool = True    

   
    seed: int = 1337
    device: str = "cuda"           
    demo_mode: bool = False        

    def ensure_dirs(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)


CFG = Config()
