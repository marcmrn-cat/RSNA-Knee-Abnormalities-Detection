import re
import numpy as np
import pandas as pd
from typing import Dict
from src.config import Config

class ClinicalNLPParser:
    def __init__(self):
        # Multilingual negation patterns
        self.negations = r"(no|not|without|negative for|unremarkable|intact|sin|ausencia|no hay|kein|keine|pas|sans|geen)"
        
        self.target_patterns = {
            'ACL': r"(?i)(anterior cruciate ligament|acl)",
            'MCL': r"(?i)(medial collateral ligament|mcl)",
            'Medial_Meniscus': r"(?i)(medial meniscus)",
            'Lateral_Meniscus': r"(?i)(lateral meniscus)",
            'Medial_OA': r"(?i)(medial compartment.*osteoarthritis|medial.*oa)",
            'Lateral_OA': r"(?i)(lateral compartment.*osteoarthritis|lateral.*oa)",
            'PF_OA': r"(?i)(patellofemoral.*osteoarthritis|pf.*oa)",
            'Effusion': r"(?i)(joint effusion|effusion)",
            'Synovitis': r"(?i)(synovitis)",
            'Bakers': r"(?i)(baker'?s cyst|popliteal cyst)",
            'Contusion': r"(?i)(bone contusion|bruise)",
            'Fracture': r"(?i)(fracture|fx)"
        }

    def parse_report(self, report_text: str) -> Dict[str, float]:
        if not isinstance(report_text, str) or pd.isna(report_text):
            return {target: np.nan for target in Config.TARGETS}
            
        labels = {}
        for target, pattern in self.target_patterns.items():
            if re.search(pattern, report_text):
                # Negation lookbehind equivalent (checking 6-word preceding window)
                negated_pattern = rf"(?i){self.negations}(?:\s+\w+){{0,5}}\s+{pattern}"
                if re.search(negated_pattern, report_text):
                    labels[target] = 0.0
                else:
                    labels[target] = 1.0
            else:
                labels[target] = 0.0 
        return labels

def apply_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    parser = ClinicalNLPParser()
    if 'Report' in df.columns:
        pseudo_labels = df['Report'].apply(parser.parse_report).apply(pd.Series)
        for col in Config.TARGETS:
            if col in df.columns:
                df[col] = df[col].fillna(pseudo_labels[col])
            else:
                df[col] = pseudo_labels[col]
    return df
