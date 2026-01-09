import json
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import classifiers as modules
import classifier1
import classifier3

# --- Configuration ---
DEFAULT_INTENTS_JSON = "intents.json"

@dataclass
class VotingResources:
    idx1: classifier1.IntentIndex
    idx3: classifier3.IntentIndex
    intents: Dict[str, Dict[str, List[str]]]

def load_resources(intents_path: str = DEFAULT_INTENTS_JSON) -> VotingResources:
    """Loads resources for both classifiers."""
    intents = classifier1.load_intents_from_json(intents_path)
    intents_hash = classifier1._file_sha1(intents_path)
    
    # Load indexes
    # Note: They use different cache files (defined in their modules)
    print("Loading Index 1 (Centroid)...")
    idx1 = classifier1.build_or_load_index(intents, intents_hash)
    
    print("Loading Index 3 (Ensemble KNN)...")
    idx3 = classifier3.build_or_load_index(intents, intents_hash)
    
    return VotingResources(idx1, idx3, intents)

def predict_voting(text: str, res: VotingResources) -> Dict:
    """
    Predicts endpoint using a voting mechanism between Classifier 1 and Classifier 3.
    """
    
    pred1 = classifier1.predict_endpoint(text, res.idx1, res.intents)
    pred3 = classifier3.predict_endpoint(text, res.idx3, res.intents)
    
    ep1 = pred1["endpoint"]
    score1 = pred1["score"]
    method1 = pred1["method"]
    
    ep3 = pred3["endpoint"]
    score3 = pred3["score"]
    method3 = pred3["method"]
    
    # --- Voting Logic ---
    
    # Priority to Classifier 3 (Ensemble KNN) because it has proven higher accuracy (0.89 vs lower for C1).
    # C1 tends to be overconfident.
    
    # Threshold for trusting C3 implicitly
    C3_TRUST_THRESHOLD = 0.85

    # Case 1: Agreement
    if ep1 == ep3 and ep1 != "":
        final_score = max(score1, score3)
        return {
            "endpoint": ep1,
            "score": final_score,
            "method": f"voting_agreement",
            "details": {"c1": pred1, "c3": pred3}
        }

    # Case 2: Disagreement
    # If C3 is confident enough, trust it.
    if score3 >= C3_TRUST_THRESHOLD:
        return {**pred3, "method": f"voting_trust_c3_high_score", "details": {"c1": pred1, "c3": pred3}}

    # If C3 is not confident, check if C1 is significantly better?
    # Or fallback to score comparison (risky if C1 is overconfident)
    
    # If scores are close, prefer C3
    if score3 >= score1 - 0.03: # Handicap: C3 wins even if slightly lower
         return {**pred3, "method": f"voting_prefer_c3", "details": {"c1": pred1, "c3": pred3}}

    # If C1 is MUCH more confident than C3, trust C1
    if score1 > score3:
         return {**pred1, "method": f"voting_c1_override_low_c3", "details": {"c1": pred1, "c3": pred3}}
    
    # Default to C3
    return {**pred3, "method": f"voting_default_c3", "details": {"c1": pred1, "c3": pred3}}


def demo():
    p = argparse.ArgumentParser()
    p.add_argument("--intents", type=str, default=DEFAULT_INTENTS_JSON)
    p.add_argument("--text", type=str)
    args = p.parse_args()
    
    res = load_resources(args.intents)
    
    text = args.text or "Weterynarz Marka podała koniu Mewie 2 tabletki leku przeciwbólowego."
    print(f"Text: {text}")
    
    result = predict_voting(text, res)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    demo()
