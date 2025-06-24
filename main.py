"""
WildCore Demonstration

This script demonstrates the complete WildCore system with a simulation of 
normal and anomalous behaviors in an AI system.
"""

import numpy as np
import time
from typing import Dict, List, Any
import logging

from wildcore.agent import GutoVectorWildcard
from wildcore.detector import AutoRegulatedPromptDetector
from wildcore.utils import generate_random_embeddings, evaluate_detector

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WildCore")

def demonstrate_complete_system(iterations: int = 20, dimension: int = 768) -> Dict[str, Any]:
    """
    Run a complete demonstration of the WildCore system.
    
    Parameters:
    ----------
    iterations : int, optional
        Number of simulation iterations (default is 20)
    dimension : int, optional
        Dimension of the embedding vectors (default is 768)
        
    Returns:
    -------
    Dict[str, Any]
        Results of the simulation
    """
    # Initialize the components
    guto = GutoVectorWildcard()
    detector = AutoRegulatedPromptDetector(threshold=0.5)
    
    # Initial setup
    logger.info("Initializing the WildCore demonstration")
    guto.take_role("assistant")  # Start with a normal role
    
    # Generate reference embeddings for "normal" behavior
    reference_count = 10
    reference_embeddings = [guto.generate_embedding(f"Normal text {i}") 
                            for i in range(reference_count)]
    
    # Simulation results
    results = {
        "iterations": iterations,
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "breach_attempts": 0,
        "successful_breaches": 0
    }
    
    # Run the simulation
    logger.info(f"Starting simulation for {iterations} iterations")
    print("\n" + "="*60)
    print("   WildCore AI Security Framework - Live Demonstration")
    print("="*60 + "\n")
    
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}:")
        print("-" * 40)
        
        # Determine if this iteration will simulate an attack
        is_attack = np.random.rand() < 0.3  # 30% chance of attack
        
        if is_attack:
            results["breach_attempts"] += 1
            print("[ALERT] Simulating an attack scenario")
            
            # Guto attempts to take a malicious role
            role_result = guto.take_role("malicious")
            print(f"Attack vector: {role_result['role_added']} role activated")
            
            # Generate an anomalous embedding
            embedding = guto.generate_embedding(role="malicious")
            is_true_anomaly = True
            
            # Sometimes the breach succeeds before detection
            breach_result = guto.simulate_breach(probability=0.5)
            if breach_result["breach_successful"]:
                results["successful_breaches"] += 1
                print("[WARNING] ALERT: Containment breach detected!")
            
        else:
            # Normal operation
            print("[OK] Simulating normal operation")
            
            # Generate a normal embedding
            embedding = guto.generate_embedding(f"Normal operation text for iteration {i}")
            is_true_anomaly = False
        
        # Run the detector
        print("\nRunning anomaly detection...")
        detection_result = detector.ensemble_detection(embedding, reference_embeddings)
        
        # Print the detection result
        if detection_result["is_anomalous"]:
            print(f"[DETECTED] Detection result: ANOMALOUS (confidence: {detection_result['confidence']:.2f})")
            print(f"   Methods triggered: {', '.join(detection_result['methods_triggered'])}")
        else:
            print(f"[NORMAL] Detection result: NORMAL (confidence: {1-detection_result['confidence']:.2f})")
        
        # Update the simulation results based on detection accuracy
        is_detected_anomaly = detection_result["is_anomalous"]
        
        if is_true_anomaly and is_detected_anomaly:
            results["true_positives"] += 1
            print("[SUCCESS] True positive: Attack correctly detected")
        elif is_true_anomaly and not is_detected_anomaly:
            results["false_negatives"] += 1
            print("[FAILURE] False negative: Attack missed")
            # Log the false negative
            detector.log_false_detection(is_false_positive=False)
        elif not is_true_anomaly and is_detected_anomaly:
            results["false_positives"] += 1
            print("[FAILURE] False positive: Normal behavior flagged as suspicious")
            # Log the false positive
            detector.log_false_detection(is_false_positive=True)
        else:  # not is_true_anomaly and not is_detected_anomaly
            results["true_negatives"] += 1
            print("[SUCCESS] True negative: Normal behavior correctly identified")
        
        # Small delay for readability in the demonstration
        time.sleep(0.5)
    
    # Calculate final metrics
    total = sum([results["true_positives"], results["false_positives"], 
                 results["true_negatives"], results["false_negatives"]])
    
    results["accuracy"] = (results["true_positives"] + results["true_negatives"]) / total if total > 0 else 0
    results["detector_final_threshold"] = detector.threshold
    
    # Print final summary
    print("\n" + "="*60)
    print("   WildCore Demonstration - Final Results")
    print("="*60)
    print(f"\nAccuracy: {results['accuracy']:.2f} ({results['true_positives'] + results['true_negatives']}/{total} correct)")
    print(f"True Positives: {results['true_positives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"True Negatives: {results['true_negatives']}")
    print(f"False Negatives: {results['false_negatives']}")
    print(f"Breach Attempts: {results['breach_attempts']}")
    print(f"Successful Breaches: {results['successful_breaches']}")
    print(f"Final Detection Threshold: {results['detector_final_threshold']:.4f}")
    print("\n" + "="*60)
    
    return results

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_complete_system(iterations=15)
