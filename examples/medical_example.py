"""
Medical Diagnosis Example

This example demonstrates the medical diagnosis system using channel algebra:
1. Basic diagnosis system
2. Patient monitoring over time
3. Batch diagnosis and analysis
4. Real-world medical scenarios
5. Alert system and trend analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from channelpy.applications.medical import (
    MedicalDiagnosisSystem, PatientMonitoringSystem, create_sample_patient_data,
    demonstrate_medical_system
)
from channelpy.examples.datasets import make_medical_data
from channelpy import StateArray, EMPTY, DELTA, PHI, PSI


def generate_medical_data():
    """Generate medical data for testing"""
    print("=== Generating Medical Data ===")
    
    # Generate synthetic medical data
    symptoms, tests, labels = make_medical_data(
        n_samples=100, 
        disease_prevalence=0.3,
        test_sensitivity=0.9,
        test_specificity=0.95
    )
    
    print(f"Generated {len(symptoms)} patient records")
    print(f"Disease prevalence: {labels.mean():.2f}")
    print(f"Test sensitivity: {tests.sum() / labels.sum():.2f}")
    
    return symptoms, tests, labels


def demonstrate_basic_diagnosis():
    """Demonstrate basic medical diagnosis"""
    print("\n=== Basic Medical Diagnosis ===")
    
    # Create diagnosis system
    system = MedicalDiagnosisSystem()
    
    # Test with sample patient
    patient_data = create_sample_patient_data()
    print("Sample patient data:")
    for category, data in patient_data.items():
        print(f"  {category}: {data}")
    
    # Diagnose patient
    diagnosis = system.diagnose(patient_data)
    print(f"\nDiagnosis:")
    print(f"  Urgency: {diagnosis['urgency']}")
    print(f"  Recommendation: {diagnosis['recommendation']}")
    print(f"  Confidence: {diagnosis['confidence']:.2f}")
    print(f"  Explanation: {diagnosis['explanation']}")
    print(f"  Channels: {diagnosis['channels']}")
    
    return system, diagnosis


def demonstrate_patient_monitoring():
    """Demonstrate patient monitoring over time"""
    print("\n=== Patient Monitoring ===")
    
    # Create monitoring system
    monitor = PatientMonitoringSystem()
    
    # Add initial patient
    patient_id = "P001"
    initial_data = create_sample_patient_data()
    initial_diagnosis = monitor.add_patient(patient_id, initial_data)
    print(f"Initial diagnosis: {initial_diagnosis['urgency']} - {initial_diagnosis['recommendation']}")
    
    # Simulate patient updates over time
    time_points = [
        # Day 1: Patient improves
        {
            'symptoms': {'fever': 0.3, 'cough': 0.2, 'fatigue': 0.4, 'headache': 0.1},
            'tests': {'white_blood_cells': 0.4, 'c_reactive_protein': 0.3, 'blood_sugar': 0.2},
            'vitals': {'temperature': 37.1, 'heart_rate': 85, 'systolic_bp': 125, 'diastolic_bp': 80, 'oxygen_saturation': 98},
            'history': {'diabetes': 0.0, 'hypertension': 0.0, 'smoking': 0.0, 'age_risk': 0.3}
        },
        # Day 2: Patient deteriorates
        {
            'symptoms': {'fever': 0.9, 'cough': 0.8, 'fatigue': 0.9, 'headache': 0.7},
            'tests': {'white_blood_cells': 0.95, 'c_reactive_protein': 0.9, 'blood_sugar': 0.4},
            'vitals': {'temperature': 39.1, 'heart_rate': 110, 'systolic_bp': 145, 'diastolic_bp': 95, 'oxygen_saturation': 92},
            'history': {'diabetes': 0.0, 'hypertension': 0.0, 'smoking': 0.0, 'age_risk': 0.3}
        },
        # Day 3: Patient stabilizes
        {
            'symptoms': {'fever': 0.4, 'cough': 0.3, 'fatigue': 0.5, 'headache': 0.2},
            'tests': {'white_blood_cells': 0.6, 'c_reactive_protein': 0.5, 'blood_sugar': 0.3},
            'vitals': {'temperature': 37.5, 'heart_rate': 90, 'systolic_bp': 135, 'diastolic_bp': 85, 'oxygen_saturation': 96},
            'history': {'diabetes': 0.0, 'hypertension': 0.0, 'smoking': 0.0, 'age_risk': 0.3}
        }
    ]
    
    # Process updates
    for i, update_data in enumerate(time_points):
        print(f"\nDay {i+1} Update:")
        update = monitor.update_patient(patient_id, update_data)
        
        print(f"  Diagnosis: {update['diagnosis']['urgency']} - {update['diagnosis']['recommendation']}")
        print(f"  Confidence: {update['diagnosis']['confidence']:.2f}")
        
        if update['changes']['urgency_changed']:
            print(f"  ⚠️  Urgency changed!")
        
        if update['changes']['confidence_change'] != 0:
            change = update['changes']['confidence_change']
            print(f"  📈 Confidence change: {change:+.2f}")
        
        if update['alerts']:
            print(f"  🚨 Alerts:")
            for alert in update['alerts']:
                print(f"    {alert['type']}: {alert['message']}")
    
    # Get patient summary
    summary = monitor.get_patient_summary(patient_id)
    print(f"\nPatient Summary:")
    print(f"  Total updates: {summary['total_updates']}")
    print(f"  Current urgency: {summary['current_urgency']}")
    print(f"  Current confidence: {summary['current_confidence']:.2f}")
    print(f"  Urgency trend: {summary['urgency_trend']}")
    print(f"  Confidence trend: {summary['confidence_trend']}")
    
    return monitor


def demonstrate_batch_diagnosis():
    """Demonstrate batch diagnosis of multiple patients"""
    print("\n=== Batch Diagnosis ===")
    
    # Create diagnosis system
    system = MedicalDiagnosisSystem()
    
    # Generate multiple patients
    patients = []
    for i in range(10):
        # Create patient with varying severity
        severity = np.random.uniform(0, 1)
        
        patient_data = {
            'symptoms': {
                'fever': np.random.uniform(0, 1) * severity,
                'cough': np.random.uniform(0, 1) * severity,
                'fatigue': np.random.uniform(0, 1) * severity,
                'headache': np.random.uniform(0, 1) * severity
            },
            'tests': {
                'white_blood_cells': np.random.uniform(0, 1) * severity,
                'c_reactive_protein': np.random.uniform(0, 1) * severity,
                'blood_sugar': np.random.uniform(0, 1)
            },
            'vitals': {
                'temperature': 37.0 + np.random.uniform(0, 2) * severity,
                'heart_rate': 70 + np.random.uniform(0, 30) * severity,
                'systolic_bp': 120 + np.random.uniform(0, 40) * severity,
                'diastolic_bp': 80 + np.random.uniform(0, 20) * severity,
                'oxygen_saturation': 95 + np.random.uniform(0, 5) * (1 - severity)
            },
            'history': {
                'diabetes': np.random.choice([0, 1], p=[0.8, 0.2]),
                'hypertension': np.random.choice([0, 1], p=[0.7, 0.3]),
                'smoking': np.random.choice([0, 1], p=[0.6, 0.4]),
                'age_risk': np.random.uniform(0, 1)
            }
        }
        patients.append(patient_data)
    
    # Batch diagnose
    diagnoses = system.batch_diagnose(patients)
    
    # Get summary
    summary = system.get_diagnosis_summary(diagnoses)
    print("Batch diagnosis summary:")
    print(f"  Total patients: {summary['total_patients']}")
    print(f"  Urgency distribution: {summary['urgency_distribution']}")
    print(f"  Recommendation distribution: {summary['recommendation_distribution']}")
    print(f"  Average confidence: {summary['average_confidence']:.2f}")
    print(f"  Critical cases: {summary['critical_cases']}")
    print(f"  High priority cases: {summary['high_priority_cases']}")
    
    # Show individual diagnoses
    print("\nIndividual diagnoses:")
    for i, diagnosis in enumerate(diagnoses):
        print(f"  Patient {i+1}: {diagnosis['urgency']} - {diagnosis['recommendation']} (confidence: {diagnosis['confidence']:.2f})")
    
    return diagnoses, summary


def demonstrate_medical_scenarios():
    """Demonstrate different medical scenarios"""
    print("\n=== Medical Scenarios ===")
    
    system = MedicalDiagnosisSystem()
    
    # Scenario 1: Healthy patient
    healthy_patient = {
        'symptoms': {'fever': 0.0, 'cough': 0.0, 'fatigue': 0.1, 'headache': 0.0},
        'tests': {'white_blood_cells': 0.2, 'c_reactive_protein': 0.1, 'blood_sugar': 0.3},
        'vitals': {'temperature': 36.8, 'heart_rate': 72, 'systolic_bp': 118, 'diastolic_bp': 78, 'oxygen_saturation': 98},
        'history': {'diabetes': 0.0, 'hypertension': 0.0, 'smoking': 0.0, 'age_risk': 0.2}
    }
    
    # Scenario 2: Critical patient
    critical_patient = {
        'symptoms': {'fever': 0.9, 'cough': 0.8, 'fatigue': 0.9, 'headache': 0.8},
        'tests': {'white_blood_cells': 0.95, 'c_reactive_protein': 0.9, 'blood_sugar': 0.8},
        'vitals': {'temperature': 39.5, 'heart_rate': 125, 'systolic_bp': 160, 'diastolic_bp': 100, 'oxygen_saturation': 88},
        'history': {'diabetes': 1.0, 'hypertension': 1.0, 'smoking': 1.0, 'age_risk': 0.9}
    }
    
    # Scenario 3: Moderate case
    moderate_patient = {
        'symptoms': {'fever': 0.5, 'cough': 0.4, 'fatigue': 0.6, 'headache': 0.3},
        'tests': {'white_blood_cells': 0.6, 'c_reactive_protein': 0.5, 'blood_sugar': 0.4},
        'vitals': {'temperature': 38.0, 'heart_rate': 95, 'systolic_bp': 135, 'diastolic_bp': 85, 'oxygen_saturation': 95},
        'history': {'diabetes': 0.0, 'hypertension': 0.5, 'smoking': 0.0, 'age_risk': 0.6}
    }
    
    scenarios = [
        ("Healthy Patient", healthy_patient),
        ("Critical Patient", critical_patient),
        ("Moderate Case", moderate_patient)
    ]
    
    for scenario_name, patient_data in scenarios:
        print(f"\n{scenario_name}:")
        diagnosis = system.diagnose(patient_data)
        print(f"  Urgency: {diagnosis['urgency']}")
        print(f"  Recommendation: {diagnosis['recommendation']}")
        print(f"  Confidence: {diagnosis['confidence']:.2f}")
        print(f"  Explanation: {diagnosis['explanation']}")


def demonstrate_alert_system():
    """Demonstrate the alert system"""
    print("\n=== Alert System ===")
    
    # Create monitoring system
    monitor = PatientMonitoringSystem()
    
    # Add patient
    patient_id = "P002"
    initial_data = create_sample_patient_data()
    monitor.add_patient(patient_id, initial_data)
    
    # Simulate deterioration
    deterioration_data = {
        'symptoms': {'fever': 0.95, 'cough': 0.9, 'fatigue': 0.95, 'headache': 0.8},
        'tests': {'white_blood_cells': 0.98, 'c_reactive_protein': 0.95, 'blood_sugar': 0.9},
        'vitals': {'temperature': 40.0, 'heart_rate': 130, 'systolic_bp': 170, 'diastolic_bp': 105, 'oxygen_saturation': 85},
        'history': {'diabetes': 1.0, 'hypertension': 1.0, 'smoking': 1.0, 'age_risk': 0.95}
    }
    
    # Update patient
    update = monitor.update_patient(patient_id, deterioration_data)
    
    print("Alert system response:")
    print(f"  Diagnosis: {update['diagnosis']['urgency']} - {update['diagnosis']['recommendation']}")
    print(f"  Confidence change: {update['changes']['confidence_change']:+.2f}")
    
    if update['alerts']:
        print("  🚨 Generated alerts:")
        for alert in update['alerts']:
            print(f"    {alert['type']}: {alert['message']} (Severity: {alert['severity']})")
    else:
        print("  ✅ No alerts generated")


def demonstrate_trend_analysis():
    """Demonstrate trend analysis"""
    print("\n=== Trend Analysis ===")
    
    # Create monitoring system
    monitor = PatientMonitoringSystem()
    
    # Add patient
    patient_id = "P003"
    initial_data = create_sample_patient_data()
    monitor.add_patient(patient_id, initial_data)
    
    # Simulate trend over time
    time_points = [
        # Improving trend
        {'symptoms': {'fever': 0.6, 'cough': 0.5, 'fatigue': 0.6, 'headache': 0.4}},
        {'symptoms': {'fever': 0.4, 'cough': 0.3, 'fatigue': 0.4, 'headache': 0.2}},
        {'symptoms': {'fever': 0.2, 'cough': 0.1, 'fatigue': 0.2, 'headache': 0.1}},
        # Deteriorating trend
        {'symptoms': {'fever': 0.4, 'cough': 0.5, 'fatigue': 0.6, 'headache': 0.4}},
        {'symptoms': {'fever': 0.7, 'cough': 0.8, 'fatigue': 0.8, 'headache': 0.6}},
        {'symptoms': {'fever': 0.9, 'cough': 0.9, 'fatigue': 0.9, 'headache': 0.8}}
    ]
    
    # Process updates
    for i, update_data in enumerate(time_points):
        # Complete patient data
        full_data = create_sample_patient_data()
        full_data.update(update_data)
        
        update = monitor.update_patient(patient_id, full_data)
        print(f"Day {i+1}: {update['diagnosis']['urgency']} (confidence: {update['diagnosis']['confidence']:.2f})")
    
    # Get trend analysis
    summary = monitor.get_patient_summary(patient_id)
    print(f"\nTrend Analysis:")
    print(f"  Urgency trend: {summary['urgency_trend']}")
    print(f"  Confidence trend: {summary['confidence_trend']}")
    print(f"  Total updates: {summary['total_updates']}")


def create_medical_visualization(diagnoses, summary):
    """Create visualization of medical data"""
    print("\n=== Creating Medical Visualization ===")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Urgency distribution
    urgency_counts = summary['urgency_distribution']
    axes[0, 0].bar(urgency_counts.keys(), urgency_counts.values(), color=['green', 'yellow', 'orange', 'red'])
    axes[0, 0].set_title('Urgency Distribution')
    axes[0, 0].set_ylabel('Number of Patients')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Recommendation distribution
    rec_counts = summary['recommendation_distribution']
    axes[0, 1].bar(range(len(rec_counts)), list(rec_counts.values()), color='skyblue')
    axes[0, 1].set_title('Recommendation Distribution')
    axes[0, 1].set_ylabel('Number of Patients')
    axes[0, 1].set_xticks(range(len(rec_counts)))
    axes[0, 1].set_xticklabels(list(rec_counts.keys()), rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confidence distribution
    confidences = [d['confidence'] for d in diagnoses]
    axes[1, 0].hist(confidences, bins=10, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Confidence Distribution')
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Urgency vs Confidence
    urgency_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
    urgencies = [urgency_map.get(d['urgency'], 0) for d in diagnoses]
    axes[1, 1].scatter(urgencies, confidences, alpha=0.6, s=50)
    axes[1, 1].set_title('Urgency vs Confidence')
    axes[1, 1].set_xlabel('Urgency Level')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].set_xticks([1, 2, 3, 4])
    axes[1, 1].set_xticklabels(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('medical_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: medical_analysis.png")


def main():
    """Main medical example function"""
    print("ChannelPy Medical Diagnosis Example")
    print("=" * 50)
    
    # 1. Generate medical data
    symptoms, tests, labels = generate_medical_data()
    
    # 2. Basic diagnosis
    system, diagnosis = demonstrate_basic_diagnosis()
    
    # 3. Patient monitoring
    monitor = demonstrate_patient_monitoring()
    
    # 4. Batch diagnosis
    diagnoses, summary = demonstrate_batch_diagnosis()
    
    # 5. Medical scenarios
    demonstrate_medical_scenarios()
    
    # 6. Alert system
    demonstrate_alert_system()
    
    # 7. Trend analysis
    demonstrate_trend_analysis()
    
    # 8. Visualization
    create_medical_visualization(diagnoses, summary)
    
    print("\n" + "=" * 50)
    print("Medical diagnosis example completed successfully!")
    print("\nGenerated files:")
    print("- medical_analysis.png: Medical data visualization")
    print("\nNext steps:")
    print("- Experiment with different patient scenarios")
    print("- Customize thresholds for your medical domain")
    print("- Integrate with real medical data systems")
    print("- Explore advanced monitoring and alerting features")


if __name__ == "__main__":
    main()







