#!/usr/bin/env python
"""
Script to train both random forest and neural network models
and provide a command-line interface to switch between them.
"""

import os
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

from utils.ml_trainer import MLRerouteTrainer


def train_models(training_data_path, force_train=False):
    """
    Train both random forest and neural network models

    Args:
        training_data_path (str): Path to the training data CSV file
        force_train (bool): Force retraining even if models already exist
    """
    model_dir = os.path.join('models', 'resolvers', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    
    rf_model_path = os.path.join(model_dir, 'reroute_classifier.pkl')
    nn_model_path = os.path.join(model_dir, 'reroute_classifier_nn.pkl')
    
    # Check if models already exist
    rf_exists = os.path.exists(rf_model_path)
    nn_exists = os.path.exists(nn_model_path)
    
    if rf_exists and nn_exists and not force_train:
        print("Both models already exist. Use --force to retrain.")
        return
    
    # Train random forest model
    rf_success = False
    if not rf_exists or force_train:
        print(f"\n{'Retraining' if rf_exists else 'Training'} Random Forest model...")
        try:
            trainer = MLRerouteTrainer(
                training_data_path=training_data_path,
                model_type='random_forest'
            )
            rf_model = trainer.train_model(output_path=rf_model_path)
            if rf_model:
                print(f"Random Forest model saved to {rf_model_path}")
                rf_success = True
            else:
                print("Random Forest model training failed.")
        except Exception as e:
            print(f"Error training Random Forest model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Random Forest model already exists at {rf_model_path}")
        rf_success = True
    
    # Train neural network model
    nn_success = False
    if not nn_exists or force_train:
        print(f"\n{'Retraining' if nn_exists else 'Training'} Neural Network model...")
        try:
            trainer = MLRerouteTrainer(
                training_data_path=training_data_path,
                model_type='neural_network'
            )
            
            # Temporarily set the DEFAULT_MODEL_PATH to the NN path to avoid overriding the RF model
            original_default_path = trainer.DEFAULT_MODEL_PATH
            trainer.DEFAULT_MODEL_PATH = nn_model_path
            
            try:
                # Try with standard neural network first
                nn_model = trainer.train_model(output_path=nn_model_path)
                if nn_model:
                    print(f"Neural Network model saved to {nn_model_path}")
                    nn_success = True
                else:
                    print("Neural Network model training failed. Trying with simpler architecture...")
                    
                    # If standard model fails, try with a simpler architecture
                    trainer = MLRerouteTrainer(
                        training_data_path=training_data_path,
                        model_type='neural_network_simple'  # A simpler model defined in ml_trainer.py
                    )
                    trainer.DEFAULT_MODEL_PATH = nn_model_path
                    nn_model = trainer.train_model(output_path=nn_model_path)
                    
                    if nn_model:
                        print(f"Simple Neural Network model saved to {nn_model_path}")
                        nn_success = True
                    else:
                        print("All Neural Network model training attempts failed.")
            except Exception as e:
                print(f"Error training Neural Network model: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Restore original default path
                trainer.DEFAULT_MODEL_PATH = original_default_path
        except Exception as e:
            print(f"Error initializing Neural Network trainer: {e}")
    else:
        print(f"Neural Network model already exists at {nn_model_path}")
        nn_success = True
    
    # Return success status
    if rf_success and nn_success:
        return True
    elif rf_success:
        print("Only Random Forest model training succeeded.")
        return "rf_only"
    elif nn_success:
        print("Only Neural Network model training succeeded.")
        return "nn_only"
    else:
        print("Both model training attempts failed.")
        return False


def switch_model(model_type):
    """
    Configure the application to use the specified model type
    by creating or updating a config file

    Args:
        model_type (str): 'random_forest' or 'neural_network'
    """
    if model_type not in ['random_forest', 'neural_network']:
        print(f"Invalid model type: {model_type}. Must be 'random_forest' or 'neural_network'.")
        return False
    
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / 'ml_model_config.pkl'
    
    # Save configuration
    config = {
        'model_type': model_type
    }
    
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"Application configured to use {model_type} model.")
    return True


def compare_models():
    """
    Compare both model types against the rule-based resolver.
    Runs the resolver_comparison.py script for each model type
    and produces a summary.
    """
    model_dir = os.path.join('models', 'resolvers', 'saved_models')
    rf_model_path = os.path.join(model_dir, 'reroute_classifier.pkl')
    nn_model_path = os.path.join(model_dir, 'reroute_classifier_nn.pkl')
    
    # Check if models exist
    if not os.path.exists(rf_model_path):
        print(f"Random Forest model not found at {rf_model_path}.")
        print("Please train the models first using the 'train' command.")
        return False
    
    if not os.path.exists(nn_model_path):
        print(f"Neural Network model not found at {nn_model_path}.")
        print("Please train the models first using the 'train' command.")
        return False
    
    # Save current model configuration if it exists
    current_model_type = None
    config_file = Path('config') / 'ml_model_config.pkl'
    if config_file.exists():
        try:
            with open(config_file, 'rb') as f:
                config = pickle.load(f)
                if 'model_type' in config:
                    current_model_type = config['model_type']
        except Exception as e:
            print(f"Error reading current model configuration: {e}")
    
    # Run comparison for random forest
    print("\n=== COMPARING RANDOM FOREST MODEL VS RULE-BASED RESOLVER ===\n")
    switch_model('random_forest')
    try:
        result_rf = subprocess.run([sys.executable, 'utils/resolver_comparison.py'], 
                                 capture_output=True, text=True)
        print(result_rf.stdout)
        if result_rf.stderr:
            print("Errors during Random Forest comparison:")
            print(result_rf.stderr)
    except Exception as e:
        print(f"Error running Random Forest comparison: {e}")
    
    # Run comparison for neural network
    print("\n=== COMPARING NEURAL NETWORK MODEL VS RULE-BASED RESOLVER ===\n")
    switch_model('neural_network')
    try:
        result_nn = subprocess.run([sys.executable, 'utils/resolver_comparison.py'], 
                                 capture_output=True, text=True)
        print(result_nn.stdout)
        if result_nn.stderr:
            print("Errors during Neural Network comparison:")
            print(result_nn.stderr)
    except Exception as e:
        print(f"Error running Neural Network comparison: {e}")
    
    # Compare results
    print("\n=== SUMMARY OF MODEL COMPARISONS ===\n")
    print("Both models have been compared against the rule-based resolver.")
    print("Check the output above for detailed performance metrics.")
    
    # Restore original model configuration if it existed
    if current_model_type:
        switch_model(current_model_type)
        print(f"\nRestored original model configuration: {current_model_type}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Train and switch ML models for the disruption resolver')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train both random forest and neural network models')
    train_parser.add_argument('--data', type=str, required=True, help='Path to the training data CSV file')
    train_parser.add_argument('--force', action='store_true', help='Force retraining even if models exist')
    
    # Switch subcommand
    switch_parser = subparsers.add_parser('switch', help='Switch between random forest and neural network models')
    switch_parser.add_argument('model_type', choices=['random_forest', 'neural_network'], 
                              help='Type of model to use')
    
    # Compare subcommand
    compare_parser = subparsers.add_parser('compare', help='Compare both models against the rule-based resolver')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_models(args.data, args.force)
    elif args.command == 'switch':
        switch_model(args.model_type)
    elif args.command == 'compare':
        compare_models()
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 