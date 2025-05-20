import argparse
import os
import pickle
import traceback
from pathlib import Path

from utils.ml_trainer import MLRerouteTrainer


def train_models(training_data_path, force_train=False):
    model_dir = os.path.join('models', 'resolvers', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)

    rf_model_path = os.path.join(model_dir, 'reroute_classifier.pkl')
    nn_model_path = os.path.join(model_dir, 'reroute_classifier_nn.pkl')

    rf_exists = os.path.exists(rf_model_path)
    nn_exists = os.path.exists(nn_model_path)

    if rf_exists and nn_exists and not force_train:
        print("Both models already exist. Use --force to retrain.")
        return None

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
            traceback.print_exc()
    else:
        print(f"Random Forest model already exists at {rf_model_path}")
        rf_success = True
    nn_success = False
    if not nn_exists or force_train:
        print(f"\n{'Retraining' if nn_exists else 'Training'} Neural Network model...")
        try:
            trainer = MLRerouteTrainer(
                training_data_path=training_data_path,
                model_type='neural_network'
            )
            original_default_path = trainer.DEFAULT_MODEL_PATH
            trainer.DEFAULT_MODEL_PATH = nn_model_path

            try:
                nn_model = trainer.train_model(output_path=nn_model_path)
                if nn_model:
                    print(f"Neural Network model saved to {nn_model_path}")
                    nn_success = True
                else:
                    print(
                        "Neural Network model training failed. Trying with simpler architecture...")

                    trainer = MLRerouteTrainer(
                        training_data_path=training_data_path,
                        model_type='neural_network_simple'
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
                traceback.print_exc()
            finally:
                trainer.DEFAULT_MODEL_PATH = original_default_path
        except Exception as e:
            print(f"Error initializing Neural Network trainer: {e}")
    else:
        print(f"Neural Network model already exists at {nn_model_path}")
        nn_success = True

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
    if model_type not in ['random_forest', 'neural_network']:
        print(f"Invalid model type: {model_type}. Must be 'random_forest' or 'neural_network'.")
        return False

    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / 'ml_model_config.pkl'

    config = {
        'model_type': model_type
    }

    with open(config_file, 'wb') as f:
        pickle.dump(config, f)

    print(f"Application configured to use {model_type} model.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Train and switch ML models for the disruption resolver')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    train_parser = subparsers.add_parser('train',
                                         help='Train both random forest and neural network models')
    train_parser.add_argument('--data', type=str, required=True,
                              help='Path to the training data CSV file')
    train_parser.add_argument('--force', action='store_true',
                              help='Force retraining even if models exist')

    switch_parser = subparsers.add_parser('switch',
                                          help='Switch between random forest and neural network models')
    switch_parser.add_argument('model_type', choices=['random_forest', 'neural_network'],
                               help='Type of model to use')

    args = parser.parse_args()

    if args.command == 'train':
        train_models(args.data, args.force)
    elif args.command == 'switch':
        switch_model(args.model_type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
