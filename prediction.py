from unimol_tools.unimol_tools import MolPredict
import numpy as np

pesticide_type_dict = {0: 'Fungicide',
                       1: 'Herbicide', 
                       2: 'Insecticide', 
                       3: 'Cannot be classified but potentially pesticidal activity',
                       4: 'No predicted pesticide activity'}

human_tox_type_dict = {0: 'High toxicity',
                       1: 'Moderate toxicity',
                       2: 'Low toxicity',
                       3: 'Not toxic'}

ecological_tox_type_dict = {0: 'High toxicity',
                            1: 'Moderate toxicity',
                            2: 'Low toxicity',
                            3: 'Not toxic'}

environmental_tox_type_dict = {0: 'High toxicity',
                               1: 'Moderate toxicity',
                               2: 'Low toxicity',
                               3: 'Not toxic'}

def select_model(model_name):
    if model_name == 'pest':
        predictor = MolPredict(load_model='../pest')
    elif model_name == 'human':
        predictor = MolPredict(load_model='../human')
    elif model_name == 'eco':
        predictor = MolPredict(load_model='../eco')
    elif model_name == 'env':
        predictor = MolPredict(load_model='../env')
    return predictor


def prepare_input(smiles_input):
    if isinstance(smiles_input, str):
        # Single SMILES string or multiple comma-separated SMILES strings
        data = [s.strip() for s in smiles_input.split(',')]
    elif isinstance(smiles_input, list):
        # List of SMILES strings
        data = smiles_input
    elif isinstance(smiles_input, np.ndarray):
        # NumPy array of SMILES strings
        data = smiles_input.tolist()
    else:
        raise ValueError("Input must be a SMILES string, a comma-separated list of SMILES strings, a list of SMILES strings, or a NumPy array of SMILES strings")
    
    return data

def map_labels(labels, model_name):
    # Map the labels to their corresponding categories based on the model name
    if model_name == 'pest':
        label_dict = pesticide_type_dict
    elif model_name == 'human':
        label_dict = human_tox_type_dict
    elif model_name == 'eco':
        label_dict = ecological_tox_type_dict
    elif model_name == 'env':
        label_dict = environmental_tox_type_dict
    else:
        raise ValueError("Invalid model name")

    # Map numeric labels to their string representations
    predicted_categories = [label_dict[label] for label in labels]
    return predicted_categories

def main():
    smiles_input = input("Enter SMILES string or list of SMILES strings: ")
    model_name = input("Enter model name (pest, human, eco, env): ")
    predictor = select_model(model_name)
    data = prepare_input(smiles_input)
    predictions = predictor.predict(data=data)
    labels = np.argmax(predictions, axis=1)
    predicted_categories = map_labels(labels, model_name)
    #zip the predicted categories with the smiles input
    zipped_data = list(zip(predicted_categories, data))
    print(zipped_data)

if __name__ == "__main__":
    main()

    #TODO: add a function to save the zipped data to a csv file
    #how to use: give smiles string or list of smiles strings, give model name, get the predicted categories
    #we can change how the input is given.