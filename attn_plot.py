import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy as np
from io import BytesIO
from rdkit.Chem.Draw import rdMolDraw2D

def plot_attention_on_molecule(attn_probs, smiles, predictions, predicted_category):
    # Calculate average attention
    avg_attn = attn_probs.mean(axis=0)
    
    # Convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles[0])
    
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Get number of atoms
    num_atoms = mol.GetNumAtoms()
    
    # Exclude first and last tokens
    resized_attn = avg_attn[1:-1, 1:-1]
    
    # Take the max value of each row
    atom_attention = resized_attn.max(axis=0)[0]
    
    # Normalize attention scores
    # Set threshold for attention highlighting
    threshold = 0.6
    
    # Normalize attention scores
    norm_attention = (atom_attention - atom_attention.min()) / (atom_attention.max() - atom_attention.min())
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() + str(i) for i in range(num_atoms)]
    print(norm_attention)
    # Create atom highlights based on attention scores
    atom_highlights = {}
    for i, score in enumerate(norm_attention):

        if score.item() > threshold:
            print(score.item(), threshold)
            cmap = plt.cm.get_cmap('YlOrRd')
            color = cmap(score)
            # Convert to tuple format that RDKit expects
            rgb_tuple = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            atom_highlights[i] = rgb_tuple
    print(atom_highlights)
    
    # Draw molecule with attention highlights
    d = rdMolDraw2D.MolDraw2DCairo(800, 800)
    d.drawOptions().addStereoAnnotation = True
    d.drawOptions().addAtomIndices = True
    
    # Draw the molecule with the dictionary of colors
    d.DrawMolecule(mol, highlightAtoms=list(atom_highlights.keys()), 
                   highlightAtomColors=atom_highlights)
    d.FinishDrawing()
    
    # Convert the drawing to an image
    img = Image.open(BytesIO(d.GetDrawingText()))
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Add SMILES and prediction as figure suptitle
    fig.suptitle(f'SMILES: {smiles[0]}\nPredicted Category: {predicted_category[0]}', 
                 fontsize=10, wrap=True)
    
    # Plot the molecule image
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Molecule with Attention Highlights')
    
    # Plot attention heatmap
    sns.heatmap(resized_attn[:num_atoms, :num_atoms], ax=ax2, cmap='YlOrRd',
                xticklabels=atom_symbols, yticklabels=atom_symbols)
    ax2.set_title('Attention Heatmap')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    # Add colorbar
    cbar = ax2.collections[0].colorbar
    cbar.set_label('Attention Score')
    
    # Adjust layout and display
    plt.tight_layout()
    # Adjust the layout to prevent title overlap
    plt.subplots_adjust(top=0.85)
    plt.savefig('attention_map.png')

