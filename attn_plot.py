import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
from io import BytesIO
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

def get_attention_scores(attn_probs, smiles):
    """
    Get attention scores and visualization data for a molecule.
    Args:
        attn_probs (torch.Tensor): Attention probability matrix from transformer model
        smiles (list): List containing single SMILES string of molecule

    Returns:
        tuple: Contains:
            - mol (rdkit.Chem.rdchem.Mol): RDKit molecule object
            - resized_attn (torch.Tensor): Resized attention matrix excluding special tokens
            - atom_highlights (dict): Dictionary mapping atom indices to RGB highlight colors
            - atom_symbols (list): List of atom symbols with indices
            - num_atoms (int): Total number of atoms in molecule
            - highlight_attentions (dict): Dictionary mapping atom indices to attention scores
    """
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
    norm_attention = (atom_attention - atom_attention.min()) / (atom_attention.max() - atom_attention.min())
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() + str(i) for i in range(num_atoms)]

    # Create sorted list of (atom_idx, attention_score) pairs
    attention_pairs = [(i, score.item()) for i, score in enumerate(norm_attention)]
    attention_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Determine number of atoms to highlight based on molecule size
    if num_atoms > 30:
        num_to_highlight = 10
    elif num_atoms >= 20:
        num_to_highlight = num_atoms // 4
    else:
        num_to_highlight = num_atoms // 2.5
        
    # Create atom highlights for top atoms
    atom_highlights = {}
    highlight_attentions = {}
    for i in range(num_to_highlight):
        atom_idx, score = attention_pairs[i]
        cmap = plt.cm.get_cmap('YlOrRd')
        color = cmap(score)
        rgb_tuple = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        atom_highlights[atom_idx] = rgb_tuple
        highlight_attentions[atom_idx] = score
            
    return mol, resized_attn, atom_highlights, atom_symbols, num_atoms, highlight_attentions

def plot_attention_visualization(mol, resized_attn, atom_highlights, atom_symbols, num_atoms, smiles, predicted_category):
    """
    Plot attention visualization for a molecule with attention highlights and heatmap.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object
        resized_attn (numpy.ndarray): Resized attention matrix excluding first and last tokens
        atom_highlights (dict): Dictionary mapping atom indices to RGB color tuples
        atom_symbols (list): List of atom symbols with indices
        num_atoms (int): Number of atoms in the molecule
        smiles (list): List containing SMILES string of the molecule
        predicted_category (list): List containing predicted category for the molecule

    The function creates a figure with two subplots:
    1. Molecule visualization with highlighted atoms based on attention scores
    2. Attention heatmap showing attention scores between atoms
    """
    print(f"Atoms to highlight: {atom_highlights}")
    
    # Draw molecule with attention highlights
    d = rdMolDraw2D.MolDraw2DCairo(800, 800)
    
    # Set drawing options
    opts = d.drawOptions()
    opts.addStereoAnnotation = True
    opts.addAtomIndices = True
    opts.highlightRadius = 0.5
    opts.highlightBondWidthMultiplier = 1
    
    # Prepare molecule for drawing
    mol = rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=True)
    
    # Convert highlight atoms and colors to the format RDKit expects
    highlight_atoms = list(atom_highlights.keys())
    highlight_colors = {}
    for atom_idx, color in atom_highlights.items():
        highlight_colors[atom_idx] = (color[0]/255, color[1]/255, color[2]/255)
    
    # Draw the molecule
    d.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=[],
    )
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
    plt.subplots_adjust(top=0.85)
    plt.show()