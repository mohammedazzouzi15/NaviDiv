import io

import streamlit as st
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D


def draw_molecule(mol):
    """Draw a molecule using RDKit and return the image."""
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        return img
    except Exception:
        st.error("Error drawing molecule")
        return None
