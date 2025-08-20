import streamlit as st


def get_scorer_descriptions() -> dict[str, dict[str, str]]:
    """Get descriptions for different scoring functions."""
    return {
        "Ngram": {
            "title": "N-gram String Analysis",
            "description": "Analyzes molecular diversity using string-based "
            "n-gram patterns from SMILES representations. "
            "Identifies common substrings that may represent "
            "chemical motifs.",
            "use_case": "Useful for identifying string-based patterns and "
            "recurring sequences in molecular representations.",
        },
        "Scaffold": {
            "title": "Scaffold Diversity Analysis",
            "description": "Extracts and analyzes molecular scaffolds using "
            "Murcko framework decomposition. Focuses on the "
            "core ring systems and connecting bonds.",
            "use_case": "Essential for understanding the structural diversity "
            "of core molecular frameworks in your dataset.",
        },
        "Cluster": {
            "title": "Molecular Clustering",
            "description": "Groups molecules based on structural similarity "
            "using molecular fingerprints and Tanimoto "
            "similarity metrics.",
            "use_case": "Identifies clusters of structurally similar "
            "molecules and analyzes cluster diversity.",
        },
        "Original": {
            "title": "Reference Dataset Comparison",
            "description": "Compares generated molecules against a reference "
            "dataset to identify novel vs. known structures.",
            "use_case": "Evaluates how well your generated molecules match "
            "or diverge from known molecular space.",
        },
        "RingScorer": {
            "title": "Ring System Analysis",
            "description": "Focuses specifically on ring systems and cyclic "
            "structures within molecules.",
            "use_case": "Analyzes the diversity of ring systems, which are "
            "crucial for drug-like properties.",
        },
        "FGscorer": {
            "title": "Functional Group Analysis",
            "description": "Identifies and analyzes functional groups "
            "present in the molecular dataset.",
            "use_case": "Evaluates the chemical functionality diversity "
            "through functional group distribution.",
        },
        "Fragments_basic": {
            "title": "Basic Fragment Analysis",
            "description": "Performs fragment analysis using basic wire "
            "frame transformation (atoms → carbon, "
            "preserve bonds).",
            "use_case": "Analyzes structural patterns while focusing on "
            "connectivity rather than atom types.",
        },
        "Fragments_default": {
            "title": "Default Fragment Analysis",
            "description": "Standard fragment analysis without chemical "
            "transformations, preserving original atom types "
            "and bonds.",
            "use_case": "Comprehensive fragment analysis maintaining full "
            "chemical information.",
        },
    }


def create_scoring_info_section() -> None:
    """Create an expandable information section about scoring functions."""
    with st.expander(
        "ℹ️ **About Molecular Diversity Scoring Functions**", expanded=False
    ):
        st.markdown("""
        ### Overview
        This tool provides multiple scoring functions to analyze different
        aspects of molecular diversity:
        """)

        scorer_descriptions = get_scorer_descriptions()

        for scorer_name, info in scorer_descriptions.items():
            st.markdown(f"""
            **{info["title"]}** (`{scorer_name}`)

            {info["description"]}

            *{info["use_case"]}*

            ---
            """)

        st.markdown("""
        ### How to Use
        1. **Load your CSV file** containing SMILES strings and optionally
           'step' and 'Score' columns
        2. **Run t-SNE** to create 2D visualizations of molecular diversity
        3. **Run All Scorers** to perform comprehensive diversity analysis
        4. **Explore results** in the Per Fragment and Per Step tabs

        ### Requirements
        - CSV file must contain a column with SMILES strings
        - For scorer analysis: include 'step' and 'Score' columns
        - For t-SNE: ensure sufficient molecular diversity in your dataset
        """)
