NaviDiv: A Comprehensive Framework for Monitoring Chemical Diversity in Generative Molecular Design
====================================================================================================

**NaviDiv** is a comprehensive framework for analyzing chemical diversity in generative molecular design, with a focus on understanding how different diversity metrics evolve during reinforcement learning optimization. The framework introduces multiple complementary metrics that capture different aspects of molecular variation: representation distance-based, string-based, fragment-based, and scaffold-based approaches.

Features
--------

**Multiple Diversity Metrics**

- **Representation Distance-Based**: Using molecular fingerprints (Morgan, RDKit) and similarity metrics (Tanimoto coefficient)
- **String-Based Analysis**: N-gram analysis of SMILES representations for sequence-level diversity assessment  
- **Fragment-Based Metrics**: Systematic molecular decomposition using BRICS fragmentation and frequency analysis
- **Scaffold-Based Methods**: Bemis-Murcko scaffold analysis for core molecular framework comparison
- **Ring System Analysis**: Identification and analysis of ring systems and their sizes
- **Functional Group Analysis**: Detection and diversity assessment of functional groups

**Real-Time Monitoring & Visualization**

- **Interactive Molecular Visualization**: 2D structural representations with sorting and filtering options
- **Temporal Analysis**: Monitor evolution of specific molecular fragments and cluster formation patterns
- **Chemical Space Projection**: t-SNE and PCA visualization of molecular diversity evolution
- **Comparative Analysis**: Similarity assessment against user-defined reference sets

**Integration Capabilities**

- **REINVENT4 Compatible**: Seamless integration with reinforcement learning workflows
- **Real-Time Penalty Functions**: Adaptive diversity constraints during generation
- **Computational Efficiency**: Minimal overhead (~3 seconds per 100 molecules)
- **Statistical Analysis**: Comprehensive diversity trend reports with significance testing

Installation
------------

To install NaviDiv, follow these steps:

1. **Clone the Repository**:

   .. code-block:: bash

      git clone https://github.com/mohammedazzouzi15/NaviDiv.git
      cd NaviDiv

2. **Create and Activate Conda Environment**:

   .. code-block:: bash

      conda create -n NaviDiv python==3.12
      conda activate NaviDiv

3. **Choose Installation Type**:

   - **Standard Installation (Core Framework)**:
   
     Install the core NaviDiv package with essential dependencies for diversity analysis:

     .. code-block:: bash

        pip install -e .

   - **Full Installation (with REINVENT4 Integration)**:
   
     For complete generative molecular design workflows with REINVENT4:

     First, install PyTorch following the `official documentation <https://pytorch.org/get-started/locally/>`_:

     .. code-block:: bash

        conda install pytorch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 pytorch-cuda=12.4 -c pytorch -c nvidia

     Then install REINVENT4 and NaviDiv with full dependencies:

     .. code-block:: bash

        git clone https://github.com/mohammedazzouzi15/REINVENT4_div.git
        cd REINVENT4_div
        pip install --no-deps -e .
        cd ../
        pip install -e .[reinvent]

4. **Optional Dependencies**:

   For enhanced molecular manipulation capabilities:

   .. code-block:: bash

      conda install openeye::openeye-toolkits

Quick Start
-----------

**Interactive Dashboard**

Launch the Streamlit dashboard for comprehensive diversity analysis:

.. code-block:: bash

   streamlit run app.py

**Programmatic Usage**

.. code-block:: python

   from navidiv.diversity.diversity import diversity_all
   from rdkit import Chem
   
   # Load SMILES strings
   smiles_list = ["CCO", "CCN", "CCC"]  # Your SMILES data
   
   # Calculate various diversity metrics
   richness = diversity_all(smiles=smiles_list, mode="Richness")
   internal_diversity = diversity_all(smiles=smiles_list, mode="IntDiv")
   scaffold_diversity = diversity_all(smiles=smiles_list, mode="BM")
   
   # Analyze functional groups and ring systems
   functional_groups = diversity_all(smiles=smiles_list, mode="FG")
   ring_systems = diversity_all(smiles=smiles_list, mode="RS")

**Integration with REINVENT4**

.. code-block:: python

   from navidiv.reinvent.run_staged_learning_2 import run_staged_learning
   from navidiv.reinvent.InputGenerator import InputGenerator
   from omegaconf import DictConfig
   
   # Create configuration
   cfg = DictConfig({...})  # Your REINVENT config
   
   # Generate input files with diversity filters
   input_generator = InputGenerator(cfg)
   input_generator.generate_input()
   
   # Run staged learning with diversity constraints
   run_staged_learning(cfg)

.. Use Cases
.. ---------

.. **Research Applications**

.. - **Materials Discovery**: Monitor chemical space exploration in organic electronics, catalysis
.. - **Drug Discovery**: Ensure diverse scaffold exploration during lead optimization
.. - **Chemical Space Analysis**: Understand trade-offs between property optimization and diversity

.. **Educational & Industrial**

.. - **Teaching Tool**: Visualize how generative models explore chemical space
.. - **Industrial Pipelines**: Quality control for automated molecular discovery workflows
.. - **Research Validation**: Compare diversity across different generative approaches

Performance
-----------

- **Real-Time Analysis**: <3 seconds per 100 molecules on standard CPU
- **Scalable**: Complete analysis of 10,000 molecules in ~5 minutes
- **Memory Efficient**: Optimized for large-scale molecular datasets
- **Integration Ready**: Minimal computational overhead for existing workflows

Citation
--------

If you use NaviDiv in your research, please cite:

.. code-block:: bibtex

   Comming soon

**Data Availability**: The framework is freely available on GitHub at https://github.com/mohammedazzouzi15/NaviDiv.

.. Documentation
.. -------------

.. - **API Documentation**: Detailed function and class documentation
.. - **Tutorials**: Step-by-step guides for common use cases  
.. - **Case Studies**: Example applications in singlet fission material discovery
.. - **Integration Guides**: REINVENT4 and custom workflow integration

Contributing
------------

We welcome contributions! Please see our contribution guidelines for:

- **Bug Reports**: Issue templates and debugging information
- **Feature Requests**: Enhancement proposals and use case descriptions
- **Code Contributions**: Pull request guidelines and coding standards
- **Documentation**: Help improve examples and tutorials

Development Setup:

.. code-block:: bash

   git clone https://github.com/mohammedazzouzi15/NaviDiv.git
   cd NaviDiv
   pip install -e .[dev]
   pre-commit install

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
---------------

This work was supported by the Swiss National Science Foundation (SNSF).