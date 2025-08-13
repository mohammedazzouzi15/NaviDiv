Navi_diversity
==============

A package for molecular diversity analysis using fragment and n-gram scoring.

Installation
------------

To install the Navi_diversity package, follow these steps:

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/mohammedazzouzi15/Navi_diversity.git
      cd Navi_diversity

2. **Create and activate a conda environment**:

   .. code-block:: bash

      conda create -n NaviDiv_test python==3.12
      conda activate NaviDiv_test

3. **Choose your installation type**:

   - **Standard Installation (without REINVENT4)**:
   
     This will install the core `navidiv` package and its essential dependencies.

     .. code-block:: bash

        pip install -e .

   - **Full Installation (with REINVENT4)**:
   
     This will install the core package plus all dependencies required to run with REINVENT4.

     First, install PyTorch by selecting your installer, OS, and CPU or CUDA version following the 
     `official PyTorch documentation <https://pytorch.org/get-started/locally/>`_.
     
     For example:

     .. code-block:: bash

        conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia

     Then install REINVENT4 and the full Navi_diversity package:

     .. code-block:: bash

        git clone https://github.com/mohammedazzouzi15/REINVENT4_div.git
        cd REINVENT4_div
        pip install --no-deps -e .
        cd ../
        pip install -e .[reinvent]

4. **Additional setup for REINVENT4** (Optional):

   If you are using REINVENT4, you may need to install the OpenEye Toolkits separately:

   .. code-block:: bash

      conda install openeye::openeye-toolkits

Usage
-----

To run the app with all the features, use the following command in the root directory of the project:

.. code-block:: bash

   streamlit run app.py

This script demonstrates how to score molecules using both fragment and n-gram methods.

Features
--------

- **Fragment-based scoring**: Analyze molecular diversity using fragment decomposition
- **N-gram scoring**: Score molecules based on n-gram patterns
- **REINVENT4 integration**: Compatible with REINVENT4 for molecular generation workflows
- **Flexible installation**: Choose between lightweight core installation or full REINVENT4 integration

Contributing
------------

Contributions are welcome! Please feel free to submit a Pull Request.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

