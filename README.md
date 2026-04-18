# cmpe257-project-garbage-cnn
CMPE257 (Machine Learning) Project - Garbage Image Classification using CNN

## Datasets
Note: Datasets are not uploaded to this repository, users must download from the posted sources here. 
    * Active Datasets are ones planned for project usage
    * Inactive Datasets are ones that may be used in future, if needed
### Active Datasets
* RealWaste, UC Irvine ML Repository, https://archive.ics.uci.edu/dataset/908/realwaste
### Inactive Datasets
* Waste Classification Dataset, Mendeley Data, https://archive.ics.uci.edu/dataset/908/realwaste


## Development Environment Setup
0. Make sure you use the Python Version in .python-version file 
    - Currently 3.11.7
1. Inside the project's root directory, create virtual environment with: "venv /usr/bin/python3 -m venv venv"
2. Activate virtual environment with: "source venv/bin/activate"
3. Upgrade installer tools with: "python -m pip install --upgrade pip setuptools wheel"
4. Install project libraries with: "python -m pip install -r requiremenets.txt"
5. Verify libraries with "python -c "import torch, torchvision, numpy, matplotlib; print('libaries successfully imported')""
6. Ready to run project scripts from /src