# cmpe257-project-garbage-cnn
CMPE257 (Machine Learning) Project - Garbage Image Classification using CNN

## Datasets
Note: Datasets are not uploaded to this repository, users must download from the posted sources here. 
* Active Datasets are ones planned for project usage
* Inactive Datasets are ones that may be used in future, if needed
### Active Datasets
* RealWaste, UC Irvine ML Repository, https://archive.ics.uci.edu/dataset/908/realwaste
    * Note: Rename Labels When Downloading Raw Data From Source
        * Food Organics -> Organics
        * Miscellaneous Trash -> Miscellaneous
        * Textile Trash -> Clothing
### Inactive Datasets
* Waste Classification Dataset, Mendeley Data, https://archive.ics.uci.edu/dataset/908/realwaste

## Using Datasets
* Make directory: [root]/data/raw
* Extract the dataset image folder into the new directory


## Development Environment Setup
0. Make sure you use the Python Version in .python-version file 
    - Currently 3.11.7
1. Inside the project's root directory, create virtual environment with: /usr/bin/python3 -m venv venv
2. Activate virtual environment with: source venv/bin/activate
3. Upgrade installer tools with: python -m pip install --upgrade pip setuptools wheel
4. Install project libraries with: python -m pip install -r requirements.txt
5. Verify libraries with: python -c 'import torch, torchvision, numpy, matplotlib, sklearn; print("libraries successfully imported")'
    - Note: running this line might take several seconds
6. Ready to run project scripts from /src

## Training with GPU
* Make sure you have cuda-compatible GPU (e.g. Nvidia)
* Can test with: python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'No GPU'"
* If you see "False No GPU", you need to reinstall torch + torchvision with cuda-compatibility: 
    * First Uninstall: python -m pip uninstall torch torchvision -y
    * Then Install from URL: python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121