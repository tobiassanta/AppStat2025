[Back to main page](../README.md)

### Minimal installation: VS code and Miniconda

- 1: Install Miniconda following [these steps](https://www.anaconda.com/docs/getting-started/miniconda/install). The _anaconda.com/download_ website will ask you to create an account if you want to dowload the graphical installer. You can bypass this step by directly dowloading the the latest version for your system [here](https://repo.anaconda.com/miniconda/).


    **Extra step (Mac/Linux)**: the recommended initialization option makes Miniconda start automatically everytime you open a new terminal. This is good for the conda environment to be recognised, but conda will now enter the base environment every time you open a terminal. To stop this, run in a terminal:
    ```bash
    conda config --set auto_activate_base false
    ```


- 2: [Install Git](https://git-scm.com/install/) to easily pull the latest exercises from the AppStat GitHub. The default installation options are good for us.

- 3: Clone the github repository: On the GitHub page, click on the green `<>code` button, select `HTTPS` and copy the https link.
In your machine terminal, go to the directory where you want to save the exercises and dowload the GitHub content:
    ```bash
    git clone <PASTE HTTPS LINK HERE>
    ```

    Later, when new exercises will appear, you can automatically update your local copy by going in this directory and run:
    ```bash
    git pull
    ```
    **Important:** Always make a copy of the documents (like *filename_yourname.ipynb*) that you modify, so your changes don't get overwritten when you pull the new changes.
 
- 4: Create a new conda environment: Download the .yaml file with the pakages list from the [course website](https://www.nbi.dk/~petersen/Teaching/Stat2025/pythonpackages.html). Then from the Anaconda Prompt app (Windows) or your terminal (Mac/Linux), setup your conda environment with:
    ```bash
    conda env create -f /path/to/file/AppStat25_environment.yaml
    ```

- 5: [Install VS code](https://code.visualstudio.com/download)

- 6: Start VS code*, click on the "Extensions" icon on the left and serach for "Jupyter" in the search bar. Select and install the Jupyter extension by Microsoft. Then install the Python extension as well to get access to all the smart features (auto-completion, debugger, etc.).

    *On Windows, start VS code with the `code` command in the Anaconda Prompt app, this make sure VS code will detect the conda environment.


- 7: Open a Jupyter notebook. In the top right corner of a notebook, click on `Select kernel -> Python environment`. Your `appstat` conda environment should appear, select it. You can now run the Jupyter notebooks inside of VS code with your conda environment!

- (Optionnal): At some point, you may want to install new packages in the conda environment. From the Anaconda Prompt app (Windows) or your terminal (Mac/Linux), enter the conda environment and run the standard pip install command:
    ```bash
    conda activate appstat25
    pip install <PACKAGE NAME>
    ```
