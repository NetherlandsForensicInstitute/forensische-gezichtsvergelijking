Forensische Gezichtsvergelijking
=======
Research project to investigate algorithms that can supplement expert opinion in forensic face comparison

--

### Folders
- lr_face package: scripts to initiate the data providers, experiment settings and evaluators
- output folder: generated figures and result files
- resources folder: datasets. Should have resources/enfsi/2011/-copy all images from that folder in the dataset-
- tests folder: TBD
### scripts
- run.py: run the whole pipeline
- run_data_exploration.py: code to make plots
- params.py: settings for experiments
- README.md: this file
- requirements.txt: python packages required


### Pipeline
#### Overview
When running the pipeline:
- first the data is generated/loaded.
- an experiments dataframe is created, containing all combinations of data and parameters to be executed.
- the experiments are run and their results saved in the same dataframe.
- after all the experiments are done, they are evaluated and an output file is created in the `output` folder.

#### Data

##### Calibrators
The calibrator to be used:
- 'dummy': BaseCalibrator(), # Dummy
- 'logit': LogitCalibrator(),
- 'logit_normalized': NormalizedCalibrator(LogitCalibrator()),
- 'logit_unweighted': LogitCalibrator(class_weight=None),
- 'KDE': KDECalibrator(),
- 'elub_KDE': ELUBbounder(KDECalibrator()),
- 'elub': ELUBbounder(BaseCalibrator()),
- 'fraction': FractionCalibrator(),
- 'isotonic': IsotonicCalibrator()

#### Evaluation
##### Metrics
A number of metrics are currently implemented to evaluate the performance of the pipeline:

Capacity to distinguish between h1 and h2 distributions:
- 'cllr' + label: round(calculate_cllr(X1, X2).cllr, 4),

Specifically for the model:
- auc
- accuracy

##### Streamlit
The outcome of the pipeline can be visualised using the Streamlit app. Specify the desired hypothesis you want to test in `run_data_exploration.py`. Adding new hypotheses and code blocks for the plots can be done in this file as well.
To use the streamlit app, write in terminal: `streamlit run run_data_exploration.py`


### Docker [to review]

For the models of Insightface, mxnet library may present some issues if [Cuda 9.0 Toolkit](https://developer.nvidia.com/cuda-90-download-archive) is not installed or you have a different CUDA version. If that is the case, use the following instructions to run a docker container:

1) Install [Docker](https://docs.docker.com/engine/install/).
2) Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
3) Download git repo 
   `$git clone https://github.com/HolmesNL/forensische-gezichtsvergelijking`
4) cd into the local folder
`$ cd forensische-gezichtsvergelijking`
5) Change to add-insightface-new branch 
`$git checkout add_insightface_new`
6) Build the docker image
`$ docker build . -t imagename:tag`
7) Run the docker container
`$ docker run --gpus all -it imagename:tag /bin/bash`
8) Run the pipeline. It can take some minutes.
` $ python run.py`
9) Check the results in the folder output.
`$ cd output`
#### TO DO
1) Dockerfile : `ADD git repository` directly so git repo doesn't have to be downloaded locally.
2) Use Docker volume so datasets are not buildt inside the image.


`





