Forensische Gezichtsvergelijking
=======
Research project to investigate algorithms that can supplement expert opinion in forensic face comparison

---

### Repo structure:
***deepface package:***
fork from the deepface github (https://github.com/serengil/deepface), containing the Facenet, FbDeepFace, OpenFace and VGGFace models

***insightface:***
fork from https://github.com/peteryuX/arcface-tf2 (Arcface) and from https://github.com/shaoanlu/face_toolbox_keras (LResNet, Ir50m1sm, ir50asia)

***lr_face package:*** 
scripts to initiate the data providers, experiment settings and evaluators

***output:*** 
contains generated figures and result files

***root:***
contains runnable scripts

***scratch:***
contains temporary scratch files

***tests:***
contains tests

***weights:***
contains model weights (downloaded and/or finetuned)

---

## Setup 
This repo is using python 3.7 and is not compatible with earlier versions.

Install the requirements:
```bash
pip install -r requirements.txt
```

If you run into problems with connection timeouts or proxies, try the following instead:
```bash
pip install -i https://pypi.org/simple -r requirements.txt
```

---
## Overview
### Finetuning models
##### Setup DGX / Docker

Building the container
```bash
docker build . -f Dockerfile -t <image_name> --build-arg http_proxy=$HTTP_PROXY
```

Pushing the container to the DGX
```bash
docker tag <image_name> <dgx_address>/<image_name>
docker push <dgx_address>/<image_name>
```

To run a model on the DGX, you first need to mount the weights and dataset folder. 
- host/volume = `/opt/data/FaceLR/weights` with path in container = `/root/.deepface/weights`
- host/volume = `/opt/data/FaceLR/resources` with path in container = `/app/resources`
- host/volume = `/opt/data/FaceLR/weights` with path in container = `/app/weights`

##### Finetuning a model
Execute the finetuning from inside the Docker container.

```bash
python3.7 finetuning.py -a <architecture_name> -t <tag_name>
```
If running on the DGX, the scripts expects to 
find the data and weights in `/opt/data/FaceLR/resources` and `/opt/data/FaceLR/weights` respectively. The number of 
epochs is set to 100 by default, but the training can be stopped at any point and the latest weights will be saved.

---
### Running experiments
Run the experiments defined in `params.py`

```bash
python3.7 run.py
```

Running experiments on the DGX is currently not supported. Results are saved in the `output` folder.

#### Experiment settings
In the `params.py` file the following parameters can be set:

***TIMES:***
Number of repetitions

***DATA:***
Dataset(s). Define `current_set_up` from one or more of the options in `all`.

***SCORERS:***
Scorer models. Define `current_set_up` from one or more of the options in `all`.

***CALIBRATORS:***
Calibrator models. Define `current_set_up` from one or more of the options in `all`.

#### Evaluation
##### Metrics
A number of metrics are currently implemented to evaluate the performance of the pipeline:

- cllr: Capacity to distinguish between h1 and h2 distributions
- auc
- accuracy

##### Visualisation
The outcome of the pipeline can be visualised using a Streamlit app. 
Specify the desired hypothesis you want to test in `run_data_exploration.py`. 
Adding new hypotheses and code blocks for the plots can be done in this file as well.

The results from the ENFSI proficiency tests (`Proficiency_test.xlsx`) should be 
added to the resources/enfsi folder.

To use the streamlit app, write in terminal: `streamlit run run_data_exploration.py`

---
### Labeling

For labeling we will use label-studio, which can be installed via pip. In order to set up you labeling project 
use the following commands: 

```
label-studio init annotations/<project-id> \          
    --input-path <path/to/images>  \
    --input-format image-dir \
    --label-config label-studio.xml

label-studio start annotations/<project-id>
```

You can then start labeling at localhost:8200, your labels will be saved in the annotations/<project-id> folder. 
(It might automatically open localhost:8200/start, remove the /start to start labeling in that case). 
