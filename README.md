[![Code style:black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

[![DeepSource](https://static.deepsource.io/deepsource-badge-light.svg)](https://deepsource.io/gh/ternaus/iglovikov_helper_functions/?ref=repository-badge)

# Modeling status of all traffic lights at an intersection based on sequencing of lanes
## The Goal
Self-driving vehicles(SDV) cannot observe states of all traffic lights at an intersection. Traffic lights seem to be relevant for a good understanding of the overall situation at the intersection.

At the same time, the vehicles observed by the SDV suggest statuses of different traffic lights at the intersection.

In this pet project, I would like to try estimating the statuses of all traffic lights based on a limited amount of SDV observations.

## Data
[Prediction dataset fromLyft level 5 self-driving system](https://self-driving.lyft.com/level5/data/).

## The idea overview

### Inputs
The sequence of observed traffic lights and lanes form an input for an RNN:

![](input/input_lane_seq.gif)

### Targets
Based on the limited observations we can heuristically guess some traffic light statuses and use it as labels. 

For the illustration please consider the GIF below. 

    * First, we observe red traffic light, so we know its status forsure.

    * Next, we observe idle vehicles, so we derive that the correspondingtraffic light to be red.
    * Later, we observe vehicles exiting lanes controlled by a traffic light, therefore we derive that thetraffic light has a green color, etc.
![](input/heuristic_labels.gif)

### Model
A bidirectional LSTM model. 

To provide an analogy with NLP applications for a better intuitive understanding of the underlying model, each event type can be viewed as a separate word, and green light can be viewed as a positive sentiment of a document. The main difference from LSTM-based classifiers in NLP is the fact that events are not evenly sampled in time, so we also use the time of each event. 

In addition to traffic light status, I attempted to model Weibull distribution of the remaining time before the next color change in accordance with [this work](http://publications.lib.chalmers.se/records/fulltext/253611/253611.pdf),but the initial attempt did not seem to give reasonable results: the mode of the remaining time decreases just slightly before the actual change. Likely,I've caused in by a very imbalanced dataset, as I clipped the remaining time at5 seconds, motivated by the fact that the Lyft level 5 challenge required prediction for the next 5 seconds. Sparse and noise heuristic labels might have complicated the distribution modeling as well.

![](input/pred.gif)

## Reproducibility
All the used packages can be installed via standard package managers. The exact package versions I used can be found in the file *lyft_env.yml*.

After the data are downloaded and stored into *input/scenes*, running the bash script ***run_all.sh*** should prepare the data, train, score the models, and create the visualizations shown above.  