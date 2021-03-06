# ML-rapid-text-labeling
Using machine-learning to enable a user to rapidly label a large text corpus.

This repo was originally produce for the final Capstone project for the University of Michigan Masters in Applied Data Science. It contains code to evaluate and analyze the performance of another repo also produced for that project on [rapid text labeling](https://github.com/Tak-Man/ML-rapid-text-labeling-app). The other repo contains a working web app that implements a variety of Data Science techniques to enable a user to rapidly label a text corpus and have control over how they manage the trade-off between accuracy and speed in the labeling process. At the time of writing, the [web app](http://ml-rapid-text-labeling-app.herokuapp.com/) is located on a free hosting service where users can see what all the fuss is about. Initial feedback has been very positive from many users who have seen a demo of the functionality. We got a lot of comments saying it was cool, a great project with lots of real-world applications, and people were super-impressed with the time savings it can make to the labeling process. The code in this repo contains the analysis identifies the time savings the app makes along with earlier analysis that helped us to choose good settings and an appropriate model to use in the app.

The code in this repo includes:
* Automation code to interact with the web app and measure the speed and accuracy of model trained on the labels provided into the app against a test set. This is a way of measuring how effectively the app allows a user to rapidly and accurately label a text corpus within the bounds of the inevitable speed vs accuracy trade-off. This automation code is located in [web-testing](https://github.com/Tak-Man/ML-rapid-text-labeling/tree/main/web-testing) for example in the main [web automation script](https://github.com/Tak-Man/ML-rapid-text-labeling/blob/main/web-testing/web_automate2.py). This folder also contains some analysis of the results of these evaluation experiments on the app, for example in this [automation results evaluation notebook](https://github.com/Tak-Man/ML-rapid-text-labeling/tree/main/web-testing/simulation_eda2.ipynb). The analysis shows that the web app performed well in terms of speed and accuracy. Specifically the experiments conducted showed that the web app resulted in a time-saving of 82% in user labeling. For more details see the [blog](https://michp-ai.github.io/ML-rapid-text-labeling/).
* Analysis of Benchmarks that could be achieved in terms of speed and accuracy outside the app. This was done as an exercise to understand things like, what is the best accuracy that can be achieved using all the training data and labels against the test data? What is the best model in terms of speed and accuracy? What hyperparamaters would work well in the vectorizer in terms of speed and accuracy? All of these type of questions are covered in notebooks and resulting visualizations found in the [baseline classifier](https://github.com/Tak-Man/ML-rapid-text-labeling/tree/main/baseline-classifier) folder. This analysis indicated that the SGDClassifier achieved the best trade-off between speed and accuracy, and, that the initial vectorizer settings in the web app could be changed to enhance the accuracy with very little cost in terms of speed.

## Demo Video
Here is a Demo Video of the app that is what is evaluated in this repo:
<video src="https://user-images.githubusercontent.com/48130648/146453588-6ce8dbb9-14d3-46e9-9dd2-abc9f4b70380.mp4" controls="controls" style="max-width: 730px;">
</video>

## Prerequisites
### 1. Create a Conda environment
This is the project development team's preferred method for setting up the environment required to run this code.
```
$ git clone https://github.com/Tak-Man/ML-rapid-text-labeling.git
```

```
$ conda env create -f environment.yml
```

## Automated Evaluation

Unlike the app which has a single file that is executed to start the app, this repo is a collection of automation tools and notebooks that were used to evaluate the web app.

To run any automation code, the web app also needs to be installed locally and needs to be running before the automation code is started. This will require a different python conda environment. Instructions can be found here:
[here](https://github.com/Tak-Man/ML-rapid-text-labeling-app/blob/main/README.md) on how to install and run the web app.

Some of the important scripts for evaluation can be run with these commands:

```
$ python web_automate2.py
```

and

```
$ python web_automate3_search_exclude_label.py
```

A summary of the main findings from automation is found in the simulation_eda2.ipynb 
[notebook](https://github.com/Tak-Man/ML-rapid-text-labeling/blob/main/web-testing/simulation_eda2.ipynb).

## Main User Benefit
We conducted a lot of analysis and found some interesting and cool stuff but the most important thing we found is that the web app resulted in an 82% time saving to achieve the same accuracy threshold by smart selection of which texts to label next compared to labeling texts in a random order. The red bar shows how long it takes for accuracy to reach a key threshold where labels are added in a random order. The green bars show how quickly that threshold can be reached when the functionality of the app is used to make the labeling process smarter. This is done by using the ML model in the app to indicate which texts it is least confident about and getting the user to label those first.

![time saving at 0.95 threshold using web app difficult texts functionality](https://github.com/Tak-Man/ML-rapid-text-labeling/blob/main/web-testing/viz/time_saving_0.95.png)

This could make a huge real-world difference to a labeling user and shows the value of the web app.

## Benchmark Analysis
This part of our evaluation work was done independently of the web app and can be run without installing the web app.

A summary of the main findings from benchmark analysis is found in the baseline_classifier_accuracy_by_subset.ipynb 
[notebook](https://github.com/Tak-Man/ML-rapid-text-labeling/blob/main/baseline-classifier/baseline_classifier_accuracy_by_subset.ipynb) although viewing this directly in github does not render the painstakingly prepared visualizations.


## Contributors
* [https://github.com/Tak-Man](https://github.com/Tak-Man)
* [https://github.com/michp-ai](https://github.com/michp-ai)

