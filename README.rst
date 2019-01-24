Overview
---------

This is copy-paste from MLaut, it will not work out of the box for sktime but if you like the design I can work on integrating it in the sktime workflow. The focus here is on benchmarking of multiple estimators.

The central method here is ``analyze_results.prediction_errors``. It returns two dictionaties with the scores per estimator and the scores per dataset and per estimator. The dictionary with the scores per estimator can be used as input for the other methods which perform statistical tests or visualize the results.

Class Architecture
-------------------

``class AnalyseResults``: main class, implements statistical tests.

``class Losses`` is used by ``AnalyzeResults.prediction_errors`` by composition, mainly for better code readability. ``Losses.evaluate`` calculates the prediction error and builds a dictionary with the results.

``Scores`` are a collection of classes each implementing a different scoring function. An instance of the scoring function ins passes as an argument to the constructor of ``class Losses``.