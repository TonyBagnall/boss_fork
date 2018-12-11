package timeseriesweka.core;

interface TimeSeriesClassifier {
    void buildClassifer(TimeSeriesInstances ins);
    void distributionForInstance(TimeSeriesInstance ins);
}
