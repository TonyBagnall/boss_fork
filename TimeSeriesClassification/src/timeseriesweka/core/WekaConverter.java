package timeseriesweka.core;

import weka.core.Instances;

interface WekaConverter {
     Instances convertToWekaFormat(TimeSeriesInstances ts);
}


