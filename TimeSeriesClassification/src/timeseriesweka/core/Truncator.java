/*
Functors to convert from timeseriesweka format into standard weka format


*/
package timeseriesweka.core;

import java.util.ArrayList;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class Truncator implements WekaConverter{
    
        @Override
        public Instances convertToWekaFormat(TimeSeriesInstances ts) {
            Instances data=null;
            if(ts.isUnivariate()){
//Set up a standard problem
                return makeUnivariateProblem(ts);
            }
            else{
//Set up a relational attribute for multivariate               
                return makeMultivariateProblem(ts);
            }
        }

    private Instances makeMultivariateProblem(TimeSeriesInstances ts){
        int numAtts=ts.getMinLength();//Going to take the first numAtts attributes
        int numSeries=ts.getNumSeries();
        

        ArrayList<Attribute> relAtts=new ArrayList<>();
        for(int i=1;i<=numAtts;i++)
            relAtts.add(new Attribute("TimeStamp"+i));
        Instances relational = new Instances(ts.name,relAtts,numSeries);
        Attribute att= new Attribute("TimeSeries",relational);
        ArrayList<Attribute> atts=new ArrayList<>();
        atts.add(att);
        Instances multi =new Instances(ts.name,atts,ts.numInstances());
        
        for(TimeSeriesInstance t:ts){
            Instances rel= new Instances(relational);
            for(int i=0;i<numSeries;i++){
                double[] d=t.getSeries(i); // Returns the first (and only) time serieas
                Instance inst=new DenseInstance(numAtts);
                for(int j=0;j<numAtts;j++){
                    if(d[j]==Double.NaN)
                        inst.setMissing(j);
                    else
                        inst.setValue(j, d[j]);
                }
                rel.add(inst);

            }
            Instance ins=new DenseInstance(2);
            ins.setDataset(multi);
            Attribute a=ins.attribute(0);
            a.addRelation(rel);
            double d=t.getClassValue();
            ins.setValue(1, d);
        }        
        return null;
    }        
//TO DO: SORT OUT CLASS VALUES        
    private Instances makeUnivariateProblem(TimeSeriesInstances ts){
        int numAtts=ts.getMinLength();//Going to take the first numAtts attributes
        ArrayList<Attribute> atts=new ArrayList<>();
        for(int i=1;i<=numAtts;i++)
            atts.add(new Attribute("TimeStep"+i));
        //Add class values
	if(ts.hasClassAttribute()){	//Classification set, set class 
			//Get the class values a list of strings			
            ArrayList<String> vals=ts.getLabels();
            atts.add(new Attribute("ClassLabel",vals));
        }
        Instances result = new Instances(ts.name,atts,ts.numInstances());
        result.setClassIndex(atts.size());
        for(TimeSeriesInstance t:ts){
            double[] d=t.getSeries(0); // Returns the first (and only) time serieas
             Instance inst=new DenseInstance(numAtts+1);
            for(int i=0;i<numAtts;i++){
                if(d[i]==Double.NaN)
                    inst.setMissing(i);
                else
                    inst.setValue(i, d[i]);
            }
            double cls=t.getClassValue();
            inst.setValue(numAtts,cls);
            result.add(inst);
//Get class value            
        }
//Set class attribute        
        return result;
    } 
 }
