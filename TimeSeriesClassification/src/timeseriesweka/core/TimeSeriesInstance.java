/*
TimeSeriesInstance, to handle variable length series, missing value 
*/
package timeseriesweka.core;

import java.util.*;

/**
 *
 * @author ajb
 */
public class TimeSeriesInstance {
    private ArrayList<TimeSeries> ts;
    private double classLabel;
    private int minLength=0;
    private boolean equalLength=true;
    private boolean univariate=true;
    private boolean hasMissing=false;
    public TimeSeriesInstance(){
        ts=new  ArrayList<>();
    }
    public void add(TimeSeries t){
        ts.add(t);
        if(t.hasMissing())
            hasMissing=true;
        if(ts.size()==1){
            minLength=t.length();
            univariate=true;
            equalLength=true;
        }
        else{
            if(t.length()<minLength){
                minLength=t.length();
                equalLength=false;
            }
            univariate=false;//reset repeatedly, but hardly expensive
        }
    }
    public void setClassValue(double d){
        classLabel=d;
    }
    public double getClassValue(){
        return classLabel;
    }
    public double[] getSeries(int index){
        if(index>ts.size()) return null;//throw an exception
        return ts.get(index).getSeries();
    }
    public String toString(){
        String str="";
        for(TimeSeries t:ts)
            str+=t;
        if(classLabel!=Double.NaN)
            str+=":"+classLabel;
        return str;
    }
    public int numSeries(){ return ts.size();}
    public boolean isEqualLength(){ return equalLength;}
    public boolean isUnivariate(){ return univariate;}
    public boolean hasMissing(){ return hasMissing;}
    
    public boolean checkEqualLength(){ 
        int l=ts.get(0).length();
        for(int i=1;i<ts.size();i++){
            if(l!=ts.get(i).length())
                return false;
        }
        return true;
    }
    int getMinLength(){
        return minLength;
    }

}
