/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.core;

import fileIO.InFile;
import java.util.ArrayList;
import java.util.Iterator;

/**
 *
 * @author ajb
 */
public class TimeSeriesInstances implements Iterable<TimeSeriesInstance>{
    public static String TIMESERIESDELIMITER=":";

    ArrayList<TimeSeriesInstance> data;
//Probably encapsulate all these    
    boolean univariate=true;
    boolean equalLength=true;
    boolean timeStamps=false;
    boolean missingValues=false;
    boolean classAttribute=false;
    ArrayList<String> classLabels;
    int numClasses=0;
    String name="";
    String dateFormat="";
    int minLength=0;
    int maxLength=0;
    public TimeSeriesInstances(){
        data = new ArrayList<>();
    }
    public void setUnivariate(boolean b){
        univariate=b;
    }
    public boolean isUnivariate(){
        return univariate;
    }
    public void setEqualLength(boolean b){
        equalLength=b;
    }
    public boolean isEqualLength(){
        return equalLength;
    }
    public void setMissingValues(boolean b){
        missingValues=b;
    }
    public boolean hasMissingValues(){
        return missingValues;
    }
    public boolean hasClassAttribute(){
        return classAttribute;
    }
    public void hasClassValues(int nClasses,ArrayList<String> labels){
        classAttribute=true;
        numClasses=nClasses;
        classLabels=labels;
    }
    public void add(TimeSeriesInstance ts){
        data.add(ts);
        if(ts.hasMissing())
            missingValues=true;
        if(!ts.isUnivariate())
            univariate=false;
        int l=ts.getMinLength();
        if(data.size()==1){//First series
            equalLength=ts.isEqualLength();//If all equal internally, then equal
            minLength=ts.getMinLength();
        }
        else if(equalLength){//Need to check same as all the others and internally the same 
            if(!ts.isEqualLength())//If its not equal internally, then they are not all equal!
                equalLength=false;
            else{//Get length 
                if(l!=minLength){
                    equalLength=false;
                }
            }
        }
        if(l<minLength){
            minLength=l;
        }
    }
    public int getMinLength(){
        return minLength;
    }
    public String toString(){
        String str=printHeader()+"\n";
        for(TimeSeriesInstance ts:data)
            str+=ts.toString()+"\n";
            return str;
    }
    public void setName(String s){ name = s;    }
    public String getName(){ return name;    }
    public int numInstances(){ return data.size();}
    public ArrayList<String> getLabels(){return classLabels;}
    public void loadFromFile(String fileName){
//Change to standard file reader
        InFile file=new InFile(fileName);
        ArrayList<String> header= new ArrayList<>();
//Read meta data, inefficiently!
//Read in line by line until @data, ignoring comments
        String line=file.readLine();
        while(line!=null && !line.equals("@data")){
            if(!line.startsWith("#"))
                header.add(line);
            line=file.readLine();
        }
//Process header
        dateFormat="";
        for(String str:header){
            String[] split=str.split("\\s+");
            if(split[0].equals("@timeStamps"))
                timeStamps=Boolean.parseBoolean(split[1]);
            else if(split[0].equals("@dateFormat"))
                dateFormat=split[1];
            else if(split[0].equals("@problemName"))
                name=split[1];
            else if(split[0].equals("@classLabel")){
                classAttribute=Boolean.parseBoolean(split[1]);
 //Read in possible class values here
                if(classAttribute){
                    if(!split[2].equals("double")){//Regression place holder
                        ArrayList<String> cVals=new ArrayList<>();
                        for(int i=2;i<split.length;i++)
                            cVals.add(split[i]);
                        hasClassValues(cVals.size(),cVals);
                    }
                }
            }
            
        }
//Load each line
        String tsCase=file.readLine();
        while(tsCase!=null && !tsCase.isEmpty()){
            TimeSeriesInstance t=loadFromString(tsCase);
            add(t);
            tsCase=file.readLine();
        }
    }
    public TimeSeriesInstance loadFromString(String instance){
        TimeSeriesInstance ts= new TimeSeriesInstance();
        String[] split=instance.split(TIMESERIESDELIMITER);

        if(timeStamps)//Means there should be twice the number 
        {
            System.out.println("NOT IMPLEMENTED CORRECTLY, QUIT (or throw an exception .....)");
            System.exit(0);
        }
        else{
            boolean missing=false;
            int numSeries=split.length;
            if(classAttribute)
                numSeries--;
            for(int i=0;i<numSeries;i++){
                String[] single=split[i].split(",");
                double[] data = new double[single.length];
                for(int j=0;j<data.length;j++){
                    if(single[j].equals("?")){//Missing
                        data[j]=Double.NaN; 
                        missing=true;
                    }
                    else
                        data[j]=Double.parseDouble(single[j]);       
                }
                TimeSeries s=new TimeSeries();
                s.setMissing(missing);
                s.setSeries(data);
                ts.add(s);
            }
            if(classAttribute){
//Could be more efficient but it is a one off operation
//But will switch to a HashMap when I have time
                if(validClassLabel(split[split.length-1])){ 
                    double d=getClassValue(split[split.length-1]);
                    
                    ts.setClassValue(d);                
                }
                else{
                    System.out.println("INVALID CLASS LABEL THROW EXCEPTION");
                    System.exit(0);
                }
            }
        }
        return ts;
    }        
    public boolean validClassLabel(String s){
        return classLabels.contains(s);
    }
    public int getNumSeries(){
        return data.size();
    }

    public int getClassValue(String s){
        for(int i=0;i<classLabels.size();i++)
            if(classLabels.get(i).equals(s))
                return i;
        return -1;//Not present!
        
        
    }
    public String printHeader(){
        String str ="\nProblem ="+name+"\nUnivariate ="+univariate+"\nEqual length ="+equalLength;
        str+="\nTime Stamps ="+timeStamps;
        if(timeStamps)
            str+="  Date Format "+dateFormat;
        str+="\nMissing Values ="+missingValues;
        str+="\nClass Attribute ="+classAttribute;
        if(classAttribute){
            str+=" Class Labels :";
            for(String s:classLabels)
                str+=s+",";
        }        
        str+="\nMin length ="+minLength;
        str+="\nNum Cases ="+data.size();
        return str;        
    }    
    
    @Override
    public Iterator<TimeSeriesInstance> iterator() {
        return data.iterator();
    }
}
