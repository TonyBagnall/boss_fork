package timeseriesweka.core;

import java.util.Date;

/**
 *
 * @author ajb
 */
public class TimeSeries {
    private double[] series;
    private int[] indices;
    private MetaData md;
    public TimeSeries(){
        series=null;
        indices=null;
        md=new MetaData();
    }
    public void setSeries(double[] d){
        series=d;
        for(double x:d)
            if(x==Double.NaN)
                md.hasMissing=true;
    }
    public double[] getSeries(){ return series;}
    public void setIndex(int[] ind){
        indices=ind;
    }
    public static int parseTimeStamp(String t,String pattern){
        String temp=pattern.toLowerCase();
        if(temp.equals("integer"))
            return Integer.parseInt(t);
        else{
            System.out.println("NOT HANDLING ANYTHING BUT INTEGERS: EXIT");
            System.exit(0);
            return -1;
        } 
    }
    public int length(){ return series.length;}
    public void setMissing(boolean m){ md.hasMissing=m;}

    public boolean hasMissing(){ return md.hasMissing;}
    public class MetaData{
        String name;
        Date startDate;
        double increment;  //Base unit to be .......
        boolean classValue;//really here place for it .....
        boolean hasMissing;
//        boolean 
        public String toString(){
            return "";
        }
    }
    public String toString(){
        String str=md.toString()+"<";
        if(indices!=null){
            for(int i=0;i<series.length-1;i++)
                str+="("+indices[i]+","+series[i]+"),";
            str+="("+indices[series.length-1]+","+series[series.length-1]+")>";
        }
        else{
            for(int i=0;i<series.length-1;i++)
                str+=series[i]+",";
            str+=series[series.length-1]+">";
        }
        return str;
    }
}
