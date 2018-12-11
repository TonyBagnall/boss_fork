package timeseriesweka.core;

import fileIO.InFile;
import fileIO.OutFile;
import java.util.ArrayList;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class ExampleApplication {
    static void testTruncator() throws Exception{
        String path= "C:\\Users\\ajb\\Dropbox\\Turing Project\\ExampleDataSets\\TestProblems\\";
        String[] probs={"GunPoint","PLAID"};
//        String[] probs={"GunPoint","PLAID","DodgerLoopDay","BasicMotions","JapaneseVowels"};
        for(String str:probs){
            TimeSeriesInstances ts= new TimeSeriesInstances();
            ts.loadFromFile(path+str+"\\"+str+"_TRAIN.csv");
            Truncator trunc=new Truncator();
            System.out.println("Train set info = "+ts.printHeader());
            Instances train=trunc.convertToWekaFormat(ts);
            Classifier c = new J48();
//            Classifier c = new IBk();
            c.buildClassifier(train);
            ts= new TimeSeriesInstances();
            ts.loadFromFile(path+str+"\\"+str+"_TEST.csv");
            System.out.println("Test set info = "+ts.printHeader());
            
            Instances test=trunc.convertToWekaFormat(ts);
            double acc=0;
            for(Instance ins:test){
                double p = c.classifyInstance(ins);
                if(p==ins.classValue())
                    acc++;
            }
            acc/=test.numInstances();
            System.out.println("ACC for "+str+" = "+acc);
        }
    }

    static void testLoadMethods(){
//Load a univariate         
        String path="C:\\Users\\ajb\\Dropbox\\Turing Project\\ExampleDataSets\\";
        TimeSeriesInstances ts= new TimeSeriesInstances();
        ts.loadFromFile(path+"SingleSeriesEqualSpace.txt");
        System.out.println("Single case, single series, no timestamps\n"+ts);
        TimeSeriesInstances ts2= new TimeSeriesInstances();
        ts2.loadFromFile(path+"MultipleSeries.txt");
        System.out.println("Single case, multiple series, no timestamps\n"+ts2);
        TimeSeriesInstances ts3= new TimeSeriesInstances();
        ts3.loadFromFile(path+"SingleSeriesWithIndex2.txt");
        System.out.println("Single case, single series, timestamps\n"+ts3);
        

    }
//Assumes 
    static void convertUnivariateARFF(String source, String dest){
        if(isPadded(source))
          stripOutPadding(source,dest);
        else
           convertUnivariateNoTimeStamps(source,dest);
    }
    
//Converts an arff into internal representation, with padded ? removed
//input format, aRFF header each row: class last, series (scientific), then ? padding
//Output format, each row: series (scientific) unequal length, class     
    static void stripOutPadding(String source, String dest){
        InFile inf= new InFile(source);
        OutFile outf=new OutFile(dest);
        outf.writeLine("#Meta data here. Comments with # for pythonistas");
        outf.writeLine("#Scenario 2: Univariate TSC with series not necessarily the same length");
        outf.writeLine("#but intervals assumed the same, start time assumed irrelevant."); 
        outf.writeLine("#Class values in the same file.");
        outf.writeLine("#This problem is PLAID, from here:");
        String[] name=source.split("\\");
        name=name[name.length-1].split(".");//Not tested
        outf.writeLine("@problemName "+name[0]);
        outf.writeLine("@timeStamps false");
        outf.writeLine("@classLabel true");
        outf.writeLine("@data");
        
        String line=inf.readLine();
//Read header 
        line=line.trim();
        while(line!=null && !line.equals("@data")){
            line=inf.readLine();
            line=line.trim();
        }
        line=inf.readLine();
        while(line!=null){
            String[] split=line.split(",");
            String classV=split[split.length-1];
            int end=split.length-2;
            while(end>0 && split[end].equals("?"))
                end--;
            for(int i=0;i<end-1;i++)
                outf.writeString(split[i]+",");
            outf.writeString(split[end-1]+":");
            outf.writeLine(classV);
            line=inf.readLine();
        }
    }
   
    static void convertUnivariateNoTimeStamps(String source, String dest){
        InFile inf= new InFile(source);
        OutFile outf=new OutFile(dest);
        outf.writeLine("#Meta data here. Comments with # for pythonistas");
        outf.writeLine("#Scenario 1: Univariate TSC with series the same length. May have missing values, but not padded");
        outf.writeLine("#intervals assumed the same, start time assumed irrelevant."); 
        outf.writeLine("Class values in the same file.");
        outf.writeLine("@problemName");
        outf.writeLine("@timeStamps false");
        outf.writeLine("@classLabel true");
        outf.writeLine("@data");
        
        String line=inf.readLine();
//Read header 
        line=line.trim();
        while(line!=null && !line.equals("@data")){
            line=inf.readLine();
            line=line.trim();
        }
        line=inf.readLine();
        while(line!=null){
//Need to strip out the class and insert ":" 
            String[] split= line.split(",");
            for(int i=0;i<split.length-2;i++)
                outf.writeString(split[i]+",");
            outf.writeString(split[split.length-2]+":");
            outf.writeString(split[split.length-1]+"\n");
            line=inf.readLine();
        }
    }
   

    static void convertMultivariateNoTimeStamps(String source, String dest){
        InFile inf= new InFile(source);
        OutFile outf=new OutFile(dest);
        outf.writeLine("#Meta data here. Comments with # for pythonistas");
        outf.writeLine("#Multivariate TSC with series any length. May have missing values, but not padded");
        outf.writeLine("#intervals assumed the same, start time assumed irrelevant."); 
        outf.writeLine("Class values in the same file.");
        outf.writeLine("@problemName");
        outf.writeLine("@timeStamps false");
        outf.writeLine("@multivariate true");
        outf.writeLine("@classLabel true");
        outf.writeLine("@data");
        Instances data=ClassifierTools.loadData(source);
        System.out.println("data"+data);
        for(Instance ins:data){
            Instances series=ins.relationalValue(0);
            for(Instance s:series){
                double[] d=s.toDoubleArray();
                for(int i=0;i<d.length-1;i++)
                    outf.writeString(d[i]+",");
                outf.writeString(d[d.length-1]+":");
            }
            outf.writeLine(ins.classValue()+"");
        }

    }
    public static void processJapaneseVowels(){
        InFile inf=new InFile("C:\\Users\\ajb\\Dropbox\\Turing Project\\ExampleDataSets\\TestProblems\\JapaneseVowels\\ae.train.txt");
        int[] trainCases={30,30,30,30,30,30,30,30,30};
        int tr=0;
        for(int t:trainCases)
            tr+=t;
        for(int i=1;i<trainCases.length;i++)
            trainCases[i]+=trainCases[i-1];
        int[] testCases={31,35,88,44,29,24,40,50,29};
        int te=0;
        for(int t:testCases)
            te+=t;
        for(int i=1;i<testCases.length;i++)
            testCases[i]+=testCases[i-1];
        OutFile outf = new OutFile("C:\\Users\\ajb\\Dropbox\\Turing Project\\ExampleDataSets\\TestProblems\\JapaneseVowels\\JapaneseVowels_TRAIN.csv");        
        outf.writeLine("#Meta data here. Comments with # for pythonistas");
        outf.writeLine("#Scenario 2: Multivariate TSC with series any length. May have missing values, but not padded");
        outf.writeLine("#intervals assumed the same, start time assumed irrelevant."); 
        outf.writeLine("Class values in the same file.");
        outf.writeLine("@problemName");
        outf.writeLine("@timeStamps false");
        outf.writeLine("@multivariate true");
        outf.writeLine("@variableLength true");
        outf.writeLine("@classLabel true");
        outf.writeLine("@data");
        int caseCount=0;
        int currentClass=0;
        while(caseCount<tr){
            ArrayList<String> lines=new ArrayList<>();
            String str=inf.readLine();
            while(!str.equals("")){
                lines.add(str);
                str=inf.readLine();
            }
            double[][] data = new double[12][lines.size()];
            for(int i=0;i<lines.size();i++){
                String[] vals=lines.get(i).split("\\s+");
                for(int j=0;j<vals.length;j++)
                    data[j][i]=Double.parseDouble(vals[j]);
            }
            for(int i=0;i<data.length;i++){
                for(int j=0;j<data[i].length-1;j++)
                    outf.writeString(data[i][j]+",");
                outf.writeString(data[i][data[i].length-1]+":");
            }
//Class value                
            outf.writeLine(currentClass+"");
            caseCount++;
            if(caseCount>trainCases[currentClass])
                currentClass++;
            
        }


        inf=new InFile("C:\\Users\\ajb\\Dropbox\\Turing Project\\ExampleDataSets\\TestProblems\\JapaneseVowels\\ae.test.txt");
        outf = new OutFile("C:\\Users\\ajb\\Dropbox\\Turing Project\\ExampleDataSets\\TestProblems\\JapaneseVowels\\JapaneseVowels_TEST.csv");        
        outf.writeLine("#Meta data here. Comments with # for pythonistas");
        outf.writeLine("#Scenario 2: Multivariate TSC with series any length. May have missing values, but not padded");
        outf.writeLine("#intervals assumed the same, start time assumed irrelevant."); 
        outf.writeLine("Class values in the same file.");
        outf.writeLine("@problemName");
        outf.writeLine("@timeStamps false");
        outf.writeLine("@multivariate true");
        outf.writeLine("@variableLength true");
        outf.writeLine("@classLabel true");
        outf.writeLine("@data");
        caseCount=0;
        currentClass=0;
        while(caseCount<te){
            ArrayList<String> lines=new ArrayList<>();
            String str=inf.readLine();
            while(!str.equals("")){
                lines.add(str);
                str=inf.readLine();
            }
            double[][] data = new double[12][lines.size()];
            for(int i=0;i<lines.size();i++){
                String[] vals=lines.get(i).split("\\s+");
                for(int j=0;j<vals.length;j++)
                    data[j][i]=Double.parseDouble(vals[j]);
            }
            for(int i=0;i<data.length;i++){
                for(int j=0;j<data[i].length-1;j++)
                    outf.writeString(data[i][j]+",");
                outf.writeString(data[i][data[i].length-1]+":");
            }
//Class value                
            outf.writeLine(currentClass+"");
            caseCount++;
            if(caseCount>testCases[currentClass])
                currentClass++;
            
        }
        
    }
/*    
    
    public static class Padder implements Converter{
         Instances convertToWekaFormat(TimeSeriesInstances ts){
             return null;
         }        
    }
    public static class Truncator implements Converter{
     Instances convertToWekaFormat(TimeSeriesInstances ts){
             return null;
     }
    }
*/    
/*If there  is a block of NaN at the end, we assume padding. A NaN anywhere but at the 
  end, assume missing value
*/
    static boolean hasMissing(String source){
        InFile inf= new InFile(source);
        String line=inf.readLine();
//Read header 
        line=line.trim();
        while(line!=null && !line.equals("@data")){
            line=inf.readLine();
            line=line.trim();
        }
        line=inf.readLine();
//        System.out.println(source+" First line ="+line);
        
//Padding registers as ? in arff
        while(line!=null){
            String[] split=line.split(",");
            //Pos split.length-1 is class value, ignore
            int end=split.length-2;
            while(end>0 && split[end].equals("?"))
                end--;
            for(int i=1;i<end;i++){
                if(split[i].equals("?")){
                    return true;
                }
            }
            line=inf.readLine();
        }
        return false;
    }
    static boolean isPadded(String source){
        InFile inf= new InFile(source);
        String line=inf.readLine();
//Read header 
        line=line.trim();
        while(line!=null && !line.equals("@data")){
            line=inf.readLine();
            line=line.trim();
        }
        line=inf.readLine();
        
        while(line!=null){
            String[] split=line.split(",");
            //Pos split.length-1 is class value, ignore
            int end=split.length-2;
            if(split[end].equals("?"))
              return true;
            line=inf.readLine();
        }
        return false;
    }

    
    public static void testLoadAll(String path, String[] probs) {
        TimeSeriesInstances train,test; 
        for(String str:probs){
            train=new TimeSeriesInstances();
            test=new TimeSeriesInstances();
            
            System.out.println("Loading "+path+str+"\\"+str+"_TRAIN.csv");
            train.loadFromFile(path+str+"\\"+str+"_TRAIN.csv");
            System.out.println("TRAIN ="+train.printHeader());
            test.loadFromFile(path+str+"\\"+str+"_TEST.csv");
            System.out.println("TEST ="+test.printHeader());
        }
    }

    public static void main(String[] args) throws Exception {
        testTruncator();
        System.exit(0);
        
        String path= "C:\\Users\\ajb\\Dropbox\\Turing Project\\ExampleDataSets\\TestProblems\\";
        String[] probs={"GunPoint","DodgerLoopDay","PLAID","BasicMotions","JapaneseVowels"};
//        String[] probs={"GunPoint","PLAID","DodgerLoopDay","BasicMotions","JapaneseVowels"};
//        String[] probs={"PLAID","DodgerLoopDay"};
        testLoadAll(path,probs);
        System.exit(0);
        
        for(String str:probs){
            if(isPadded(path+str+"\\"+str+"_TEST.arff"))
                System.out.println("Problem "+str+" Test file is padded");
            else
                System.out.println("Problem "+str+" Test file is NOT padded");
            if(isPadded(path+str+"\\"+str+"_TRAIN.arff"))
                System.out.println("Problem "+str+" Train file is padded");
            else
                System.out.println("Problem "+str+" Train file is NOT padded");
            if(hasMissing(path+str+"\\"+str+"_TEST.arff"))
                System.out.println("Problem "+str+" Test Has Missing");
            else    
                System.out.println("Problem "+str+" Test Has No Missing");
                
            if(hasMissing(path+str+"\\"+str+"_TRAIN.arff"))
                System.out.println("Problem "+str+" Train Has Missing");
            else    
                System.out.println("Problem "+str+" Train Has No Missing");

            convertUnivariateARFF(path+str+"\\"+str+"_TRAIN.arff",path+str+"\\"+str+"_TRAIN.csv");            
            convertUnivariateARFF(path+str+"\\"+str+"_TEST.arff",path+str+"\\"+str+"_TEST.csv");            

//            convertMultivariateNoTimeStamps(path+str+"\\"+str+"_TRAIN.arff",path+str+"\\"+str+"_TRAIN.csv");            
//            convertMultivariateNoTimeStamps(path+str+"\\"+str+"_TEST.arff",path+str+"\\"+str+"_TEST.csv");            
        }
        
    }
}
