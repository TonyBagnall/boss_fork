/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package development;

import java.io.*;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import timeseriesweka.classifiers.FastDTW_1NN;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.NormalizeCase;

/**
 *
 * @author ajb
 */
public class OneOffCodeNothingToSeeHere {
    public static void robot(){
        Instances robot= ClassifierTools.loadData("C:\\Users\\ajb\\Dropbox\\Temp\\Robot");
        Classifier c=new RotationForest();
//        ((RandomForest)c).setNumTrees(500);
                ((RotationForest)c).setNumIterations(200);
        double a=ClassifierTools.stratifiedCrossValidation(robot, c,10, 0);
        System.out.println("CV accuracy = "+a);
    } 
    
    public static void electricFirstGo() throws Exception{
        Instances robot= ClassifierTools.loadData("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Bags\\Classification Results\\Histogram Classification on Pre-Segmented GT\\BagsTwoClassHistogramProblem\\BagsTwoClassHistogramProblem");
//        Classifier c=new RandomForest();
//       ((RandomForest)c).setNumTrees(500);
        Classifier c=new RotationForest();
       ((RotationForest)c).setNumIterations(200);
        double a=ClassifierTools.stratifiedCrossValidation(robot, c,10, 0);
        System.out.println(c.getClass().getName()+"  CV electric device accuracy not normalised= "+a);
        NormalizeCase nc = new NormalizeCase();
        Instances normed=nc.process(robot);
         a=ClassifierTools.stratifiedCrossValidation(normed, c,10, 0);
        System.out.println(c.getClass().getName()+"  CV electric device accuracy normalised= "+a);
        
        
    } 


    public static void copyFile(File in, File out) 
        throws IOException 
    {
        FileChannel inChannel = new
            FileInputStream(in).getChannel();
        FileChannel outChannel = new
            FileOutputStream(out).getChannel();
        try {
            inChannel.transferTo(0, inChannel.size(),
                    outChannel);
        } 
        catch (IOException e) {
            throw e;
        }
        finally {
            if (inChannel != null) inChannel.close();
            if (outChannel != null) outChannel.close();
        }
    }    
    
    public static void tidyShapelets() throws IOException {
        String sourcePath="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\BalancedClassShapeletTransform\\";
        String destPath="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\Shapelets30Resamples\\";
        for(String str:DataSets.tscProblems85){
            
            File dirIn=new File(sourcePath+str);
            if(dirIn.exists()){
                File dirOut=new File(destPath+str);
                if(!dirOut.exists())
                    dirOut.mkdirs();
                System.out.println("Copying "+str);
                for(int i=0;i<30;i++){
                    File inTrain,inTest;
                if(str.equals("NonInvasiveFetalECGThorax1")||str.equals("NonInvasiveFetalECGThorax2")){
                    inTrain=new File(sourcePath+str+"\\"+str+i+"_TRAIN.arff");
                    inTest=new File(sourcePath+str+"\\"+i+str+"_TEST.arff");
                }
                else if (str.equals("SonyAIBORobotSurface1")){              
                    inTrain=new File(sourcePath+str+"\\SonyAIBORobotSurface"+i+"_TRAIN.arff");
                    inTest=new File(sourcePath+str+"\\SonyAIBORobotSurface"+i+"_TEST.arff");

                }
                else if(str.equals("SonyAIBORobotSurface2")){
                    inTrain=new File(sourcePath+str+"\\SonyAIBORobotSurfaceII"+i+"_TRAIN.arff");
                    inTest=new File(sourcePath+str+"\\SonyAIBORobotSurfaceII"+i+"_TEST.arff");
                    
                }
                else{
                    inTrain=new File(sourcePath+str+"\\"+str+i+"_TRAIN.arff");
                    inTest=new File(sourcePath+str+"\\"+str+i+"_TEST.arff");
                }
                    File outTrain=new File(destPath+str+"\\"+str+i+"_TRAIN.arff");
                    if(!outTrain.exists())
                        copyFile(inTrain,outTrain);
                    File outTest=new File(destPath+str+"\\"+str+i+"_TEST.arff");
                    if(!outTest.exists())
                        copyFile(inTest,outTest);
                }

            }
        }
        
        
    }
    enum Testy{FIRST(10),SECOND(22);
     int v;
     Testy(int x){v=x;}
    
    }
    public static void main(String[] args) throws IOException, Exception {
       // tidyShapelets();
       electricFirstGo();
        System.exit(0);
        robot();
        ArrayList<Testy> in=new ArrayList<>();
        in.add(Testy.FIRST);
        in.add(Testy.SECOND);
        int sum =in.stream().mapToInt(a->a.v).sum();
        System.out.println("sum = "+sum);
        
    }
}
