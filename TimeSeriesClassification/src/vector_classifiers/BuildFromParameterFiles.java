/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package vector_classifiers;

import development.DataSets;
import fileIO.FullAccessOutFile;
import fileIO.OutFile;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import utilities.ClassifierResults;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import vector_classifiers.TunedTwoLayerMLP;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class BuildFromParameterFiles {
    			
    
    

    static Random rng=new Random();
    
    
    public static void buildTunedTwoLayerMLP(String[] args) throws Exception{
        String basePath=args[0];//"\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Results\\UCIContinuous\\TunedTwoLayerMLP\\Predictions\\";
        String problemPath=args[1];//"\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\UCIContinuous\\";
        String problem=args[2];
        int fold =Integer.parseInt(args[3])-1;
        
//        for(String problem:problems){
            Instances all = ClassifierTools.loadData(problemPath+problem+"\\"+problem);
//            for(int i=0;i<folds;i++){
//Check if we need to do this
                rng.setSeed(fold);
                String resultsPath=basePath+problem+"\\";
                File f= new File(resultsPath+"testFold"+fold+".csv");
                System.out.println(resultsPath+"testFold"+fold+".csv");
                if(!f.exists()){
//If so load the problme and do the split
                    Instances[] data = InstanceTools.resampleInstances(all, fold, .5);            
                    TunedTwoLayerMLP mlp2=new TunedTwoLayerMLP();
                    mlp2.setParamSearch(false);
                    mlp2.setSeed(fold);
                    mlp2.setTrainingTime(200);
                    mlp2.setStandardParaSearchSpace(data[0].numAttributes()-1);  
//Find best para set
                    int c=0;
                    ClassifierResults tempResults=null;
                    ClassifierResults bestResults=null;
                    double minErr=1.0;
                    TunedTwoLayerMLP.ResultsHolder best=null;
                    int count=0;
                    long combinedBuildTime=0;
                    for(String p1:mlp2.paraSpace1){
                        for(double p2:mlp2.paraSpace2){
                            for(double p3:mlp2.paraSpace3){
                                for(boolean p4:mlp2.paraSpace4){
                                    count++;
                                    f=new File(basePath+problem+"\\fold"+fold+"_"+count+".csv");
//                                    System.out.print("Para count ="+count);
                                    if((f.exists() && f.length()>0)){//Exists
//                                        System.out.println("\t file exists");
                                        c++;
                                        tempResults = new ClassifierResults();
                                        System.out.println("Loading "+basePath+problem+"\\fold"+fold+"_"+count+".csv");
                                        tempResults.loadFromFile(basePath+problem+"\\fold"+fold+"_"+count+".csv");
                                        combinedBuildTime+=tempResults.buildTime;
                                        double e=1-tempResults.acc;
                                        if(e<minErr){
                                            minErr=e;
                                           best=new TunedTwoLayerMLP.ResultsHolder(p1,p2,p3,p4,tempResults);
                                           bestResults=tempResults;
                                        }
                                    }
//                                    else
//                                        System.out.println(" \t\t file "+basePath+problem+"\\fold"+fold+"_"+count+".csv"+" does not exist");
                                }
                            }            
                        }
                    }
                    System.out.println(problem+ "fold "+fold+" Number of para files present = "+c);
                    if(c>0){

                /* SET UP FINAL MODEL */
                        String bestNumNodes=best.nodes;
                        double bestLearningRate=best.lRate;
                        double bestMomentum=best.mRate;
                        boolean bestDecay=best.decay;
                        mlp2.setHiddenLayers(bestNumNodes);
                        mlp2.setLearningRate(bestLearningRate);
                        mlp2.setMomentum(bestMomentum);
                        mlp2.setDecay(bestDecay);
                        mlp2.res=best.res;
                //Copy over to the train file
                        FullAccessOutFile out= new FullAccessOutFile(resultsPath+"trainFold"+fold+".csv");
                        out.writeLine(data[0].relationName()+","+mlp2.getClass().getName()+",train");
                        if(mlp2 instanceof SaveParameterInfo)
                          out.writeLine(((SaveParameterInfo)mlp2).getParameters());
                        else
                            out.writeLine("No parameter info");
                        out.writeLine(bestResults.acc+"");
                        out.writeString(bestResults.writeInstancePredictions());
                        out.closeFile();
                    System.out.println(problem+ "fold "+fold+" train built");

                        //Build final classifier
                        mlp2.buildClassifier(data[0]);
                        Instances test=data[1];
                        int numInsts = test.numInstances();
                        int pred;
                        ClassifierResults testResults = new ClassifierResults(test.numClasses());
                        double[] trueClassValues = test.attributeToDoubleArray(test.classIndex()); //store class values here

                        for(int testInstIndex = 0; testInstIndex < numInsts; testInstIndex++) {
                            test.instance(testInstIndex).setClassMissing();//and remove from each instance given to the classifier (just to be sure)

                            //make prediction
                            double[] probs=mlp2.distributionForInstance(test.instance(testInstIndex));
                            testResults.storeSingleResult(probs);
                        }
                        testResults.finaliseResults(trueClassValues); 

                        //Write results
                        OutFile testOut=new FullAccessOutFile(resultsPath+"testFold"+fold+".csv");
                        testOut.writeLine(test.relationName()+","+mlp2.getClass().getName()+",test");
                        if(mlp2 instanceof SaveParameterInfo)
                          testOut.writeLine(((SaveParameterInfo)mlp2).getParameters());
                        else
                            testOut.writeLine("No parameter info");
                        testOut.writeLine(testResults.acc+"");
                        testOut.writeString(testResults.writeInstancePredictions());
                        testOut.closeFile();                    
                        System.out.println(problem+ "fold "+fold+" test built");
                    }
                    else
                        throw new Exception("ERROR NO PARAMETER FILES");
                }
                else
                    System.out.println(f.toString()+ "Exists. skipping");
                
//Delete all partial folds                
                
    //Write the test file            
//            }
 //       }    
    }
    static String[] debug={"hill-valley"};
    static String[] probsAJBPC={"adult","magic","chess-krvk","connect-4"};
    static String[] probsArran={"hill-valley","letter","waveform","yeast","conn-bench-vowel-deterding","arrhythmia","cardiotocography-3clases","statlog-shuttle","wall-following","waveform-noise"};
    static String[] probs2398={"molec-biol-splice","musk-1","musk-2","optical","ozone",};
    static String[] probs4419={"conn-bench-vowel-deterding","plant-margin","plant-shape","plant-texture","semeion","spambase","statlog-landsat"};
    static int folds=30;  
    
 static String[] singleLayerMissing={
"adult",
"arrhythmia",
"conn-bench-vowel-deterding",
"cylinder-bands",
"letter",
"musk-2",
"optical",
"ozone",
"plant-margin",
"plant-shape",
"plant-texture",
"semeion",
"spambase",
"steel-plates",
"tic-tac-toe",
"wall-following"};
    
    
    
    public static void buildTunedSingleLayerMLP(String[] args) throws Exception{
        String basePath=args[0];//"\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Results\\UCIContinuous\\TunedTwoLayerMLP\\Predictions\\";
        String problemPath=args[1];//"\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\UCIContinuous\\";
        String problem=args[2];
        int fold =Integer.parseInt(args[3])-1;
        
//        for(String problem:problems){
            Instances all = ClassifierTools.loadData(problemPath+problem+"\\"+problem);
//            for(int i=0;i<folds;i++){
//Check if we need to do this
                rng.setSeed(fold);
                String resultsPath=basePath+problem+"\\";
                File f= new File(resultsPath+"testFold"+fold+".csv");
                System.out.println(resultsPath+"testFold"+fold+".csv");
                if(!f.exists()){
//If so load the problme and do the split
                    Instances[] data = InstanceTools.resampleInstances(all, fold, .5);            
                    TunedTwoLayerMLP mlp2=new TunedTwoLayerMLP();
                    mlp2.setParamSearch(false);
                    mlp2.setSeed(fold);
                    mlp2.setTrainingTime(200);
                    mlp2.setStandardParaSearchSpace(data[0].numAttributes()-1);  
//Find best para set
                    int c=0;
                    ClassifierResults tempResults=null;
                    ClassifierResults bestResults=null;
                    double minErr=1.0;
                    TunedTwoLayerMLP.ResultsHolder best=null;
                    int count=0;
                    long combinedBuildTime=0;
                    for(String p1:mlp2.paraSpace1){
                        for(double p2:mlp2.paraSpace2){
                            for(double p3:mlp2.paraSpace3){
                                for(boolean p4:mlp2.paraSpace4){
                                    count++;
                                    f=new File(basePath+problem+"\\fold"+fold+"_"+count+".csv");
//                                    System.out.print("Para count ="+count);
                                    if((f.exists() && f.length()>0)){//Exists
//                                        System.out.println("\t file exists");
                                        c++;
                                        tempResults = new ClassifierResults();
                                        System.out.println("Loading "+basePath+problem+"\\fold"+fold+"_"+count+".csv");
                                        tempResults.loadFromFile(basePath+problem+"\\fold"+fold+"_"+count+".csv");
                                        combinedBuildTime+=tempResults.buildTime;
                                        double e=1-tempResults.acc;
                                        if(e<minErr){
                                            minErr=e;
                                           best=new TunedTwoLayerMLP.ResultsHolder(p1,p2,p3,p4,tempResults);
                                           bestResults=tempResults;
                                        }
                                    }
//                                    else
//                                        System.out.println(" \t\t file "+basePath+problem+"\\fold"+fold+"_"+count+".csv"+" does not exist");
                                }
                            }            
                        }
                    }
                    System.out.println(problem+ "fold "+fold+" Number of para files present = "+c);
                    if(c>0){

                /* SET UP FINAL MODEL */
                        String bestNumNodes=best.nodes;
                        double bestLearningRate=best.lRate;
                        double bestMomentum=best.mRate;
                        boolean bestDecay=best.decay;
                        mlp2.setHiddenLayers(bestNumNodes);
                        mlp2.setLearningRate(bestLearningRate);
                        mlp2.setMomentum(bestMomentum);
                        mlp2.setDecay(bestDecay);
                        mlp2.res=best.res;
                //Copy over to the train file
                        FullAccessOutFile out= new FullAccessOutFile(resultsPath+"trainFold"+fold+".csv");
                        out.writeLine(data[0].relationName()+","+mlp2.getClass().getName()+",train");
                        if(mlp2 instanceof SaveParameterInfo)
                          out.writeLine(((SaveParameterInfo)mlp2).getParameters());
                        else
                            out.writeLine("No parameter info");
                        out.writeLine(bestResults.acc+"");
                        out.writeString(bestResults.writeInstancePredictions());
                        out.closeFile();
                    System.out.println(problem+ "fold "+fold+" train built");

                        //Build final classifier
                        mlp2.buildClassifier(data[0]);
                        Instances test=data[1];
                        int numInsts = test.numInstances();
                        int pred;
                        ClassifierResults testResults = new ClassifierResults(test.numClasses());
                        double[] trueClassValues = test.attributeToDoubleArray(test.classIndex()); //store class values here

                        for(int testInstIndex = 0; testInstIndex < numInsts; testInstIndex++) {
                            test.instance(testInstIndex).setClassMissing();//and remove from each instance given to the classifier (just to be sure)

                            //make prediction
                            double[] probs=mlp2.distributionForInstance(test.instance(testInstIndex));
                            testResults.storeSingleResult(probs);
                        }
                        testResults.finaliseResults(trueClassValues); 

                        //Write results
                        OutFile testOut=new FullAccessOutFile(resultsPath+"testFold"+fold+".csv");
                        testOut.writeLine(test.relationName()+","+mlp2.getClass().getName()+",test");
                        if(mlp2 instanceof SaveParameterInfo)
                          testOut.writeLine(((SaveParameterInfo)mlp2).getParameters());
                        else
                            testOut.writeLine("No parameter info");
                        testOut.writeLine(testResults.acc+"");
                        testOut.writeString(testResults.writeInstancePredictions());
                        testOut.closeFile();                    
                        System.out.println(problem+ "fold "+fold+" test built");
                    }
                    else
                        throw new Exception("ERROR NO PARAMETER FILES");
                }
                else
                    System.out.println(f.toString()+ "Exists. skipping");
                
//Delete all partial folds                
                
    //Write the test file            
//            }
 //       }    
    }
    
    
    public static void main(String[] args) throws Exception {
        
        String[] str=new String[4];
        str[0]="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Results\\UCIContinuous\\TunedTwoLayerMLP\\Predictions\\";
        str[1]="\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\UCIContinuous\\";
        for(String temp:probsAJBPC){
            str[2]=temp;
            for(int i=1;i<=folds;i++){
                str[3]=i+"";
                buildTunedTwoLayerMLP(str);
            }
        }
        
    }
}
