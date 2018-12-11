/*
Code to accompany Lecture 8 of Machine learning
 */
package development;

import fileIO.InFile;
import fileIO.OutFile;
import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.text.DecimalFormat;
import utilities.ClassifierResults;
import utilities.ClassifierTools;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.*;
import weka.classifiers.lazy.*;
import weka.classifiers.trees.*;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ajb
 */

public class MachineLearningEvaluationExamples {
    public static String basePath=//"C:\\Users\\ajb\\Dropbox\\UCI Problems\\";
 "\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\ForML\\";    
    
    
    public static void buildAndTestClassifier(Classifier a, String problem, String outFilePath, int fold) throws Exception{
        
        File f= new File(outFilePath+"\\testFold"+fold+".csv");
        if(f.exists()&& f.length()>0)
            return;
        Instances all = ClassifierTools.loadData(basePath+problem);
        Instances[] split=InstanceTools.resampleInstances(all,fold,0.5);
//This is a bespoke class of my own, do it normally if you want!        
        OutFile of=new OutFile(outFilePath+"\\testFold"+fold+".csv");
        a.buildClassifier(split[0]);
        of.writeLine(problem+","+a.getClass().getName());
        of.writeLine("NoParameterInfo");
        double testAcc=0;
        int[] actual=new int[split[1].numInstances()];
        int[] predicted=new int[split[1].numInstances()];
        double[][] probs=new double[split[1].numInstances()][];
        
        for(int i=0;i<split[1].numInstances();i++){
            Instance tr=split[1].instance(i);
            actual[i]=(int)tr.classValue();
            predicted[i]=(int)a.classifyInstance(tr);
            probs[i]=a.distributionForInstance(tr);
            if(actual[i]==predicted[i])//Correct
                testAcc++;
        }
        testAcc/=split[1].numInstances();
        of.writeLine(testAcc+"");
        for(int i=0;i<split[1].numInstances();i++){
            of.writeString(actual[i]+","+predicted[i]+",");
            for(double p:probs[i])
                of.writeString(","+p);
            of.writeString("\n");
        }
    }
    public static String generateStats(String singleFold) throws FileNotFoundException{
        ClassifierResults results=new ClassifierResults();
        results.loadFromFile(singleFold);
        results.findAllStats();
        DecimalFormat df=new DecimalFormat("##.####");
        String r="";
//        r=singleFold+"\n";
        r+="Acc,"+df.format(results.acc)+",BalAcc,"+df.format(results.balancedAcc)+",NLL,"+df.format(results.nll)+",AUROC,"+df.format(results.meanAUROC);
/*        r+="\n";
        double[][]cm=results.confusionMatrix;
        for(int i=0;i<cm.length;i++){
            for(int j=0;j<cm.length;j++)
                r+=(int)cm[j][i]+",";
            r+="\n";
        }
 */       return r;
    }
    public static Classifier setClassifier(String c){
        switch(c){
            case "J48":
                return new J48();
            case"SimpleCart":
                return new SimpleCart();
            case "FT":
                return new FT();
            case "HoeffdingTree":
                return new HoeffdingTree();
            case "LADTree":
                return new LADTree();
            case "NBTree":
                return new NBTree();
            case "REPTree":
                return new REPTree();
            case "DecisionStump":
                return new DecisionStump();
            case "J48graft":
                return new J48graft();
            case "AODE"://Cant handle numeric atts
                return new AODE();
            case "AODEsr"://Cant handle numeric atts
                return new AODEsr();
            case "BayesNet":
                return new BayesNet();
            case "BayesianLogisticRegression"://Cant handle 3+ class problems
                return new BayesianLogisticRegression();
            case "ComplementNaiveBayes": //Numeric attribute values must all be greater or equal to zero
                return new ComplementNaiveBayes();
            case "DMNBtext":
                return new DMNBtext();
            case "HNB": //Cant handle numeric atts
                return new HNB();
            case "NaiveBayes":
                return new NaiveBayes();
            case "NaiveBayesMultinomial": //Numeric attribute values must all be greater or equal to zero
                return new NaiveBayesMultinomial();
            case "NaiveBayesSimple":
                return new NaiveBayesSimple();//attribute f1: standard deviation is 0 for class 0
            case "WAODE"://Cant handle numeric atts
                return new WAODE();
            case "IB1":
                return new IB1();
            case"IBk":
                    return new IBk();
            case "KStar":
                return new KStar();
            case "LBR"://Cant handle numeric atts
                return new LBR();
            case "LWL":
                return new LWL();
//            case "RSC": //java.lang.Exception: No Spheres in the set
//                return new RSC();
            case "kNN":
                return new kNN();
        }
        return null;
        
    }//,"trains"
    static String[] allProblems ={"hayes-roth","abalone","acute-inflammation","acute-nephritis","annealing","arrhythmia","audiology-std","balance-scale","balloons","bank","blood","breast-cancer","breast-cancer-wisc","breast-cancer-wisc-diag","breast-cancer-wisc-prog","breast-tissue","car","cardiotocography-10clases","cardiotocography-3clases","chess-krvkp","congressional-voting","conn-bench-sonar-mines-rocks","conn-bench-vowel-deterding","contrac","credit-approval","cylinder-bands","dermatology","echocardiogram","ecoli","energy-y1","energy-y2","flags","glass","haberman-survival","heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display","lenses","libras","low-res-spect","lung-cancer","lymphography","mammographic","molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","musk-1","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f","oocytes_trisopterus_states_5b","ozone","parkinsons","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D","pittsburg-bridges-TYPE","planning","plant-margin","plant-texture","post-operative","primary-tumor","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit","statlog-german-credit","statlog-heart","statlog-image","statlog-vehicle","steel-plates","synthetic-control","teaching","tic-tac-toe","titanic","vertebral-column-2clases","vertebral-column-3clases","wine","wine-quality-red","wine-quality-white","yeast","zoo"};

//    static String[] allProblems ={"abalone","acute-inflammation","hayes-roth","bank","car","monks-1","breast-cancer"};
//    static String[] treeClassifiers={"J48","SimpleCart","FT","HoeffdingTree","LADTree","REPTree","DecisionStump","J48graft"};
//    static String[] bayesClassifiers={"BayesNet","DMNBtext","NaiveBayes"};
        static String[] allClassifiers={"IB1","IBk","KStar","LWL","kNN"};

    
    public static boolean isPig(String str){
        switch(str){
             case "miniboone":
             case "connect-4":
             case "statlog-shuttle":
             case "adult":
             case "chess-krvk":
             case "letter":
             case "magic":
             case "nursery":
             case "pendigits":
             case "mushroom":
             case "ringnorm":
             case "twonorm":
             case "thyroid":
             case "musk-2":
             case "statlog-landsat":
             case "optical":
             case "page-blocks":
             case "wall-following":
             case "waveform":
             case "waveform-noise":
                return true;
        }
        return false;

    }
    
    
    public static void sortOutDataSets(){
        for(String problem:allProblems){
            if(!isPig(problem)){
                Instances in=ClassifierTools.loadData("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\UCIContinuous\\"+problem+"\\"+problem+".arff");
                if(in.numClasses()==2){
                    OutFile out= new OutFile("\\\\cmptscsvr.cmp.uea.ac.uk\\ueatsc\\Data\\ForML\\"+problem+".arff");
                    out.writeLine(in.toString());
                }
            }
        }
        
        
    }




    public static void generateResultsExample(int folds) throws Exception{
        Classifier c;  
        for(String problem:allProblems){
            System.out.println("Problem ="+problem);
            for(String classifier:allClassifiers){
            System.out.println("\t\t Classifier ="+classifier);
                String results="E:\\UCIResults\\"+classifier+"\\Predictions\\";
                File f=new File(results+problem);
                if(!f.isDirectory())
                    f.mkdirs();
                c=setClassifier(classifier);    
                for(int i=0;i<folds;i++)
                    buildAndTestClassifier(c,problem,results+problem,i); 
            }
        }

        
    }
    public static void createStatsExample() throws Exception{
        String str1=generateStats("E:\\UCIResults\\SimpleCart\\Predictions\\hayes-roth\\testFold0.csv");
        String str2=generateStats("E:\\UCIResults\\J48\\Predictions\\hayes-roth\\testFold0.csv");
        System.out.println(str1+"\n"+str2);
    }   
    public static void collateStatsExample(int folds) throws Exception{
        for(String problem:allProblems){
            for(String classifier:allClassifiers){
                File f= new File("E:\\UCIResults\\"+classifier+"\\Statistics\\");
                if(!f.isDirectory())
                    f.mkdirs();
                OutFile of=new OutFile("E:\\UCIResults\\"+classifier+"\\Statistics\\"+problem+".csv");
                for(int i=0;i<folds;i++){
                    String str1=generateStats("E:\\UCIResults\\"+classifier+"\\Predictions\\"+problem+"\\testFold"+i+".csv");
                    of.writeLine(str1);
                }
            }
        }
    }   
    public static void createSingleResultsFile(int folds){
        OutFile results=new OutFile("E:\\UCIResults\\TreesAcc.csv");
                OutFile of2=new OutFile("E:\\UCIResults\\TreesBalAcc.csv");
                OutFile of3=new OutFile("E:\\UCIResults\\TreesNLL.csv");
                OutFile of4=new OutFile("E:\\UCIResults\\TreesAUROC.csv");
        for(String classifier:allClassifiers)
            results.writeString(","+classifier);
        results.writeString("\n");
        for(String problem:allProblems){
            results.writeString(problem);
            for(String classifier:allClassifiers){//Find average accuracy for this combo
                double acc=0;
                double bacc=0;
                double nll=0;
                double auroc=0;
                InFile f=null;
                try{
                    f=new InFile("E:\\UCIResults\\"+classifier+"\\Statistics\\"+problem+".csv");
                }
                catch(Exception e){
                        System.out.println("E:\\UCIResults\\"+classifier+"\\Statistics\\"+problem+".csv");
                        System.out.println(e.toString());
                        System.exit(0);
                        }
                for(int i=0;i<folds;i++){
                    String[] str=f.readLine().split(",");
                    acc+=Double.valueOf(str[1]);
                    bacc+=Double.valueOf(str[3]);
                    nll+=Double.valueOf(str[5]);
                    auroc+=Double.valueOf(str[7]);
                }
                results.writeString(","+acc/folds);
                of2.writeString(","+bacc/folds);
                of3.writeString(","+nll/folds);
                of4.writeString(","+auroc/folds);
            }
            results.writeString("\n");
            of2.writeString("\n");
            of3.writeString("\n");
            of4.writeString("\n");
        }
        
        
    }
    public static void main(String[] args) throws Exception {
//        sortOutDataSets();
        generateResultsExample(30);
//         collateStatsExample(30);
//        createSingleResultsFile(30);
    }
    
}
