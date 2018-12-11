/*
ContractRotationForestVersion1.

Contractable and Checkpointable version of RotationForest

Set to guarantee at least 50 trees
1. Estimate time for single tree (from model)
2. If fewer than 50 trees possible, estimate maximum number of attributes allowed, x.
3. Build each tree with between x/2 and x attributes
4. Once 50 trees built, continue building with random number of attributes  between x/2 and max number by model for time remaining
any comments? I am going to test on the concatenated MTSC problems
*/
package vector_classifiers;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.CheckpointClassifier;
import timeseriesweka.classifiers.ContractClassifier;
import timeseriesweka.classifiers.ParameterSplittable;
import utilities.ClassifierResults;
import utilities.ClassifierTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import weka.classifiers.meta.RotationForest;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ajb
 * Handling a variable number of trees is awkward, because it is all done with arrays
 * 
 *   /** The attributes of each group 
  protected int [][][] m_Groups = null;

  /** The type of projection filter 
  protected Filter m_ProjectionFilter = null;

  /** The projection filters 
  protected Filter [][] m_ProjectionFilters = null;

  /** Headers of the transformed dataset 
  protected Instances [] m_Headers = null;

  /** Headers of the reduced datasets 
  protected Instances [][] m_ReducedHeaders = null;

 * 
 * Going to have
 */
public class ContractRotationForestVersion1 extends RotationForest  implements SaveParameterInfo, ContractClassifier, CheckpointClassifier, Serializable{
    double contractHours=0.04;    //Defaults to an approximate build time of 1 hour
    protected ClassifierResults res;
    double estSingleTree;
    int minNumTrees=50;
    int maxNumTrees=200;
    int maxNumAttributes;
    String checkpointPath;
    boolean debug=true;
    TimingModel tm;
    double timeUsed;
    ArrayList<RotationForest> classifiers;
    public ContractRotationForestVersion1(){
        super();
        tm=new TimingModel();
        res=new ClassifierResults();
        classifiers=new ArrayList<>();
        checkpointPath=null;
        timeUsed=0;
        
    }
    @Override
    public void buildClassifier(Instances train) throws Exception{
//Initialisation        
        long startTime=System.currentTimeMillis(); 
//

//1. Look for a checkpointed version        
        int n=train.numInstances();
        int m=train.numAttributes()-1;
        timeUsed=0;
        if(checkpointPath!=null){
            //Look for previous model. Protocol for saving is ProblemName+ClassifierName.ser
            String serPath=checkpointPath+"/"+train.relationName()+this.getClass().getSimpleName()+".ser";
            File f =new File(serPath);
            if(f.exists()){
                if(debug)
                    System.out.println("Serialised version exists, trying to load from "+serPath);
                loadFromFile(serPath);
           }
            else
                if(debug)
                    System.out.println("No Serialised versionat location "+serPath);
                
        } 
//Re-estimate even if loading serialised, may be different hardware ....        
        estSingleTree=tm.estimateSingleTreeHours(n,m);
        if(debug)
            System.out.println("n ="+n+" m = "+m+" estSingleTree = "+estSingleTree);
        
        if(estSingleTree*50<contractHours){ //Can do at least 50 trees we think, just build them in batches
            if(debug)
                System.out.println("Able to build at least 50 trees");
            maxNumAttributes=m;
            int numTrees=0;
            int batchSize=setBatchSize(estSingleTree);    //Set larger for smaller data
            if(debug)
                System.out.println("Batch size = "+batchSize);
            long startBuild=System.currentTimeMillis(); 
            while(timeUsed<contractHours && numTrees<maxNumTrees){
//Build a fraction more                 
                buildTrees(train,batchSize);
                numTrees+=batchSize;
                long newTime=System.currentTimeMillis(); 
                timeUsed=(newTime-startBuild)/(1000.0*60.0*60.0);
//CHECK POINT HERE   
                if(debug)
                    System.out.println("Time used = "+timeUsed+" number of batches ="+classifiers.size()+" number of trees ="+getNumTrees());
                if(checkpointPath!=null){
                    //save the serialised version
                    try{
                        File f=new File(checkpointPath);
                        if(!f.isDirectory())
                            f.mkdirs();
                        saveToFile(checkpointPath+"/"+train.relationName()+this.getClass().getSimpleName()+".ser");
                        if(debug)
                            System.out.println("Saved to "+checkpointPath+"/"+train.relationName()+this.getClass().getSimpleName()+".ser");
                    }
                    catch(Exception e){
                        System.out.println("Serialisation to "+checkpointPath+"/"+train.relationName()+this.getClass().getSimpleName()+".ser  FAILED");
                    }
                } 
                
            }           
        }
        else{ //Estimate the size 
//Heuristic to reduce either n or m
            if(debug)
                System.out.println("unable to build 50 trees in the time allowed ");
//If m > n: Only do this for now

            if(m>n){
//estimate maximum number of attributes allowed, x, to get minNumberOfTrees.                
                int maxAtts=tm.estimateMaxAttributes(n,m);
            if(debug)
                System.out.println("using "+maxAtts+" attributes, building single tree at a time");
               double timeUsed=0;
                int numTrees=0;
                long startBuild=System.currentTimeMillis(); 
                while(timeUsed<contractHours*(1000.0*60.0*60.0) && numTrees<minNumTrees){
                //Build a fraction more                 
                    buildSingleLimitedAttributeTree(train,maxAtts/2,maxAtts);
                    numTrees++;
                    long newTime=System.currentTimeMillis(); 
                    timeUsed=(newTime-startBuild)/(1000.0*60.0*60.0);
                //CHECK POINT HERE               
                    if(debug)
                        System.out.println("Built tree number "+numTrees+" in "+timeUsed+" hours ");
                    if(checkpointPath!=null){//Look for previous model
                        //save the serialised version
                        try{
                            File f=new File(checkpointPath);
                            if(!f.isDirectory())
                                f.mkdirs();
                            saveToFile(checkpointPath+"/"+train.relationName()+this.getClass().getSimpleName()+".ser");
                            if(debug)
                                System.out.println("Saved to "+checkpointPath+"/"+train.relationName()+this.getClass().getSimpleName()+".ser");
                        }
                        catch(Exception e){
                            System.out.println("Serialisation to "+checkpointPath+"/"+train.relationName()+this.getClass().getSimpleName()+".ser  FAILED with exception "+e);
                        }
                    }                     
                }
                if(debug){
                    System.out.println("Completed the minimum "+getNumTrees()+" trees in "+timeUsed+" time");
                    System.out.println("Still have "+(contractHours-timeUsed)+" time remaining ");
                    
                }
                while(timeUsed<contractHours&& numTrees<maxNumTrees){
                    buildSingleLimitedAttributeTree(train,maxAtts/2,m);
                    numTrees++;
                    long newTime=System.currentTimeMillis(); 
                    timeUsed=(newTime-startBuild)/(1000.0*60.0*60.0);
                //CHECK POINT HERE               
                    if(debug)
                        System.out.println("Built tree number "+numTrees+" in "+timeUsed+" hours ");
                }
            if(debug)
                System.out.println("Completed the alotted time with "+getNumTrees()+" trees in "+timeUsed+" time");
            }
            else{//Subsample cases NOT IMPLEMENTED
                 throw new UnsupportedOperationException("CASE n<m Not supported yet."); //To change body of generated methods, choose Tools | Templates.               
            }
        }

//Port the trees back into the RotationForest array, so we dont have to reimplement buildClassifier        
        
        res.buildTime=System.currentTimeMillis()-startTime;
    }
//Set to checkpoint approximately every 2 hours         

    private int setBatchSize(double singleTreeHours){
        if(singleTreeHours>ContractClassifier.CHECKPOINTINTERVAL)
            return 1;
        int hrs=(int)(ContractClassifier.CHECKPOINTINTERVAL/singleTreeHours);
        return hrs;
        
    }
    
    @Override
    public int getNumIterations(){
        int numTrees=0;
        for(RotationForest rf:classifiers){
            numTrees+=rf.getNumIterations();
        }
        return numTrees;       
    }
    

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
//Weight each member by the number of trees
        int numTrees=0;
        double[] dist=new double[instance.numClasses()];
        for(RotationForest rf:classifiers){
            double[] d=rf.distributionForInstance(instance);
            int t=rf.getNumIterations();
            for(int i=0;i<d.length;i++)
                dist[i]+=t*d[i];
            numTrees+=t;
        }
        for(int i=0;i<dist.length;i++)
            dist[i]/=numTrees;
        return dist;
    }
    
    int getNumTrees(){
        int numTrees=0;
        for(RotationForest rf:classifiers){
            int t=rf.getNumIterations();
            numTrees+=t;
        }
    return numTrees;
        
    }
    
    void buildTrees(Instances train,int numTrees) throws Exception{
//Incrementally build more trees.        


//Simplest way to do it ..., use all the attributes
        RotationForest rf = new RotationForest();
        copyParameters(rf);
        rf.setNumIterations(numTrees);
        rf.buildClassifier(train);
        classifiers.add(rf);
        
    }
/**
 * This method constructs a single tree RotationForest with a subset of the attributes
 * It creates a RotationForest for the single tree, using the overridden attribute selection method. 
 * Really, we only need a single tree, but it is much easier to implement it like this for the research
 * grade implementation
  */ 
    void buildSingleLimitedAttributeTree(Instances train, int minAtts, int maxAtts) throws Exception{
        RotationForestLimitedAttributes rf = new RotationForestLimitedAttributes();
        copyParameters(rf);
        Random r = new Random();
        int s=r.nextInt(maxAtts-minAtts);
        rf.setMaxNumAttributes(minAtts+s);
        rf.setNumIterations(1);
        rf.buildClassifier(train);
        classifiers.add(rf);
        
    }    
    
    private void copyParameters(RotationForest rf){
        rf.setMaxGroup(m_MaxGroup);
        rf.setMinGroup(m_MinGroup);
        rf.setNumberOfGroups(m_NumberOfGroups);
        rf.setRemovedPercentage(m_RemovedPercentage);
        
    }
   
    
    
    @Override
    public String getParameters() {
        String result="BuildTime,"+res.buildTime+",CVAcc,"+res.acc+",RemovePercent,"+this.getRemovedPercentage()+",NumFeatures,"+this.getMaxGroup();
        int numTrees=getNumTrees();
        result+=",numTrees,"+numTrees;
        return result;
    }

    @Override
    public void setTimeLimit(long time) {
        contractHours=((double)time)/(1000.0*60.0*60);
    }

    @Override
    public void setTimeLimit(TimeLimit time, int amount) {
        switch(time){
            case MINUTE:
                contractHours=((double)amount)/60.0;
                break;
            case DAY:
                contractHours=((double)amount*24.0);
                break;
            case HOUR: default:
                contractHours=((double)amount);
                break;
        }
    }

    @Override
    public void setSavePath(String path) {
        checkpointPath=path;
    }

    @Override
    public void copyFromSerObject(Object obj) throws Exception {
        if(!(obj instanceof ContractRotationForestVersion1))
            throw new Exception("The SER file is not am instance of ContractRotationForest"); //To change body of generated methods, choose Tools | Templates.
        ContractRotationForestVersion1 saved= ((ContractRotationForestVersion1)obj);
        this.classifiers=saved.classifiers;
        this.contractHours=saved.contractHours;
        res=saved.res;
        minNumTrees=saved.minNumTrees;
        maxNumTrees=saved.maxNumTrees;
        maxNumAttributes=saved.maxNumAttributes;
        checkpointPath=saved.checkpointPath;
        debug=saved.debug;
        tm=saved.tm;
        timeUsed=saved.timeUsed;
    }
    

    
    private class TimingModel implements Serializable{
        double estimateSingleTreeHours(int n, int m){
            return 1; //This is a fraction of an hour! so .1 ==6 minutes
        }
        int estimateMaxAttributes(int n, int m){
            return m;
        }
    }
    public static void main(String[] args) {
           ContractRotationForestVersion1 rf= new ContractRotationForestVersion1();
           Instances data = ClassifierTools.loadData("Z:\\Data\\TSCProblems\\ArrowHead\\ArrowHead_TRAIN");
        try {
            rf.buildClassifier(data);
        FileOutputStream fos =
        new FileOutputStream("C:\\Temp\\c-RotF.ser");
        ObjectOutputStream out = new ObjectOutputStream(fos);
        out.writeObject(rf);
        fos.close();
        out.close();
            
        } catch (Exception ex) {
            Logger.getLogger(ContractRotationForestVersion1.class.getName()).log(Level.SEVERE, null, ex);
        }
        FileInputStream fis;
        try {
            fis = new FileInputStream("C:\\Temp\\RotF.ser");
            ObjectInputStream in = new ObjectInputStream(fis);             
            Object obj=in.readObject();
            rf=(ContractRotationForestVersion1)obj;
        } catch (Exception ex) {
            Logger.getLogger(ContractRotationForestVersion1.class.getName()).log(Level.SEVERE, null, ex);
        }
    
       
           

    }
    
/*
 Adjusted Rotation Forest to use a random selection of attributes for each
  tree.


 */
public static class RotationForestLimitedAttributes extends RotationForest{
    private int maxNumAttributes=100;
    public RotationForestLimitedAttributes(){
        super();
    }
    public void setMaxNumAttributes(int m){
            maxNumAttributes=m;
    }
    @Override
    final protected int [] attributesPermutation(int numAttributes, int classAttribute,
                                         Random random) {
        int [] permutation = new int[numAttributes-1];
        int i = 0;
        //This just ignores the class attribute
        for(; i < classAttribute; i++){
          permutation[i] = i;
        }
        for(; i < permutation.length; i++){
          permutation[i] = i + 1;
        }

        permute( permutation, random );
        if(numAttributes>maxNumAttributes){
        //TRUNCTATE THE PERMATION TO CONSIDER maxNumAttributes. 
        // we could do this more efficiently, but this is the simplest way. 
            int[] temp = new int[maxNumAttributes];
           System.arraycopy(permutation, 0, temp, 0, maxNumAttributes);
           permutation=temp;
        }
    return permutation;
    }    
    @Override
    public void buildClassifier(Instances data) throws Exception{
       super.buildClassifier(data);
    }
}
    
    
    
}
