package conf;

import java.io.File;

/**
 * @see PathConf
 * description:Path Set Config
 * @date 2016/11/5
 * @author:lh
 */

public class PathConf {
    /**
     * @see PathConf
     * constructed function
     * @param
     * @return
     * @date 2016/11/7
     * @author:lh
     */
    public PathConf(){
        File file=new File(currentPath+"/data");
        if (!file.exists()) file.mkdir();
        file=new File(currentPath+"/result");
        if (!file.exists()) file.mkdir();
    }

    // Project directory
    public static String currentPath= System.getProperty("user.dir");


    /*AbstractVectoriztion*/
    // Source file directory
    public static String sourceFilePath ="E:\\Data\\new\\Paper\\";
    // Source training set directory
    public static String sourceTrainFilePath ="E:\\Data\\NewRobotPatent20161121\\";
    // Abstract vectorization file Path,including both source file and training file
    public static String vecFilePath =currentPath+"/data/testAB.txt";
    // Training file Abstract vectorization file Path
    public static String vecTrainFilePath =currentPath+"/data/trainAB.txt";


    /*Tfidf*/
    //tfidf generated training file path
    public static String trainDataPath=currentPath+"/data/trainVec.txt";
    //tfidf generated testing file path
    public static String testDataPath=currentPath+"/data/testVec.txt";
    //label-id file path
    public static String labelIDFilePath =currentPath+"/data/labelID.txt";
    //dictionary file path
    public static String wordDictFile =currentPath+"/data/wordDict.txt";
    //High Weight Words
    public static String highWeightWordsFile=currentPath+"/data/highWeightWords.txt";
    //one cluster eachline file
    public static String clusterOnedocFile=currentPath+"/data/clusterOneDoc.txt";


    /*LdaExmaple*/
    // output dir
    public static String ldaOutputDir=currentPath+"/data/result";
}
