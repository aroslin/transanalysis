package transanalysis.preprocess;


import conf.PathConf;
import conf.ParamConf;
import conf.StopWords;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @see AbstractVectorization
 * Extracts abstract in a document, generates a abstract-only document, and each line represents a abstract
 * @date 2016/11/5
 * @author:lh
 */

@SuppressWarnings("JavadocReference")
public class AbstractVectorization {
    //Number of docs
    private int num;

    private ParamConf.DataType dataType;

    //set if write file that each line is a cluster of docs;
    public boolean clusterDocOneVec;

    /**
     * @see AbstractVectorization
     * description:constructed function
     * @param DataType dt
     * Determines the type of file currently being read
     * @return
     * @date 2016/11/5
     * @author:lh
     */
    public AbstractVectorization(ParamConf.DataType dt) {
        this.num = 0;
        this.dataType = dt;
        this.clusterDocOneVec = false;
        File file = new File(PathConf.vecFilePath);
        if (file.exists()) file.delete();
        file = new File(PathConf.vecTrainFilePath);
        if (file.exists()) file.delete();
        file = new File(PathConf.clusterOnedocFile);
        if (file.exists()) file.delete();
    }

    /**
     * @see abExtracotr
     * description:Extracts abstract in source document
     * @param
     * @return
     * @date 2016/11/6
     * @author:lh
     */
    public void abExtracotr() {
        try {
            File file1 = new File(PathConf.vecFilePath);
            if (file1.exists()) file1.delete();
            File[] files = new File(PathConf.sourceFilePath).listFiles();
            this.num = 0;
            File oldTrainFile = new File(PathConf.vecTrainFilePath);
            if (oldTrainFile.exists()) oldTrainFile.delete();
            for (File file : files) {
                if (this.dataType == ParamConf.DataType.article) articleAbWrite(file.getName(), false);
                else if (this.dataType == ParamConf.DataType.patent) patentAbWrite(file.getName(), false);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * @see trainAbExtracotr
     * description:Extracts abstract in training document
     * Each document is a category
     * @param
     * @return
     * @date 2016/11/6
     * @author:lh
     */
    public void trainAbExtracotr() {
        try {
            List<String> trainData = new ArrayList<>();
            File[] files = new File(PathConf.sourceTrainFilePath).listFiles();
            File oldFile = new File(PathConf.vecTrainFilePath);
            if (oldFile.exists()) oldFile.delete();
            int curNum=0;
            for (File file : files) {
                int FirstNum = this.num;
                if (this.dataType == ParamConf.DataType.article) articleAbWrite(file.getName(), true);
                else if (this.dataType == ParamConf.DataType.patent) patentAbWrite(file.getName(), true);
                String className = file.getName().substring(0, file.getName().length() - ParamConf.fileExtension.length());
                trainData.add(className + "\t" + (FirstNum+1) + "\t" + this.num);
            }

            // Write id-label file
            // format label-id-startNum-stopNum
            Map<Integer, String> labelID = new HashMap<>();
            int num = 0;
            for (String s : trainData) {
                labelID.put(++num, s);
            }

            BufferedWriter labelWriter = new BufferedWriter(((
                    new OutputStreamWriter(
                            new FileOutputStream
                                    (new File
                                            (PathConf.labelIDFilePath)), ParamConf.outputEncode))));
            for (Map.Entry<Integer, String> meis : labelID.entrySet()) {
                String[] ss = meis.getValue().split("\t");
                String s = ss[0] + "\t" + (meis.getKey()) + "\t" + ss[1] + "\t" + ss[2];
                labelWriter.write(s + "\n");
            }
            labelWriter.flush();
            labelWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * @see articleAbWrite
     * description: write vectorized article abstract
     * @param String filename
     * Input article file name
     * @param boolean trainfile
     * Trainfile or not
     * @return
     * @date 2016/11/6
     * @author:lh
     */
    private void articleAbWrite(String filename, boolean trainfile) {
        try {
            String soureceFileName="";
            if (trainfile) soureceFileName = PathConf.sourceTrainFilePath + "\\" + filename;
            else soureceFileName = PathConf.sourceFilePath + "\\" + filename;

            BufferedReader modelReader = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(
                                    new File(soureceFileName)), ParamConf.sourceReadingEncode));
            BufferedWriter writer =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(PathConf.vecFilePath), true), ParamConf.outputEncode))));
            BufferedWriter trainWriter =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(PathConf.vecTrainFilePath), true), ParamConf.outputEncode))));
            BufferedWriter clusterOnedocWriter =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(PathConf.clusterOnedocFile), true), ParamConf.outputEncode))));
            clusterOnedocWriter.write(this.num + "\t");
            String line = "";
            boolean flag = false;
            String myabstract = "";
            while ((line = modelReader.readLine()) != null) {
                if (line.length() > 2) {
                    String[] Fword = line.split(" ");
                    if (Fword.length < 2) continue;
                    line = line.toLowerCase().trim();
                    if (Fword[0].equals("AB")) {
                        myabstract = line.substring(3);
                        flag = true;
                    } else if (flag && Fword[0].equals("")) myabstract += line;
                    else if (flag && !Fword[0].equals("")) flag = false;
                } else {
                    flag = false;
                    if (myabstract.equals("")) continue;

                    String[]ss=myabstract.split(" ");
                    String newmy="";
                    for (String s:ss){
                        if (StopWords.stopwords.contains(s) || s.length()<2) continue;
                        newmy+=s;
                    }
                    myabstract=newmy;

                    writer.write(this.num + "\t" + myabstract + "\n");
                    writer.flush();
                    if (trainfile) {
                        trainWriter.write(this.num + "\t" + myabstract + "\n");
                        trainWriter.flush();
                        clusterOnedocWriter.write(myabstract);
                        clusterOnedocWriter.flush();
                    }
                    this.num++;
                    myabstract = "";
                }
            }
            modelReader.close();
            writer.close();
            clusterOnedocWriter.newLine();
            clusterOnedocWriter.flush();
            clusterOnedocWriter.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @see patentAbWrite
     * description: write vectorized patent abstract
     * @param String filename
     * Input petent file name
     * @param boolean trainfile
     * Trainfile or not
     * @return
     * @date 2016/11/6
     * @author:lh
     */
    private void patentAbWrite(String filename, boolean trainfile) {
        try {
            String soureceFileName="";
            if (trainfile) soureceFileName = PathConf.sourceTrainFilePath + "\\" + filename;
            else soureceFileName = PathConf.sourceFilePath + "\\" + filename;

            BufferedReader modelReader = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(
                                    new File(soureceFileName)), ParamConf.sourceReadingEncode));
            BufferedWriter writer =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(PathConf.vecFilePath), true), ParamConf.outputEncode))));
            BufferedWriter trainWriter =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(PathConf.vecTrainFilePath), true), ParamConf.outputEncode))));
            BufferedWriter clusterOnedocWriter =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(PathConf.clusterOnedocFile), true), ParamConf.outputEncode))));
            clusterOnedocWriter.write(this.num + "\t");
            String line = "";
            boolean flag = false;
            String myabstract = "";
            while ((line = modelReader.readLine()) != null) {
                if (line.length() > 8) {
                    String[] Fword = line.split(" ");
                    if (Fword.length < 2) continue;
                    line = line.toLowerCase().trim();
                    if (Fword[0].equals("abd")) {
                        myabstract = line.substring(7);
                        flag = true;
                    } else if (flag && Fword[0].equals("")) myabstract += line;
                    else if (flag && !Fword[0].equals("")) flag = false;
                } else {
                    flag = false;
                    if (myabstract.equals("")) continue;
                    writer.write(this.num + "\t" + myabstract + "\n");
                    writer.flush();
                    if (trainfile) {
                        trainWriter.write(this.num + "\t" + myabstract + "\n");
                        trainWriter.flush();
                        clusterOnedocWriter.write(myabstract);
                        clusterOnedocWriter.flush();
                    }
                    this.num++;
                    myabstract = "";
                }
            }
            modelReader.close();
            writer.close();
            trainWriter.close();
            clusterOnedocWriter.newLine();
            clusterOnedocWriter.flush();
            clusterOnedocWriter.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
