package transanalysis.mllib.clustering;

import conf.ParamConf;
import conf.PathConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.EMLDAOptimizer;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.*;
import java.util.*;

/**
 * @see LdaExmaple
 * Latent Dirichlet Allocation
 * @date 2016/11/7
 * @author:lh
 */

@SuppressWarnings("JavadocReference")
public class LdaExmaple implements Serializable {

    //Clustering number
    private int K;

    private Map<Integer,String> dictionary;


    /**
     * @see LdaExmaple
     * constructed function
     * @param int k
     * Clustering number
     * @return
     * @date 2016/11/7
     * @author:lh
     */
    public LdaExmaple(int k) {
        this.K = k;
        try {
            dictionary=new HashMap<>();
            BufferedReader modelReader = new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(
                                    new File(PathConf.wordDictFile)), ParamConf.sourceReadingEncode));

            String line="";
            while ((line = modelReader.readLine()) != null){
                String[] ss=line.split("\t");
                dictionary.put(Integer.valueOf(ss[1]),ss[0]);
            }
            modelReader.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @see ldaExample
     * start lda estimate and predict
     * @param JavaSparkContext sc
     * @return
     * @date 2016/11/7
     * @author:lh
     */
    public void run(JavaSparkContext sc) {
        final int featureSize = dictionary.size();
        //read vectorized file
        JavaPairRDD<Long, Vector> corpus = sc.textFile(PathConf.testDataPath).mapToPair(new PairFunction<String, Long, Vector>() {
            @Override
            public Tuple2<Long, Vector> call(String s) throws Exception {
                String[] ss = s.split(" ");
                Long label = Long.valueOf(ss[0]);
                int[] index = new int[ss.length - 1];
                double[] value = new double[ss.length - 1];
                for (int i = 1; i < ss.length; ++i) {
                    String[] indexValue = ss[i].split(":");
                    index[i-1] = Integer.valueOf(indexValue[0]);
                    value[i-1] = Double.valueOf(indexValue[1]);
                }
                return new Tuple2<Long, Vector>(label, Vectors.sparse(featureSize, index, value));
            }
        });
        corpus.cache();

        // lda model
        DistributedLDAModel ldaModel = (DistributedLDAModel) new org.apache.spark.mllib.clustering.LDA()
                .setK(K)
                .setOptimizer(new EMLDAOptimizer())
                .run(corpus);
        // transform format to ouput
        File file = new File(PathConf.ldaOutputDir + K);
        if (!file.exists() && !file.isDirectory()) {
            file.mkdir();
        }

        //（topic|doc）matrix
        JavaRDD<Tuple2<Object, Vector>> javalocalLDAModel = ldaModel.topicDistributions().toJavaRDD();
        JavaPairRDD<Long, Vector> docTopicMatrix = javalocalLDAModel.mapToPair(
                new PairFunction<Tuple2<Object, Vector>, Long, Vector>() {
                    public Tuple2<Long, Vector> call(Tuple2<Object, Vector> tp) {
                        double[] tpv = tp._2.toArray();
                        for (int i = 0; i < tpv.length; ++i) {
                            tpv[i] = tp._2.apply(i);
                        }
                        Vector out = Vectors.dense(tpv);
                        return new Tuple2<Long, Vector>((Long) tp._1(), out);
                    }
                }
        );
        List<Tuple2<Long, Vector>> docTopic = docTopicMatrix.collect();
        writeTopicDocs(docTopic, PathConf.ldaOutputDir + K + "\\Doc.txt");

        //（word|topic）matrix
        Matrix topics = ldaModel.topicsMatrix();
        File outfile = new File(PathConf.ldaOutputDir + K + "\\Words.txt");
        if (outfile.exists()) outfile.delete();
        for (int topic = 0; topic < K; topic++) {
            Map<String, Double> msi = new HashMap<>();
            for (int word = 0; word < ldaModel.vocabSize();word++) {
                if (dictionary.containsKey(word)){
                    msi.put(dictionary.get(word), topics.apply(word, topic));
                }
            }
            writeTopicWords(msi, PathConf.ldaOutputDir + K + "\\Words.txt", topic);
        }

        //top documents per topic
        Tuple2<long[], double[]>[] topDoc = ldaModel.topDocumentsPerTopic(20);
        writeTopDoc(topDoc, PathConf.ldaOutputDir + K + "\\高相关度文档.txt");

        //top documents per topic
        Tuple2<int[], double[]>[] describeWords = ldaModel.describeTopics(featureSize*5);
        writeDescribeWords(describeWords, PathConf.ldaOutputDir + K + "\\主题描述.txt");
    }

    /**
     * @see writeTopicWords
     * write topic words
     * @param Map<String, Double> words
     * words to be writen
     * @param String filename
     * writen file
     * @param int TopicNum
     * @return
     * @date 2016/11/7
     * @author:lh
     */
    private void writeTopicWords(Map<String, Double> words, String filename, int TopicNum){
        try {
            BufferedWriter topicWordsWriter =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(filename), true), ParamConf.outputEncode))));
            topicWordsWriter.write("Topic" + TopicNum + ":\t");


            List<Map.Entry<String, Double>> mappingList = new ArrayList<Map.Entry<String, Double>>(words.entrySet());
            //Comparator to achieve comparative sort
            Collections.sort(mappingList, new Comparator<Map.Entry<String, Double>>() {
                public int compare(Map.Entry<String, Double> mapping1, Map.Entry<String, Double> mapping2) {
                    return mapping2.getValue().compareTo(mapping1.getValue());
                }
            });

            //write top 100 by predict
            Map<String, Double> sortedMap = new TreeMap<>();
            int num = 0;
            for (Map.Entry<String, Double> mapping : mappingList) {
                if (num > 100) break;
                topicWordsWriter.write(mapping.getKey() + "\t");
                num++;
            }

            //close
            topicWordsWriter.newLine();
            topicWordsWriter.newLine();
            topicWordsWriter.flush();
            topicWordsWriter.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @see writeDocTopics
     * write doc topics
     * @param List<Tuple2<Long, Vector>> docTopics
     * topic and docs under it
     * @param String filename
     * @return
     * @date 2016/11/7
     * @author:lh
     */
    private void writeTopicDocs(List<Tuple2<Long, Vector>> topicDocs, String filename){
        try {
            File outfile = new File(filename);
            if (!outfile.exists())
                outfile.createNewFile();
            BufferedWriter topicDocsWriter =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(filename), false), ParamConf.outputEncode))));

            List<List<Integer>> outList = new ArrayList<>();
            for (int i = 0; i < K; ++i) {
                outList.add(new ArrayList<Integer>());
            }

            //get most related topic
            for (Tuple2<Long, Vector> tlv : topicDocs) {
                double[] vec = tlv._2.toArray();
                double max = 0;
                int num = 0;
                for (int i = 0; i < vec.length; ++i) {
                    if (vec[i] > max) {
                        max = vec[i];
                        num = i;
                    }
                }
                outList.get(num).add(tlv._1.intValue());
            }

            //write
            for (int i = 0; i < outList.size(); ++i) {
                topicDocsWriter.write("Topic" + i + ":\t");
                List<Integer> li = outList.get(i);
                Collections.sort(li);
                topicDocsWriter.write(li + "\n" + "\n");
            }
            topicDocsWriter.flush();
            topicDocsWriter.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @see writeTopDoc
     * write top related doc
     * @param Tuple2<long[], double[]>[] topDoc
     * topic 1 doc1:relation1 doc2:relation2
     * topic 2 doc1:relation1 doc2:relation2
     * ....
     * @param String filename
     * @return
     * @date 2016/11/7
     * @author:lh
     */
    private void writeTopDoc(Tuple2<long[], double[]>[] topDoc, String filename){
        try {
            File outfile = new File(filename);
            if (!outfile.exists())
                outfile.createNewFile();
            BufferedWriter topDocsWriter =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(filename), false), ParamConf.outputEncode))));;
            for (int i = 0; i < topDoc.length; ++i) {
                topDocsWriter.write("Topic" + i + ":\t");
                for (int j = 0; j < topDoc[i]._1.length; ++j)
                    topDocsWriter.write((int) topDoc[i]._1[j] + ":" + topDoc[i]._2[j] + ",\n");
                topDocsWriter.write("\n" + "\n");
            }
            topDocsWriter.flush();
            topDocsWriter.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @see writeDescribeWords
     * write describe words of each topic
     * @param Tuple2<int[], double[]>[] topDoc
     * topic 1 word1:relation1 word2:relation2
     * topic 2 word1:relation1 word2:relation2
     * ....
     * @param String filename
     * @return
     * @date 2016/11/7
     * @author:lh
     */
    private void writeDescribeWords(Tuple2<int[], double[]>[] topWords, String filename){
        try {
            File outfile = new File(filename);
            if (!outfile.exists())
                outfile.createNewFile();
            BufferedWriter topWordsWriter =
                    new BufferedWriter((
                            (new OutputStreamWriter
                                    (new FileOutputStream
                                            (new File(filename), false), ParamConf.outputEncode))));;
            for (int i = 0; i < topWords.length; ++i) {
                topWordsWriter.write("Topic" + i + ":\n");
                for (int j = 0; j < topWords[i]._1.length; ++j) {
                    topWordsWriter.write(dictionary.get((int) topWords[i]._1[j]) + " ");
                    System.out.println(topWords[i]._1[j] + "\t" + dictionary.get((int) topWords[i]._1[j]));
                }
                topWordsWriter.write("\n" + "\n");
            }
            topWordsWriter.flush();
            topWordsWriter.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}