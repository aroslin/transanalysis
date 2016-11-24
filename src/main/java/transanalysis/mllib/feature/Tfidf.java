package transanalysis.mllib.feature;

import conf.ParamConf;
import conf.PathConf;
import conf.StopWords;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;

import java.io.*;
import java.util.*;

/**
 * @see Tfidf
 * description:term frequency–inverse document frequency
 * @date 2016/11/6
 * @author:lh
 */

@SuppressWarnings("JavadocReference")
public class Tfidf implements Serializable{
    // Number of Features(or words)

    public int FetureNum;
    // All features
    public List<String> allWordsList;

    /**
     * @see Tfidf
     * description: constructed function
     * @param
     * @return
     * @date 2016/11/6
     * @author:lh
     */
    public Tfidf(){
        this.FetureNum=0;
        this.allWordsList=new ArrayList<>();
    }

    /**
     * @see printTfidf
     * description: Generate vecotriedData file and dictionayr file
     * @param JavaSparkContext sc
     * @return
     * @date 2016/11/6
     * @author:lh
     */
    public void printTfidf(JavaSparkContext sc) {
        //get all words with cleaning
        extractAllWords(sc);

        //set DataFrame attribution
        SQLContext sqlCtx = new SQLContext(sc);
        sqlCtx.setConf("spark.sql.tungsten.enabled", "false");
        List<StructField> fields = new ArrayList<StructField>();
        fields.add(DataTypes.createStructField("id", DataTypes.IntegerType, true));
        fields.add(DataTypes.createStructField("words", DataTypes.createArrayType(DataTypes.StringType, true), true));
        StructType schema1 = DataTypes.createStructType(fields);

        //original doc word-fre vectorize
        final List<String> wordList=allWordsList;
        JavaRDD<Row> parsedRDD = sc.textFile(PathConf.vecFilePath).map(new Function<String, Row>() {
            @Override
            public Row call(String s) throws Exception {
                String[] stp1 = s.trim().split("\t");
                String[] stp=stp1[1].split(" ");
                List<String> lout=new ArrayList<String>();
                for (String s1:stp){
                    if (wordList.contains(s1)){
                        lout.add(s1);
                    }
                }
                return RowFactory.create(Integer.valueOf(stp1[0]),lout);
            }
        });
        DataFrame words = sqlCtx.createDataFrame(parsedRDD.rdd(), schema1);
        sqlCtx.clearCache();

        //term frequency
        CountVectorizerModel cvModel = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("OutPutCol")
                .setVocabSize(FetureNum) // <-- Specify the Max size of the vocabulary.
//                .setMinDF(docNum*0.005) // Specifies the minimum number of different documents a term must appear in to be included in the vocabulary.
                .fit(words);
        DataFrame featurizedData = cvModel.transform(words);

        //inverse document frequency
        IDFModel idf = new org.apache.spark.ml.feature.IDF().setMinDocFreq(2).setInputCol("OutPutCol").fit(featurizedData);
        DataFrame tfidf = idf.setInputCol("OutPutCol").setOutputCol("tfidf").transform(featurizedData);
        tfidf.drop("OutPutCol");

        //write tfidf
        JavaPairRDD<Long, Vector> corpus= tfidf.javaRDD().mapToPair(new PairFunction<Row, Long, Vector>() {
            @Override
            public Tuple2<Long, Vector> call(Row row) throws Exception {
                long l=Long.valueOf((int)row.getAs("id"));
                Vector vc = row.getAs("tfidf");
                return new Tuple2<Long, Vector>(l, vc);
            }
        });
        List<Tuple2<Long, Vector>> vectorizedData = corpus.collect();
        vecotriedDataWrite(vectorizedData);

        //get dictionary
        String[] dic=cvModel.vocabulary();
        int num=0;
        Map<String,Integer> dict=new HashMap<>();
        for (String s:dic){
            dict.put(s,num++);
        }
        dictionaryWrite(dict);
        highWeightWords(dict,vectorizedData);
    }

    /**
     * @see highWeightWords
     * description: get words that can describe doc
     * @param Map<String,Integer> dict
     * dictionary map
     * @param List<Tuple2<Long, Vector>
     * words with tfidf
     * @return
     * @date 2016/11/21
     * @author:lh
     */
    private void highWeightWords(Map<String,Integer> dict, List<Tuple2<Long, Vector>> vectorizedData){
        try {
            BufferedWriter highWeightWriter = new BufferedWriter(((
                    new OutputStreamWriter(
                            new FileOutputStream(
                                    new File(PathConf.highWeightWordsFile), false), ParamConf.outputEncode))));
            //Get word with its label
            Map<Integer,String> labelWord=new HashMap<>();
            for (Map.Entry<String,Integer> mesi:dict.entrySet())
                labelWord.put(mesi.getValue(),mesi.getKey());

            for(Tuple2<Long, Vector> t2lv:vectorizedData){
                //write label
                highWeightWriter.write(t2lv._1.intValue()+"\t");
                Map<Integer,Double> msd=new HashMap<>();
                //get tfidf int this doc
                Vector vec=t2lv._2;
                int i=0;
                for (double d:vec.toArray()){
                    msd.put(i,d);
                    i++;
                }
                //sort
                List<Map.Entry<Integer,Double>> mappingList = new ArrayList<Map.Entry<Integer,Double>>(msd.entrySet());
                Collections.sort(mappingList, new Comparator<Map.Entry<Integer,Double>>(){
                    public int compare(Map.Entry<Integer,Double> mapping1,Map.Entry<Integer,Double> mapping2){
                        return mapping2.getValue().compareTo(mapping1.getValue());
                    }
                });
                int num=0;
                for (Map.Entry<Integer,Double> meii : mappingList){
                    //get top 100words
                    if (num>100) break;;
                    highWeightWriter.write(labelWord.get(meii.getKey())+" ");
                    num++;
                }
                highWeightWriter.newLine();
                highWeightWriter.flush();
            }
            highWeightWriter.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @see extractAllWords
     * description: Get allwords with stopwords cleaning
     * @param JavaSparkContext sc
     * @return
     * @date 2016/11/6
     * @author:lh
     */
    private void extractAllWords(JavaSparkContext sc){
        FetureNum=0;
        JavaRDD<String> data = sc.textFile(PathConf.vecFilePath);
        final Long totalDocs=data.count();

        //Get all stopwords
        final List<String> stopWords=new ArrayList<>((StopWords.stopwords));
        sc.broadcast(stopWords);

        //Get all stopwords with stopwords cleaning
        JavaRDD<String> words = data.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterable<String> call(String s) throws Exception {
                String[] stp1 = s.trim().split("\t");
                String[] stp=stp1[1].toLowerCase().replaceAll("\\d+", "").replaceAll("[\\pP]", "").split(" ");
                List<String> l = new ArrayList();
                return new ArrayList<String>(Arrays.asList(stp));
            }
        }).filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String s) throws Exception {
                return !stopWords.contains(s);
            }
        }).filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String s) throws Exception {
                return s.length()>2;
            }
        });

        // Get word-1 pairs
        JavaPairRDD<String, Integer> ones = words.mapToPair(new PairFunction<String, String, Integer>() {
            @Override
            public Tuple2<String, Integer> call(String s) {
                return new Tuple2<String, Integer>(s, 1);
            }
        });

        // Get word-fre pairs
        JavaPairRDD<String, Integer> wordList = ones.reduceByKey(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer i1, Integer i2) {
                return i1+i2;
            }
        });

        //Specifies the minimum number of different documents a term must appear in to be included in the vocabulary.
        JavaPairRDD<String,Integer> wordstp1=wordList.filter(new Function<Tuple2<String, Integer>, Boolean>() {
            @Override
            public Boolean call(Tuple2<String, Integer> stringIntegerTuple2) throws Exception {
                if (stringIntegerTuple2._2 <(totalDocs * 0.005)){
                    return false;
                }
                else return true;
            }
        });


        //Get selected words
        JavaRDD<String> selectedWords = wordstp1.map(new Function<Tuple2<String, Integer>, String>() {
            @Override
            public String call(Tuple2<String, Integer> stringIntegerTuple2) throws Exception {
                return stringIntegerTuple2._1;
            }
        });

        allWordsList= selectedWords.collect();;
        FetureNum=allWordsList.size();

        return;
    }

    /**
     * @see dictionayryWrite
     * description: Write dictionary
     * @param Map<String,Integer> dict
     * dictionary
     * @return
     * @date 2016/11/6
     * @author:lh
     */
    private void dictionaryWrite(Map<String,Integer> dict){
        try {
            BufferedWriter dictionaryWriter = new BufferedWriter(((
                    new OutputStreamWriter(
                            new FileOutputStream(
                                    new File(PathConf.wordDictFile), false), ParamConf.outputEncode))));

            for (Map.Entry<String, Integer> mesi : dict.entrySet())
                dictionaryWriter.write(mesi.getKey() + "\t" + mesi.getValue() + "\n");
            dictionaryWriter.flush();
            dictionaryWriter.close();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @see vecotriedDataWrite
     * description: Write vectorizedData
     * @param List<Tuple2<Long, Vector>> vectorizedData
     * vectorizedData,long represents the label
     * @return
     * @date 2016/11/6
     * @author:lh
     */
    private void vecotriedDataWrite(List<Tuple2<Long, Vector>> vectorizedData){
        try {
            //Read labelID file,get train absctrac start and stop number of each category
            List<Integer> boundary=new ArrayList<>();
            BufferedReader modelReader=new BufferedReader(
                    new InputStreamReader(
                            new FileInputStream(
                                    new File(PathConf.labelIDFilePath)), ParamConf.sourceReadingEncode));
            String line=modelReader.readLine();
            String[] tpss=line.split("\t");
            boundary.add(Integer.valueOf(tpss[2]));
            boundary.add(Integer.valueOf(tpss[3]));
            while ((line=modelReader.readLine())!=null){
                tpss=line.split("\t");
                boundary.add(Integer.valueOf(tpss[3]));
            }

            //write data and trainning data
            //format: label feature1:tfidf1 feature2:tfidf2 ....featuren:tfidfn
            BufferedWriter dataWriter=new BufferedWriter(((
                    new OutputStreamWriter(
                            new FileOutputStream(
                                    new File(PathConf.testDataPath)),ParamConf.outputEncode))));
            BufferedWriter trainDataWriter=new BufferedWriter(((
                    new OutputStreamWriter(
                            new FileOutputStream(
                                    new File(PathConf.trainDataPath)),ParamConf.outputEncode))));
            for (Tuple2<Long, Vector> t2lv:vectorizedData){
                String[] ss = t2lv._2().toString().trim().split("\\[");
                String[] labels=ss[1].trim().replaceAll("\\]","").split(",");
                String[] tfidfs=ss[2].trim().replaceAll("\\]","").split(",");
                for (int i=0;i<boundary.size();++i){
//                    if (t2lv._1<boundary.get(i) && i==0){
                    if (i==0){
                        dataWriter.write(t2lv._1+" ");
                        for (int j=0;j<labels.length-1;++j){
                            dataWriter.write(labels[j]+":"+tfidfs[j]+" ");
                        }
                        dataWriter.write("\n");
                        break;
                    }
                    else {
                        if ((i!=0) && t2lv._1<=boundary.get(i) ){
                            int label=i-1;
                            trainDataWriter.write(label+" ");
                            for (int j=0;j<labels.length-1;++j){
                                trainDataWriter.write(labels[j]+":"+tfidfs[j]+" ");
                            }
                            trainDataWriter.write("\n");
                            break;
                        }
                    }
                }
            }
            dataWriter.flush();
            dataWriter.close();
            trainDataWriter.flush();
            trainDataWriter.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
