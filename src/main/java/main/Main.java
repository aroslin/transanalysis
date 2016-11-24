package main;

import conf.ParamConf;
import conf.PathConf;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SQLContext;
import transanalysis.mllib.clustering.LdaExmaple;
import transanalysis.mllib.feature.Tfidf;
import transanalysis.preprocess.AbstractVectorization;

/**
 * @see Main
 * @date 2016/11/6
 * @author:lh
 */

public class Main{
    public static void main(String[] args){
        new PathConf();

//        AbstractVectorization abv=new AbstractVectorization(ParamConf.DataType.patent);
////        abv.abExtracotr();
//        abv.trainAbExtracotr();

        System.setProperty("spark.executor.memory","4G");
        SparkConf sparkConf = new SparkConf().setAppName("SVMTrain").setMaster(ParamConf.offline);
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

//        Tfidf t=new Tfidf();
//        t.printTfidf(sc);

        LdaExmaple lda=new LdaExmaple(7);
        lda.run(sc);
    }
}
