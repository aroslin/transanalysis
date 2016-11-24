package conf;

/**
 * @see ParamConf
 * description:Parmas Set Config
 * @date 2016/11/5
 * @author:lh
 */
public class ParamConf {
    //The source file reading encoding format
    public static String sourceReadingEncode="UTF-8";
    //Ouput file reading encoding format
    public static String outputEncode="UTF-8";
    //The current data type to be analyzed
    public enum DataType{article,patent};
    //offline master name
    public static String offline="local";
    //file extension;
    public static String fileExtension=".txt";

}
