/**
 * $RCSfile: Classify.java
 * $Revision: 1.0
 * $Date: 2015-6-24
 *
 * Copyright (C) 2015 EastHope, Inc. All rights reserved.
 *
 * Use is subject to license terms.
 */
package hk.newsRecommender;


import hk.mahout.bayes.kdd99.Kdd99CsvToSeqFile;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.classifier.naivebayes.AbstractNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.ComplementaryNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import au.com.bytecode.opencsv.CSVReader;

public class Classify {
	private static NaiveBayesModel naiveBayesModel = null;
	private static Map<String, Long> strOptionMap = Maps.newHashMap();
	private static List<String> strLabelList = Lists.newArrayList();
	
	public static void main(String[] args) throws Exception{
		Configuration conf = new Configuration();
		String hdfsUrl = conf.get("fs.defaultFS");
		
//		part1-------------------生成打标签的训练数据（符合mahout要求的以“，”分隔）--------------------------------------
//		Job job1 = Job.getInstance(conf, "generateUserNewsTaggedMatrix");
//		Path output1=new Path(hdfsUrl + "/data/recommend/class2");
//		HadoopUtil.delete(conf, output1);
//		job1.setJarByClass(TFIDF.class);
//		job1.setMapperClass(Mapper_Part1.class);
//		job1.setReducerClass(Reduce_Part1.class);
//		job1.setMapOutputKeyClass(Text.class);
//		job1.setMapOutputValueClass(Text.class);
//		job1.setOutputKeyClass(Text.class);
//		job1.setOutputValueClass(NullWritable.class);
//		FileInputFormat.addInputPath(job1, new Path(hdfsUrl + "/data/recommend/matrix2"));
//		FileInputFormat.addInputPath(job1, new Path(hdfsUrl + "/data/recommend/ClusterPointsInfo.txt"));
//		FileOutputFormat.setOutputPath(job1, output1);
//		job1.waitForCompletion(true);
		
//		part1---------------------------------------------------------------
		String trainFile = hdfsUrl+"/data/recommend/class2/part-r-00000";
		String trainSeqPath=hdfsUrl+"/data/recommend/class3";
		String trainSeqFile = trainSeqPath+"/matrixSeq.seq";
		String testFile = hdfsUrl+"/data/recommend/class1/matrix2/part-r-00000";
//		String testFile = hdfsUrl+"/data/recommend/class2/part-r-00000";
		String outputPath = hdfsUrl+"/data/recommend/class4";
		HadoopUtil.delete(conf, new Path[]{new Path(outputPath),new Path(trainSeqPath)});
		classify(conf,trainFile,trainSeqFile,testFile,outputPath,0);
	}
	
//	part1---------------生成打标签的词频矩阵---------------------------------------------------------
	public static class Mapper_Part1 extends Mapper<LongWritable, Text, Text, Text> {
		private String flag;
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			FileSplit split = (FileSplit) context.getInputSplit();
			flag = split.getPath().getName();// 判断读的数据集
		}
		public void map(LongWritable key, Text value, Context context) throws IOException,
		InterruptedException {
			if (flag.equals("ClusterPointsInfo.txt")) {
				String[] lineSplits = value.toString().split("\t");
				context.write(new Text(lineSplits[0]), new Text(lineSplits[1]));
			}else{
				int index=value.toString().indexOf(" ");
				String keyStr=value.toString().substring(0,index);
				String valStr=value.toString().substring(index+1);
				context.write(new Text(keyStr), new Text(valStr));
			}
		}
	}
	
	public static class Reduce_Part1 extends Reducer<Text, Text, Text, NullWritable> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			String label="";
			StringBuilder sb=new StringBuilder();
			for (Text text : values) {
				if(text.toString().contains(" ")){
					String[] valArray=text.toString().split(" ");
					for(String temp:valArray)
						sb.append(temp).append(",");
				}else
					label=text.toString();
			}
			sb.insert(0, label+",");
			context.write(new Text(sb.deleteCharAt(sb.length()-1).toString()), NullWritable.get());
		}
	}
	
//	part2---------------bayes分类---------------------------------------------------------
	public static void classify(Configuration conf,String trainFile,String trainSeqFile,String testFile,
			String outputPath,int labelIndex) throws Exception{
		// Step 1 : Convert CSV to Sequence file
		genNaiveBayesModel(conf, labelIndex,trainFile,trainSeqFile,false);
		// Step 2: Train NB
		train(conf,trainSeqFile,outputPath);
		// Step 3: Test to see result
		test(conf,testFile,labelIndex);
	}
	
	public static void genNaiveBayesModel(Configuration conf,int labelIndex,String trainFile,
			String trainSeqFile,boolean hasHeader) {
		CSVReader reader = null;
		try {
			FileSystem fs = FileSystem.get(conf);
			if(fs.exists(new Path(trainSeqFile)))
				fs.delete(new Path(trainSeqFile), true);
			SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, new Path(trainSeqFile), Text.class, VectorWritable.class);
			FileSystem fsopen = FileSystem.get(conf);
			FSDataInputStream in = fsopen.open(new Path(trainFile));
			reader = new CSVReader(new InputStreamReader(in));
			
			String[] header = null;
			if(hasHeader)
				header = reader.readNext();
			String[] line = null;
			Long l = 0L;
			while((line = reader.readNext()) != null) {
				if(labelIndex > line.length) break;
				l++;
				List<String> tmpList = Lists.newArrayList(line);
				String label = tmpList.get(labelIndex);
				if(!strLabelList.contains(label))
					strLabelList.add(label);
//				Text key = new Text("/" + label + "/" + l);
				Text key = new Text("/" + label + "/");
				tmpList.remove(labelIndex);
				
				VectorWritable vectorWritable = new VectorWritable();
				Vector vector = new RandomAccessSparseVector(tmpList.size(), tmpList.size());//???
				
				for(int i = 0; i < tmpList.size(); i++) {
					String tmpStr = tmpList.get(i);
					if(StringUtils.isNumeric(tmpStr))
						vector.set(i, Double.parseDouble(tmpStr));
					else 
						vector.set(i, parseStrCell(tmpStr)); 
				}
				vectorWritable.set(vector);
				writer.append(key, vectorWritable);
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void train(Configuration conf,String trainSeqFile,String outputPath) throws Exception	{
		System.out.println("~~~ begin to train ~~~");
		String outputDirectory = outputPath+"/result";
		String tempDirectory = outputPath+"/temp";
		FileSystem fs = FileSystem.get(conf);
		TrainNaiveBayesJob trainNaiveBayes = new TrainNaiveBayesJob();
		trainNaiveBayes.setConf(conf);
		
		fs.delete(new Path(outputDirectory),true);
		fs.delete(new Path(tempDirectory),true);
		// cmd sample: mahout trainnb -i train-vectors -el -li labelindex -o model -ow -c
		trainNaiveBayes.run(new String[] { 
				"--input", trainSeqFile, 
				"--output", outputDirectory,
				"-el", 
				"--labelIndex", "labelIndex",
				"--overwrite", 
				"--tempDir", tempDirectory });
		
		// Train the classifier
		naiveBayesModel = NaiveBayesModel.materialize(new Path(outputDirectory), conf);

		System.out.println("features: " + naiveBayesModel.numFeatures());
		System.out.println("labels: " + naiveBayesModel.numLabels());
	}
	
	public static void test(Configuration conf,String testFile,int labelIndex) throws IOException {
		System.out.println("~~~ begin to test ~~~");
		AbstractNaiveBayesClassifier classifier=new StandardNaiveBayesClassifier(naiveBayesModel);
	    
	    FileSystem fsopen = FileSystem.get(conf);
		FSDataInputStream in = fsopen.open(new Path(testFile));
		CSVReader csv = new CSVReader(new InputStreamReader(in));
	    csv.readNext(); // skip header
	    
	    String[] line = null;
	    double totalSampleCount = 0.;
	    double correctClsCount = 0.;
//	    String str="10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,0,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,4,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,8,7,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,0,0,14,0,8";
//	    List<String> newsList=new ArrayList<String>();
//	    newsList.add(str);
//	    for(int j=0;j<newsList.size();j++){
//	    	line=newsList.get(j).split(",");
	    while((line=csv.readNext())!=null){
//	    	注意：我们这里在测试数据集中多加了第一列作为新闻ID，到最后会用它来做类别与ID的对应关系的映射，因此，这里要
//	    	先将此列取出后再做分类预测。
	    	List<String> tmpList = Lists.newArrayList(line);
			String label = tmpList.get(labelIndex);
			tmpList.remove(labelIndex);
	    	totalSampleCount ++;
	    	Vector vector = new RandomAccessSparseVector(tmpList.size(),tmpList.size());
	    	for(int i = 0; i < tmpList.size(); i++) {
	    		String tempStr=tmpList.get(i);
	    		if(StringUtils.isNumeric(tempStr)) {
	    			vector.set(i, Double.parseDouble(tempStr));
	    		} else {
	    			Long id = strOptionMap.get(tempStr);
	    			if(id != null)
	    				vector.set(i, id);
	    			else {
	    				System.out.println(StringUtils.join(tempStr, ","));
	    				continue;
	    			}
	    		}
	    	}
    		Vector resultVector = classifier.classifyFull(vector);
			int classifyResult = resultVector.maxValueIndex();
			if(StringUtils.equals(label, strLabelList.get(classifyResult))) {
		    	correctClsCount++;
		    } else {
//		    	这里直接预测，其line[labelIndex]即要预测的数据这个不是分类标签，而是新闻ID号，所有都不会对应上，这里
//		    	直接将其分类的结果打印出来，下面也就不用打印准确率了，若将上面注释放开会看到测试庥的预测准确率，上面注释
//		    	部分第一列为分类标签。
		    	System.out.println("CorrectORItem=" + label + "\tClassify=" + strLabelList.get(classifyResult) );
		    }
	    }
//	    System.out.println("Correct Ratio:" + (correctClsCount / totalSampleCount));
	}
	
	private static Long parseStrCell(String str) {
		Long id = strOptionMap.get(str); 
		if( id == null) {
			id = (long) (strOptionMap.size() + 1);
			strOptionMap.put(str, id);
		} 
		return id;
	}
	
}
