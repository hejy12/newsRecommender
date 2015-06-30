/**
 * $RCSfile: Cluster.java
 * $Revision: 1.0
 * $Date: 2015-6-22
 *
 * Copyright (C) 2015 EastHope, Inc. All rights reserved.
 *
 * Use is subject to license terms.
 */
package hk.newsRecommender;

import hk.newsRecommender.TFIDF.CustomKey;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.conversion.InputDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.clustering.ClusterDumper;

public class MatrixAndCluster {

	@SuppressWarnings("deprecation")
	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		String hdfsUrl = conf.get("fs.defaultFS");

//		part1---------------------------------------------------------------
//		Job job0 = Job.getInstance(conf, "siftKeywordsDimension");
//		Path output1Path=new Path(hdfsUrl + "/data/recommend/matrix1");
//		HadoopUtil.delete(conf, output1Path);
//		job0.setJarByClass(TFIDF.class);
//		job0.setMapperClass(Mapper_Part1.class);
//		job0.setReducerClass(Reduce_Part1.class);
//		job0.setMapOutputKeyClass(Text.class);
//		job0.setMapOutputValueClass(Text.class);
//		job0.setOutputKeyClass(Text.class);
//		job0.setOutputValueClass(Text.class);
//		job0.setPartitionerClass(CustomPartitioner.class);
//		FileInputFormat.addInputPath(job0, new Path(hdfsUrl + "/data/recommend/tfidf3"));
//		FileOutputFormat.setOutputPath(job0, output1Path);
//		job0.waitForCompletion(true);

//		part2---------------------------------------------------------------
//		FileSystem fsopen = FileSystem.get(conf);
//		FSDataInputStream in = fsopen.open(new Path(hdfsUrl + "/data/recommend/matrix1/part-r-00000"));
//		Scanner scan = new Scanner(in);
//		List<String> keywordList=new ArrayList<String>();
//		while (scan.hasNext()) {
//			keywordList.add(scan.next());
//		}
////		must before job
//		conf.setStrings("keyword", keywordList.toArray(new String[keywordList.size()]));
//		Job job1 = Job.getInstance(conf, "generateMatrix");
//		Path output2Path=new Path(hdfsUrl + "/data/recommend/matrix2");
//		HadoopUtil.delete(conf, output2Path);
//		job1.setJarByClass(TFIDF.class);
//		job1.setMapperClass(Mapper_Part2.class);
//		job1.setReducerClass(Reduce_Part2.class);
//		job1.setMapOutputKeyClass(Text.class);
//		job1.setMapOutputValueClass(Text.class);
//		job1.setOutputKeyClass(Text.class);
//		job1.setOutputValueClass(NullWritable.class);
////		job1.addCacheFile(new Path("/data/recommend/matrix1/part-r-00000").toUri());
//		FileInputFormat.addInputPath(job1, new Path(hdfsUrl + "/data/recommend/tfidf3"));
//		FileOutputFormat.setOutputPath(job1, output2Path);
//		job1.waitForCompletion(true);
		
//		part3-------------------聚类并打印--------------------------------------------
		Path output3Path=new Path(hdfsUrl + "/data/recommend/cluster2");
		HadoopUtil.delete(conf, output3Path);
		EuclideanDistanceMeasure measure = new EuclideanDistanceMeasure();
		Path clusterInput = new Path(hdfsUrl + "/data/recommend/matrix2");
		Path clusterSeqInput = new Path(hdfsUrl + "/data/recommend/cluster1");
		Path clusterOutput = new Path(hdfsUrl + "/data/recommend/cluster2");
		int k = 10;
		int maxIter = 3;
//		将数据文件转为mahout向量表示（这里要自己写）
//		InputDriver.runJob(clusterInput, clusterSeqInput, "org.apache.mahout.math.RandomAccessSparseVector");
//		 随机的选择k个作为簇的中心
		Path clusters = RandomSeedGenerator.buildRandom(conf, clusterSeqInput, 
				new Path(clusterOutput,"clusters-0"), k, measure);
		KMeansDriver.run(conf,clusterSeqInput,clusters,clusterOutput,0.01,maxIter,true, 0.0, false);
		// 调用 ClusterDumper 的 printClusters 方法将聚类结果打印出来。
		ClusterDumper clusterDumper = new ClusterDumper(new Path(clusterOutput, "clusters-"
				+ (maxIter - 1)), new Path(clusterOutput, "clusteredPoints"));
		clusterDumper.printClusters(null);

		
		clusterOutput(conf,new Path(hdfsUrl + "/data/recommend/cluster2/clusteredPoints/part-m-00000"));
//		clusterOutput2(conf0,new Path(hdfsUrl0 + "/data/recommend/cluster2/clusteredPoints/part-m-00000"));
//		matrix2Vector(conf0,new Path(hdfsUrl0 + "/data/recommend/cluster1/part-m-00000"));//暂时没用到

	}
	
	// part1---------------词频矩阵的维度---------------------------------------------------------
	public static class Mapper_Part1 extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException,
				InterruptedException {
			String[] lineSplits = value.toString().split("\t");
			String keyword = lineSplits[1].split(" ")[0];
			context.write(new Text(keyword), new Text(""));
		}
	}

	public static class Reduce_Part1 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			context.write(key, new Text(""));
		}
	}

	public static class CustomPartitioner<K1, V1> extends Partitioner<K1, V1> {
		@Override
		public int getPartition(K1 key, V1 value, int numPartitions) {
			CustomKey keyK = (CustomKey) key;
			Text tmpValue = new Text(keyK.getSymbol());
			return (tmpValue.hashCode() & Integer.MAX_VALUE) % numPartitions;
		}
	}

	// part2---------------生成词频矩阵---------------------------------------------------------
	public static class Mapper_Part2 extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException,
				InterruptedException {
			String[] lineSplits = value.toString().split("\t");
			context.write(new Text(lineSplits[0]), new Text(lineSplits[1]));
		}
		
	}

	public static class Reduce_Part2 extends Reducer<Text, Text, Text, NullWritable> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			String[] keywords=conf.getStrings("keyword");
			List<String> keywordsList=Arrays.asList(keywords);
			String newsID = key.toString().split("\\|")[0];
			long newsIDKeywordsCount = Long.parseLong(key.toString().split("\\|")[1]);
			String publishTime=key.toString().split("\\|")[2];
			StringBuilder sb=new StringBuilder(newsID+"|"+publishTime);
//			StringBuilder sb=new StringBuilder();
			Map<String,String> keywordMap=new HashMap<String,String>();
			for (Text value : values) {
				keywordMap.put(value.toString().split(" ")[0], value.toString().split(" ")[1]);
			}
			for(String str:keywordsList){
				if (!keywordMap.containsKey(str))
					sb.append(" ").append(0);
				else {
					double probability = Double.parseDouble(keywordMap.get(str));
					sb.append(" ").append(Math.round(probability * newsIDKeywordsCount));
				}
			}
//			sb.append(" ").append(newsID);
//			context.write(new Text(sb.toString()), new Text(newsID));
			context.write(new Text(sb.toString()), NullWritable.get());
		}
	}
	
//	 ---------------输出聚类结果---------------------------------------------------------
	// 表示向量的维数
	public static int Cardinality = 2029;

	public static void matrix2Vector(Configuration conf,Path path) throws IOException {
		FileSystem fs = FileSystem.get(conf);

		SequenceFile.Reader reader = null;
		// 读取原来的SequenceFile，将向量封装成具有Name属性的向量
		reader = new SequenceFile.Reader(fs, path, conf);
		Writable key = (Writable) ReflectionUtils.newInstance(reader.getKeyClass(), conf);
		Writable val = (Writable) ReflectionUtils.newInstance(reader.getValueClass(), conf);
		Writer writer=null;
		try {
			writer = SequenceFile.createWriter(fs, conf, path, IntWritable.class, VectorWritable.class, CompressionType.BLOCK);
			final IntWritable key1 = new IntWritable();
			final VectorWritable value = new VectorWritable();
			int lineNum = 0;
			Vector vector = null;
			while (reader.next(key, val)) {
				int index = 0;
				StringTokenizer st = new StringTokenizer(val.toString());
				// 将SequentialAccessSparseVector进一步封装成NamedVector类型
				vector = new NamedVector(new SequentialAccessSparseVector(Cardinality), lineNum + "");
				while (st.hasMoreTokens()) {
					if (Integer.parseInt(st.nextToken()) == 1) {
						vector.set(index, 1);
					}
					index++;
				}
				key1.set(lineNum++);
				value.set(vector);
				writer.append(key, value);
			}
		} finally {
			writer.close();
			reader.close();
		}
	}
		
	public static void clusterOutput(Configuration conf, Path path) {
		try {
			BufferedWriter bw;
			FileSystem fs = FileSystem.get(conf);

			SequenceFile.Reader reader = null;
			reader = new SequenceFile.Reader(fs, path, conf);

			// 将分组信息写到文件uidOfgrp.txt，每行格式为 uid \t groupID
			bw = new BufferedWriter(new FileWriter(new File("C:\\Users\\Hk\\Desktop\\ClusterPointsInfo.txt")));
			HashMap<String, Integer> clusterIds;
			clusterIds = new HashMap<String, Integer>(120);
			IntWritable key = new IntWritable();
			WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
//			WeightedVectorWritable value = new WeightedVectorWritable();
			while (reader.next(key, value)) {
				NamedVector vector = (NamedVector) value.getVector();
				// 得到Vector的Name标识
				String vectorName = vector.getName();
				System.out.println(vectorName + "\t" + key.toString());
				bw.write(vectorName + "\t" + key.toString() + "\n");
				// 更新每个group的大小
				if (clusterIds.containsKey(key.toString())) {
					clusterIds.put(key.toString(), clusterIds.get(key.toString()) + 1);
				} else
					clusterIds.put(key.toString(), 1);
			}
			bw.flush();
			reader.close();
			// 将每个group的大小，写入grpSize文件中
			bw = new BufferedWriter(new FileWriter(new File("C:\\Users\\Hk\\Desktop\\ClusterPointsSize.txt")));
			Set<String> keys = clusterIds.keySet();
			for (String k : keys) {
				System.out.println(k + " " + clusterIds.get(k));
				bw.write(k + " " + clusterIds.get(k) + "\n");
			}
			bw.flush();
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void clusterOutput2(Configuration conf, Path path) {
		try {
			FileSystem fs = FileSystem.get(conf);
			SequenceFile.Reader reader = new SequenceFile.Reader(fs,path,conf);
			IntWritable key = new IntWritable();
			WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
			while (reader.next(key, value)) {
				System.out.println(value.toString() + " belongs to cluster " + key.toString());
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
