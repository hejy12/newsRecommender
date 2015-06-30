/**
 * $RCSfile: UserTag.java
 * $Revision: 1.0
 * $Date: 2015-6-24
 *
 * Copyright (C) 2015 EastHope, Inc. All rights reserved.
 *
 * Use is subject to license terms.
 */
package hk.newsRecommender;

import java.io.IOException;
import java.text.NumberFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.HadoopUtil;

public class UserTag {

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		String hdfsUrl = conf.get("fs.defaultFS");
		FileSystem fs = FileSystem.get(conf);

		Job job1 = Job.getInstance(conf, "generateUserNewsMapping");
		Path output1Path=new Path(hdfsUrl + "/data/recommend/user1");
		HadoopUtil.delete(conf, output1Path);
		job1.setJarByClass(TFIDF.class);
		job1.setMapperClass(Mapper_Part1.class);
		job1.setMapOutputKeyClass(Text.class);
		job1.setMapOutputValueClass(Text.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job1, new Path(hdfsUrl + "/data/recommend/data2.txt"));
		FileOutputFormat.setOutputPath(job1, output1Path);
		job1.waitForCompletion(true);
		
		
		Job job2 = Job.getInstance(conf, "generateUserNewsCatMapping");
		Path output2Path=new Path(hdfsUrl + "/data/recommend/user2");
		HadoopUtil.delete(conf, output2Path);
		job2.setJarByClass(UserTag.class);
		job2.setMapperClass(Mapper_Part2.class);
		job2.setReducerClass(Reduce_Part2.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(Text.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job2, new Path(hdfsUrl + "/data/recommend/user1"));
		FileInputFormat.addInputPath(job2, new Path(hdfsUrl + "/data/recommend/ClusterPointsInfo.txt"));
		FileOutputFormat.setOutputPath(job2, output2Path);
		job2.waitForCompletion(true);
		
		Job job3 = Job.getInstance(conf, "countUserNewsCatMapping");
		Path output3Path=new Path(hdfsUrl + "/data/recommend/user3");
		HadoopUtil.delete(conf, output3Path);
		job3.setMapperClass(Mapper_Part3.class);
		job3.setReducerClass(Reduce_Part3.class);
		job3.setMapOutputKeyClass(Text.class);
		job3.setMapOutputValueClass(Text.class);
		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job3, new Path(hdfsUrl + "/data/recommend/user2"));
		FileOutputFormat.setOutputPath(job3, output3Path);
		job3.waitForCompletion(true);
		
		Job job4 = Job.getInstance(conf, "generateClusterUniqueRecord");
		Path output4Path=new Path(hdfsUrl + "/data/recommend/user4");
		HadoopUtil.delete(conf, output4Path);
		job4.setMapperClass(Mapper_Part4.class);
		job4.setReducerClass(Reduce_Part4.class);
		job4.setMapOutputKeyClass(Text.class);
		job4.setMapOutputValueClass(Text.class);
		job4.setOutputKeyClass(Text.class);
		job4.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job4, new Path(hdfsUrl + "/data/recommend/ClusterPointsInfo.txt"));
		FileOutputFormat.setOutputPath(job4, output4Path);
		job4.waitForCompletion(true);
		
		
		FileSystem fsopen = FileSystem.get(conf);
		FSDataInputStream in = fsopen.open(new Path(hdfsUrl + "/data/recommend/user4/part-r-00000"));
		Scanner scan = new Scanner(in);
		List<String> keywordList=new ArrayList<String>();
		while (scan.hasNext()) {
			keywordList.add(scan.next());
		}
		conf.setStrings("category", keywordList.toArray(new String[keywordList.size()]));
		Path outPath4 = new Path(hdfsUrl + "/data/recommend/user5");
		if (fs.exists(outPath4)) {
			fs.delete(outPath4, true);
			System.out.println("存在此输出路径，已删除！！！");
		}
		Job job5 = Job.getInstance(conf, "generateUserPreferableMatrix");
		job5.setMapperClass(Mapper_Part5.class);
		job5.setReducerClass(Reduce_Part5.class);
		job5.setMapOutputKeyClass(Text.class);
		job5.setMapOutputValueClass(Text.class);
		job5.setOutputKeyClass(Text.class);
		job5.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job5, new Path(hdfsUrl + "/data/recommend/user3"));
		FileOutputFormat.setOutputPath(job5, new Path(hdfsUrl + "/data/recommend/user5"));
		job5.waitForCompletion(true);

	}

//	 part0---------------生成用户-新闻对应关系---------------------------------------------------------
	public static class Mapper_Part1 extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException,
				InterruptedException {
			String[] lineSplits = value.toString().split("\t");
			String publishTime=lineSplits[5];
			Calendar cal1 = Calendar.getInstance();
			try {
				cal1.setTime(new SimpleDateFormat("yyyy年MM月dd日HH:mm").parse(publishTime));
				publishTime=Long.toString(cal1.getTimeInMillis());
			} catch (Exception e) {
				publishTime="0";
			}
			context.write(new Text(lineSplits[0]), new Text(lineSplits[1]+"|"+publishTime));
		}
	}
	
//	 part1---------------用户打标签---------------------------------------------------------
	public static class Mapper_Part2 extends Mapper<LongWritable, Text, Text, Text> {
		private String flag;
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			FileSplit split = (FileSplit) context.getInputSplit();
			flag = split.getPath().getName();// 判断读的数据集
		}
		public void map(LongWritable key, Text value, Context context) throws IOException,
		InterruptedException {
			String[] lineSplits = value.toString().split("\t");
			if (flag.equals("ClusterPointsInfo.txt")) {
				context.write(new Text(lineSplits[0]), new Text("A"+lineSplits[1]));
			}else{
				context.write(new Text(lineSplits[1]), new Text("B"+lineSplits[0]));
			}
		}
	}
	
	public static class Reduce_Part2 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			List<String> userIdList=new ArrayList<String>();
			String category="";
			for (Text text : values) {
				if(text.toString().startsWith("B"))
					userIdList.add(text.toString().substring(1));
				else
					category=text.toString().substring(1);
			}
			
			for(String userId:userIdList){
				context.write(new Text(userId), new Text(key+" "+category));
			}
		}
	}

//	 part2---------------生成用户-新闻类别统计---------------------------------------------------------
	public static class Mapper_Part3 extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException,
				InterruptedException {
			String[] lineSplits = value.toString().split("\t");
			String[] lineVal = lineSplits[1].split(" ");
			context.write(new Text(lineSplits[0] + " " + lineVal[1]), new Text(""));
		}
	}

	public static class Reduce_Part3 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			int counter = 0;
			for (Text text : values) {
				counter++;
			}

			context.write(key, new Text(counter + ""));
		}
	}
	
//	part3---------------生成新闻聚类的不重复类别---------------------------------------------------------
	public static class Mapper_Part4 extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException,
		InterruptedException {
			String[] lineSplits = value.toString().split("\t");
			context.write(new Text(lineSplits[1]),new Text(""));
		}
	}
	
	public static class Reduce_Part4 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
		InterruptedException {
			context.write(key, new Text(""));
		}
	}
	
//	 part4---------------生成用户偏好矩阵---------------------------------------------------------
	public static class Mapper_Part5 extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException,
		InterruptedException {
			String[] lineSplits = value.toString().split("\t");
			String[] keyStr=lineSplits[0].split(" ");
			context.write(new Text(keyStr[0]), new Text(keyStr[1]+" "+lineSplits[1]));
		}
	}
	
	public static class Reduce_Part5 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
		InterruptedException {
			Map<String,String> userCatMap=new HashMap<String,String>();
			StringBuilder sb=new StringBuilder();
			Configuration conf = context.getConfiguration();
			String[] catArray=conf.getStrings("category");
			float scoreSum=0.0f;
			NumberFormat nf = NumberFormat.getNumberInstance();
			nf.setMinimumFractionDigits(2);
			
			for (Text text : values) {
				String[] valueSplit=text.toString().split(" ");
				userCatMap.put(valueSplit[0],valueSplit[1]);
				scoreSum+=Float.parseFloat(valueSplit[1]);
			}
			for(String cat:catArray){
				if(userCatMap.containsKey(cat)){
					sb.append(nf.format(Integer.parseInt(userCatMap.get(cat))/scoreSum)).append(" ");
				}else
					sb.append(nf.format(0)).append(" ");
			}
			context.write(key, new Text(sb.deleteCharAt(sb.length()-1).toString()));
		}
	}


}
