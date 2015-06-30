/**
 * $RCSfile: NewsRecommender2.java
 * $Revision: 1.0
 * $Date: 2015-6-9
 *
 * Copyright (C) 2015 EastHope, Inc. All rights reserved.
 *
 * Use is subject to license terms.
 */
package hk.newsRecommender;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.HadoopUtil;

public class NewsRecommender2 {
	private static String categories[]={"0","10","17","29","33","37","38","53","69","72"};
	private static String userPrefer="9106148	0.00 0.889 0.00 0.00 0.00 0.111 0.00 0.00 0.00 0.00";
	
	public static void main(String[] args) throws Exception{
		Configuration conf = new Configuration();
		String hdfsUrl = conf.get("fs.defaultFS");
		
//		part1-----------------------------------------------------------------
		String pushPath=hdfsUrl + "/data/recommend/push1";
		Job job1 = Job.getInstance(conf, "generateUserNewsTaggedMatrix");
		Path output1=new Path(pushPath);
		HadoopUtil.delete(conf, output1);
		job1.setJarByClass(TFIDF.class);
		job1.setMapperClass(Mapper_Part1.class);
		job1.setReducerClass(Reduce_Part1.class);
		job1.setMapOutputKeyClass(Text.class);
		job1.setMapOutputValueClass(Text.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(NullWritable.class);
		FileInputFormat.addInputPath(job1, new Path(hdfsUrl + "/data/recommend/ClusterPointsInfo.txt"));
		FileOutputFormat.setOutputPath(job1, output1);
		job1.waitForCompletion(true);
		
//		part2-----------------------------------------------------------------
		Path newTagNewsPath=new Path(hdfsUrl + "/data/recommend/newsClassfiedTag.txt");
		recommend(conf,new Path(pushPath+"/part-r-00000"),newTagNewsPath);
		
	}
	
//	part1---------------hack news算法（重力衰减），对不同类别中新闻排序-----------------------------------------
	public static class Mapper_Part1 extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException,
		InterruptedException {
			String[] lineSplits = value.toString().split("\t");
			context.write(new Text(lineSplits[1]), new Text(lineSplits[0]));
		}
	}
	
	public static class Reduce_Part1 extends Reducer<Text, Text, Text, NullWritable> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			String label="";
			StringBuilder sb=new StringBuilder(key.toString()).append(":");
			Map<String,String> gravityMap=new HashMap<String,String>();
			for (Text text : values) {
				gravityMap.put(text.toString().split("\\|")[0], text.toString().split("\\|")[1]);
			}
			/*score=(P–1)/(T+2)^G
			 * P表示帖子的得票数，减去1是为了忽略发帖人的投票
			 * T表示距离发帖的时间（单位为小时），加上2是为了防止最新的帖子导致分母过小（之所以选择2，可能是因为从原始文章出现在其他网站，到转贴至Hacker News，平均需要两个小时）。
			 * G表示"重力因子"（gravity power），即将帖子排名往下拉的力量，默认值为1.8。
			 */
			int voteCount=gravityMap.size();
			for(String keyStr:gravityMap.keySet()){
				String publishTime=gravityMap.get(keyStr);
				long intervalHours=(System.currentTimeMillis()-Long.parseLong(publishTime))/1000/3600;
				double score=voteCount/Math.pow(intervalHours+2, 1.8);
				gravityMap.put(keyStr, Double.toString(score));
			}
			Map<String,String> sortedGravityMap=new NewsRecommender2().new ListMapComparator().getSortedMap(gravityMap);
			for(String keyStr:sortedGravityMap.keySet()){
				String val=sortedGravityMap.get(keyStr);
				sb.append(keyStr+"="+val).append(",");
			}
			context.write(new Text(sb.deleteCharAt(sb.length()-1).toString()), NullWritable.get());
		}
	}
	
//	part1---------------推荐，根据上步计算出来的重力得分，再根据用户的偏好因子相乘，取得分高者推荐----------
	public static void recommend(Configuration conf,Path oldNewsPath,Path newTagNewsPath) throws Exception{
		FileSystem fsopen = FileSystem.get(conf);
		FSDataInputStream in1 = fsopen.open(oldNewsPath);
		FSDataInputStream in2 = fsopen.open(newTagNewsPath);
		Scanner scan = new Scanner(in1);
		BufferedReader br = new BufferedReader(new InputStreamReader(in2));
		String curLine;
		List<String> keywordList=new ArrayList<String>();
		String[] userInfo=userPrefer.split("\t");
		String userID=userInfo[0];
		String[] userPreferArray=userInfo[1].split(" ");
		Map<String,Double> userPreferMap=new HashMap<String,Double>();
		Map<String,String> userPreferScoreMap=new HashMap<String,String>();
		List<String> newTagNewsList=new ArrayList<String>();
		for(int i=0;i<userPreferArray.length;i++){
			if(Double.parseDouble(userPreferArray[i])!=0f){
				userPreferMap.put(categories[i], Double.parseDouble(userPreferArray[i]));
			}
		}
		while (scan.hasNext()) {
			String lineStr=scan.next();
			String newsCategory=lineStr.split(":")[0];
			if(userPreferMap.containsKey(newsCategory)){
				String[] itemScoreArray=lineStr.split(":")[1].split(",");
				double userpreferValue=userPreferMap.get(newsCategory);
				for(int i=0;i<itemScoreArray.length;i++){
					String[] itemArray=itemScoreArray[i].split("=");
					userPreferScoreMap.put(itemArray[0], Double.toString((userpreferValue*Double.parseDouble(itemArray[1]))));
				}
			}
		}
		while ((curLine = br.readLine()) != null) {
			String newsCategory=curLine.split("\t")[1];
			if(userPreferMap.containsKey(newsCategory)){
				String[] itemScoreArray=curLine.split(":")[0].split("\\|");
				newTagNewsList.add(itemScoreArray[0]);
			}
		}
		Map<String,String> sortedUserPreferScoreMap=new NewsRecommender2().new ListMapComparator().getSortedMap(userPreferScoreMap);
		System.out.println("旧新闻推荐："+sortedUserPreferScoreMap);
		System.out.println("新新闻推荐："+newTagNewsList);
	}
	
	public class ListMapComparator implements Comparator {
		public Map<String,String> getSortedMap(Map map){
			Map<String,String> sortedMap=new LinkedHashMap<String,String>();
			List<Map.Entry<String, String>> list2=new ArrayList<Map.Entry<String, String>>(map.entrySet());
			Collections.sort(list2, new ListMapComparator());
			for(int i=0;i<list2.size();i++){
				Map.Entry<String, String> entry=list2.get(i);
				sortedMap.put(entry.getKey(), entry.getValue());
			}
			return sortedMap;
		}
		
		public int compare(Object element1, Object element2) {
			double a1=Double.parseDouble(((Map.Entry<String, String>)element1).getValue());
			double a2=Double.parseDouble(((Map.Entry<String, String>)element2).getValue());
			return (a1-a2)>0?-1:1;//降序排列，升序时将-1,1替换
		}
	}

	
}
