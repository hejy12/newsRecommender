/**
 * $RCSfile: TFIDF.java
 * $Revision: 1.0
 * $Date: 2015-6-9
 *
 * Copyright (C) 2015 EastHope, Inc. All rights reserved.
 *
 * Use is subject to license terms.
 */
package hk.newsRecommender;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.wltea.analyzer.lucene.IKAnalyzer;

public class TFIDF {
	static long totalArticle=100;

	public static void main(String[] args) throws Exception {
		// part1----------------------------------------------------
		Configuration conf0 = new Configuration();
		// 设置文件个数，在计算DF(文件频率)时会使用
		// FileSystem hdfs = FileSystem.get(conf1);
		// FileStatus p[] = hdfs.listStatus(new Path(args[0]));
		String hdfsUrl0 = conf0.get("fs.defaultFS");

		// 获取输入文件夹内文件的个数，然后来设置NumReduceTasks
		Job job0 = Job.getInstance(conf0, "My_tdif_part0");
		job0.setJarByClass(TFIDF.class);
		job0.setMapperClass(Mapper_Part0.class);
		// job1.setCombinerClass(Combiner_Part1.class); // combiner在本地执行，效率要高点。
		job0.setReducerClass(Reduce_Part0.class);
		job0.setMapOutputKeyClass(Text.class);
		job0.setMapOutputValueClass(Text.class);
		job0.setOutputKeyClass(Text.class);
		job0.setOutputValueClass(Text.class);
		// job1.setNumReduceTasks(p.length);

		FileInputFormat.addInputPath(job0, new Path(hdfsUrl0 + "/data/recommend/data2.txt"));
		FileOutputFormat.setOutputPath(job0, new Path(hdfsUrl0 + "/data/recommend/tfidf0"));

		job0.waitForCompletion(true);
		
		// part1----------------------------------------------------
		Configuration conf1 = new Configuration();
		// 设置文件个数，在计算DF(文件频率)时会使用
		// FileSystem hdfs = FileSystem.get(conf1);
		// FileStatus p[] = hdfs.listStatus(new Path(args[0]));
		String hdfsUrl = conf1.get("fs.defaultFS");

		// 获取输入文件夹内文件的个数，然后来设置NumReduceTasks
		Job job1 = Job.getInstance(conf1, "My_tdif_part1");
		job1.setJarByClass(TFIDF.class);
		job1.setMapperClass(Mapper_Part1.class);
		// job1.setCombinerClass(Combiner_Part1.class); // combiner在本地执行，效率要高点。
		job1.setReducerClass(Reduce_Part1.class);
		job1.setMapOutputKeyClass(Text.class);
		job1.setMapOutputValueClass(Text.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(Text.class);
		// job1.setNumReduceTasks(p.length);
		job1.setPartitionerClass(MyPartitoner.class); // 使用自定义MyPartitoner

		FileInputFormat.addInputPath(job1, new Path(hdfsUrl + "/data/recommend/tfidf0"));
		FileOutputFormat.setOutputPath(job1, new Path(hdfsUrl + "/data/recommend/tfidf1"));

		job1.waitForCompletion(true);
		// part2----------------------------------------
		Configuration conf2 = new Configuration();

		Job job2 = Job.getInstance(conf2, "My_tdif_part2");
		job2.setJarByClass(TFIDF.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(Text.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);
		job2.setMapperClass(Mapper_Part2.class);
		job2.setReducerClass(Reduce_Part2.class);
		// job2.setNumReduceTasks(p.length);

		FileInputFormat.setInputPaths(job2, new Path(hdfsUrl + "/data/recommend/tfidf1"));
		FileOutputFormat.setOutputPath(job2, new Path(hdfsUrl + "/data/recommend/tfidf2"));

		job2.waitForCompletion(true);
		
//		part3----------------------------------------
//		Configuration conf3 = new Configuration();
//		
//		Job job3 = Job.getInstance(conf3, "My_tdif_part3");
//		job3.setJarByClass(TFIDF.class);
//		job3.setMapOutputKeyClass(Text.class);
//		job3.setMapOutputValueClass(Text.class);
//		job3.setOutputKeyClass(Text.class);
//		job3.setOutputValueClass(Text.class);
//		job3.setMapperClass(Mapper_Part3.class);
//		job3.setReducerClass(Reduce_Part3.class);
//		// job2.setNumReduceTasks(p.length);
//		
//		FileInputFormat.setInputPaths(job3, new Path(hdfsUrl + "/data/recommend/tfidf2"));
//		FileOutputFormat.setOutputPath(job3, new Path(hdfsUrl + "/data/recommend/tfidf3"));
//		
//		job3.waitForCompletion(true);
//		hdfs.delete(new Path(args[1]), true);
	}
	
	// part1---------------新闻去重---------------------------------------------------------
	public static class Mapper_Part0 extends Mapper<LongWritable, Text, Text, Text> {
//		long totalLine=0L;
		public void map(LongWritable key, Text value, Context context) throws IOException,
				InterruptedException {
//			totalLine++;
			String[] lineSplits = value.toString().split("\t");
			String newsID = lineSplits[1];
			String content = lineSplits[4];
			context.write(new Text(newsID+" "+content),new Text(""));
		}
//		public void cleanup(Context context) throws IOException, InterruptedException {
//			// Map的最后，写入部行数
//			String str = "";
//			str += totalLine;
//			context.write(new Text("!totalLine"), new Text(str));
//			// 主要这里值使用的 "!"是特别构造的。 因为!的ascii比所有的字母都小。
//		}
	}
	
	public static class Reduce_Part0 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			context.write(key, new Text(""));
		}
	}

	// part1------------------------------------------------------------------------
	public static class Mapper_Part1 extends Mapper<LongWritable, Text, Text, Text> {
		String newsID = ""; // 保存文件名，根据文件名区分所属文件
		String content = "";
		String word;

		public void map(LongWritable key, Text value, Context context) throws IOException,
				InterruptedException {
			int all = 0; // 单词总数统计
			String[] lineSplits = value.toString().split(" ");
			newsID = lineSplits[0];
			content = lineSplits[1];

			Analyzer analyzer = new IKAnalyzer(false);
			TokenStream ts = analyzer.tokenStream("", new StringReader(content));
			ts.reset();
			CharTermAttribute cta = ts.getAttribute(CharTermAttribute.class);
			Map<String, Long> splitWordMap = new HashMap<String, Long>();
			while (ts.incrementToken()) {
				word = cta.toString();
				word += " ";
				word += newsID;
				all++;
				if (splitWordMap.containsKey(word))
					splitWordMap.put(word, splitWordMap.get(word) + 1);
				else
					splitWordMap.put(word, 1L);
			}
			Iterator iter = splitWordMap.entrySet().iterator();
			while (iter.hasNext()) {
				Map.Entry<String, Long> entry = (Map.Entry<String, Long>) iter.next();
				String key1 = entry.getKey();
				Long val = entry.getValue();
				context.write(new Text(key1), new Text((Float.parseFloat(val.toString()) / all)
						+ ""));
			}
		}

//		public void cleanup(Context context) throws IOException, InterruptedException {
//			// Map的最后，写入部行数
//			String str = "";
//			str += totalLine;
//			context.write(new Text("!totalLine"), new Text(str));
//			// 主要这里值使用的 "!"是特别构造的。 因为!的ascii比所有的字母都小。
//		}
	}

//	public static class Combiner_Part1 extends Reducer<Text, Text, Text, Text> {
//		float all = 0;
//		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
//				InterruptedException {
//			int index = key.toString().indexOf(" ");
//			// 因为!的ascii最小，所以在map阶段的排序后，!会出现在第一个
//			if (key.toString().substring(index + 1, index + 2).equals("!")) {
//				for (Text val : values) {
//					all = Integer.parseInt(val.toString());
//				}
//				return;// 这个key-value被抛弃
//			}
//			float sum = 0; // 统计某个单词出现的次数
//			for (Text val : values) {
//				sum += Integer.parseInt(val.toString());
//			}
//			// 跳出循环后，某个单词数出现的次数就统计完了，所有 TF(词频) = sum / all
//			float tmp = sum / all;
//			String value = "";
//			value += tmp; // 记录词频
//
//			// 将key中单词和文件名进行互换。es: test1 hello -> hello test1
//			String p[] = key.toString().split(" ");
//			String key_to = "";
//			key_to += p[1];
//			key_to += " ";
//			key_to += p[0];
//			context.write(new Text(key_to), new Text(value));
//		}
//	}

	public static class Reduce_Part1 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			for (Text val : values) {
				context.write(key, val);
			}
		}
	}

	public static class MyPartitoner extends Partitioner<Text, Text> {
		// 实现自定义的Partitioner
		public int getPartition(Text key, Text value, int numPartitions) {
			// 我们将一个文件中计算的结果作为一个文件保存
			// es： test1 test2
			String ip1 = key.toString();
			ip1 = ip1.substring(0, ip1.indexOf(" "));
			Text p1 = new Text(ip1);
			return Math.abs((p1.hashCode() * 127) % numPartitions);
		}
	}

	// part2-----------------------------------------------------
	public static class Mapper_Part2 extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException,
				InterruptedException {
			String val = value.toString().replaceAll("	", " "); // 将vlaue中的TAB分割符换成空格
			int index = val.indexOf(" ");
			String s1 = val.substring(0, index); // 获取单词 作为key es: hello
			String s2 = val.substring(index + 1); // 其余部分 作为value es: test1 0.11764706
			s2 += " ";
			s2 += "1"; // 统计单词在所有文章中出现的次数, “1” 表示出现一次。 es: test1 0.11764706 1
			context.write(new Text(s1), new Text(s2));
		}
	}

	public static class Reduce_Part2 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			// 同一个单词会被分成同一个group
			float sum = 0;
			List<String> vals = new ArrayList<String>();
			for (Text str : values) {
				int index = str.toString().lastIndexOf(" ");
				sum += Integer.parseInt(str.toString().substring(index + 1)); // 统计此单词在所有文件中出现的次数
				vals.add(str.toString().substring(0, index)); // 保存
			}
			double tmp = Math.log10(totalArticle * 1.0 / (sum * 1.0)); // 单词在所有文件中出现的次数除以总文件数IDF
			for (int j = 0; j < vals.size(); j++) {
				String val = vals.get(j);
				String newsID=val.substring(0,val.indexOf(" "));
				String end = val.substring(val.lastIndexOf(" "));
				float f_end = Float.parseFloat(end); // 读取TF
				val += " ";
				val += f_end * tmp; // tf-idf值
//				context.write(key, new Text(val));
				context.write(new Text(newsID), new Text(key+" "+val.substring(val.indexOf(" ")+1)));
			}
		}
	}
	
//	public static class Mapper_Part3 extends Mapper<LongWritable, Text, Text, Text> {
//		public void map(LongWritable key, Text value, Context context) throws IOException,
//				InterruptedException {
//			String val = value.toString().replaceAll("	", " "); // 将vlaue中的TAB分割符换成空格
//			int index = val.indexOf(" ");
//			String s1 = val.substring(0, index); // 获取单词 作为key es: hello
//			String s2 = val.substring(index + 1); // 其余部分 作为value es: test1 0.11764706
//			context.write(new Text(s1), new Text(s2));
//		}
//	}
//	
//	public static class Reduce_Part3 extends Reducer<Text, Text, Text, Text> {
//		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
//				InterruptedException {
//			int limit=0;
//			for (Text str : values) {
//				if(++limit<=5){
//					context.write(key, str);
//				}
//			}
//		}
//	}

}
