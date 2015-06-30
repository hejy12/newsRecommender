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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.StringReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.common.HadoopUtil;
import org.wltea.analyzer.lucene.IKAnalyzer;

public class TFIDF2 {
	static long totalArticle=100;

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		String hdfsUrl = conf.get("fs.defaultFS");

//		part0----------------------------------------------------
		Job job0 = Job.getInstance(conf, "sfitSingleNews");
		Path output0Path=new Path(hdfsUrl + "/data/recommend/tfidf0");
		HadoopUtil.delete(conf, output0Path);
		job0.setJarByClass(TFIDF.class);
		job0.setMapperClass(Mapper_Part0.class);
		// job1.setCombinerClass(Combiner_Part1.class); // combiner在本地执行，效率要高点。
		job0.setReducerClass(Reduce_Part0.class);
		job0.setMapOutputKeyClass(Text.class);
		job0.setMapOutputValueClass(Text.class);
		job0.setOutputKeyClass(Text.class);
		job0.setOutputValueClass(Text.class);
		// job1.setNumReduceTasks(p.length);
		FileInputFormat.addInputPath(job0, new Path(hdfsUrl + "/data/recommend/data2.txt"));
		FileOutputFormat.setOutputPath(job0, output0Path);
		job0.waitForCompletion(true);
		
//		part1----------------------------------------------------
		Job job1 = Job.getInstance(conf, "computeTF");
		Path outputPath1=new Path(hdfsUrl + "/data/recommend/tfidf1");
		HadoopUtil.delete(conf, outputPath1);
		job1.setJarByClass(TFIDF.class);
		job1.setMapperClass(Mapper_Part1.class);
		job1.setReducerClass(Reduce_Part1.class);
		job1.setMapOutputKeyClass(Text.class);
		job1.setMapOutputValueClass(Text.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(Text.class);
		job1.setPartitionerClass(MyPartitoner.class); // 使用自定义MyPartitoner
		FileInputFormat.addInputPath(job1, new Path(hdfsUrl + "/data/recommend/tfidf0"));
		FileOutputFormat.setOutputPath(job1, outputPath1);
		job1.waitForCompletion(true);

//		part2----------------------------------------
		Job job2 = Job.getInstance(conf, "computeTFIDF");
		Path outputPath2=new Path(hdfsUrl + "/data/recommend/tfidf2");
		HadoopUtil.delete(conf, outputPath2);
		job2.setJarByClass(TFIDF.class);
		job2.setMapOutputKeyClass(Text.class);
		job2.setMapOutputValueClass(Text.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);
		job2.setMapperClass(Mapper_Part2.class);
		job2.setReducerClass(Reduce_Part2.class);
		FileInputFormat.setInputPaths(job2, new Path(hdfsUrl + "/data/recommend/tfidf1"));
		FileOutputFormat.setOutputPath(job2, outputPath2);
		job2.waitForCompletion(true);
		
//		part3----------------------------------------
		Configuration conf3 = new Configuration();
		Path outputPath3=new Path(hdfsUrl + "/data/recommend/tfidf3");
		HadoopUtil.delete(conf, outputPath3);
		Job job3 = Job.getInstance(conf3, "My_tdif_part3");
		job3.setMapperClass(Mapper_Part3.class);
		job3.setReducerClass(Reduce_Part3.class);
		job3.setMapOutputKeyClass(CustomKey.class);
		job3.setMapOutputValueClass(NullWritable.class);
		job3.setOutputKeyClass(CustomKey.class);
		job3.setOutputValueClass(NullWritable.class);
		job3.setGroupingComparatorClass(CustomGroupComparator.class);
		job3.setPartitionerClass(CustomPartitioner.class); // 使用自定义MyPartitoner
		FileInputFormat.addInputPath(job3, new Path(hdfsUrl + "/data/recommend/tfidf2"));
		FileOutputFormat.setOutputPath(job3, outputPath3);
		job3.waitForCompletion(true);
		
	}
	
//	part0---------------新闻去重---------------------------------------------------------
	public static class Mapper_Part0 extends Mapper<LongWritable, Text, Text, Text> {
		public void map(LongWritable key, Text value, Context context) throws IOException,
				InterruptedException {
			String[] lineSplits = value.toString().split("\t");
			String newsID = lineSplits[1];
			String content = lineSplits[4];
			String publishTime=lineSplits[5];
			Calendar cal1 = Calendar.getInstance();
			try {
				cal1.setTime(new SimpleDateFormat("yyyy年MM月dd日HH:mm").parse(publishTime));
				publishTime=Long.toString(cal1.getTimeInMillis());
			} catch (Exception e) {
				publishTime="0";
			}
			context.write(new Text(newsID+"|"+publishTime+"|"+content),new Text(""));
		}
	}
	
	public static class Reduce_Part0 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			context.write(key, new Text(""));
		}
	}
	
//	part1------------------计算TF------------------------------------------------------
	public static class Mapper_Part1 extends Mapper<LongWritable, Text, Text, Text> {
		String newsID = "";
		String content = "";
		String publishTime="";
		String word;

		public void map(LongWritable key, Text value, Context context) throws IOException,
				InterruptedException {
			int all = 0; // 单词总数统计
			String[] lineSplits = value.toString().split("\\|");
			newsID = lineSplits[0];
			publishTime = lineSplits[1];
			content = lineSplits[2];

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
//				下面的key值要加上一个单词的总字数 ，在生成每篇文章的词频矩阵时会用到。
				context.write(new Text(key1+"|"+all+"|"+publishTime), new Text((Float.parseFloat(val.toString()) / all)
						+ ""));
			}
		}
	}

	public static class Reduce_Part1 extends Reducer<Text, Text, Text, Text> {
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
				InterruptedException {
			for (Text val : values) {
				context.write(key, val);
			}
		}
	}

	public static class MyPartitoner extends Partitioner<Text, Text> {
		public int getPartition(Text key, Text value, int numPartitions) {
			// 我们将一个文件中计算的结果作为一个文件保存
			// es： test1 test2
			String ip1 = key.toString();
			ip1 = ip1.substring(0, ip1.indexOf(" "));
			Text p1 = new Text(ip1);
			return Math.abs((p1.hashCode() * 127) % numPartitions);
		}
	}

//	part2--------------计算TF.IDF---------------------------------------
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
	
//	part2--------------提取每条新闻TF.IDF前50个关键字作为维度---------------------------------------
	public static class Mapper_Part3 extends Mapper<LongWritable, Text, CustomKey, NullWritable> {
		private CustomKey newKey = new CustomKey();
		public void map(LongWritable key, Text value, Context cxt) throws IOException,
				InterruptedException {
			String[] values = value.toString().split("\t");
			String[] values2=values[1].split(" ");
			newKey.setSymbol(values[0]);
			newKey.setSymbol2(values2[0]);
			newKey.setValue(Double.parseDouble(values2[1]));
			newKey.setValue2(Double.parseDouble(values2[2]));
			cxt.write(newKey, NullWritable.get());
		}
	}
	
	public static class Reduce_Part3 extends Reducer<CustomKey, NullWritable, CustomKey, NullWritable> {
		public void reduce(CustomKey key, Iterable<NullWritable> values, Context cxt)
				throws IOException, InterruptedException {
			int limit=0;
			for (NullWritable v : values) {
				if(++limit<=50)
					cxt.write(key, v);
			}
		}
	}
	
	
	public static class CustomKey implements WritableComparable<CustomKey> {
		private String symbol;
		private String symbol2;
		private double value;
		private double value2;

		@Override
		public void write(DataOutput out) throws IOException {
			out.writeUTF(this.symbol);
			out.writeUTF(this.symbol2);
			out.writeDouble(this.value);
			out.writeDouble(this.value2);
		}

		@Override
		public void readFields(DataInput in) throws IOException {
			this.symbol = in.readUTF();
			this.symbol2 = in.readUTF();
			this.value = in.readDouble();
			this.value2 = in.readDouble();
		}

		@Override
		public int compareTo(CustomKey o) {
			int result = this.symbol.compareTo(o.symbol);
			return result != 0 ? result : (o.value2 - this.value2>0?1:-1);
		}

		@Override
		public String toString() {
			return this.symbol + "\t"+this.symbol2 + " " + this.value+ " " +this.value2;
		}

		public double getValue() {
			return value;
		}

		public void setValue(double value) {
			this.value = value;
		}
		
		public double getValue2() {
			return value2;
		}
		
		public void setValue2(double value2) {
			this.value2 = value2;
		}

		public String getSymbol() {
			return symbol;
		}

		public void setSymbol(String symbol) {
			this.symbol = symbol;
		}
		
		public String getSymbol2() {
			return symbol2;
		}
		
		public void setSymbol2(String symbol2) {
			this.symbol2 = symbol2;
		}

	}

	public static class CustomGroupComparator extends WritableComparator {
		protected CustomGroupComparator() {
			super(CustomKey.class, true);
		}

		@SuppressWarnings("rawtypes")
		@Override
		public int compare(WritableComparable a, WritableComparable b) {
			CustomKey ak = (CustomKey) a;
			CustomKey bk = (CustomKey) b;
			return ak.getSymbol().compareTo(bk.getSymbol());
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

}
