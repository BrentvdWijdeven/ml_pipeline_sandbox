from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.types as tp
import pyspark.sql.functions as F

import pandas as pd

sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName('PySpark DataFrame From RDD').getOrCreate()

# define the schema
my_schema = tp.StructType([
    tp.StructField(name= 'Batsman',      dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Batsman_Name', dataType= tp.StringType(),    nullable= True),
    tp.StructField(name= 'Bowler',       dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Bowler_Name',  dataType= tp.StringType(),    nullable= True),
    tp.StructField(name= 'Commentary',   dataType= tp.StringType(),    nullable= True),
    tp.StructField(name= 'Detail',       dataType= tp.StringType(),    nullable= True),
    tp.StructField(name= 'Dismissed',    dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Id',           dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Isball',       dataType= tp.BooleanType(),   nullable= True),
    tp.StructField(name= 'Isboundary',   dataType= tp.BooleanType(),   nullable= True),
    tp.StructField(name= 'Iswicket',     dataType= tp.BooleanType(),   nullable= True),
    tp.StructField(name= 'Over',         dataType= tp.DoubleType(),    nullable= True),
    tp.StructField(name= 'Runs',         dataType= tp.IntegerType(),   nullable= True),
    tp.StructField(name= 'Timestamp',    dataType= tp.TimestampType(), nullable= True)
])



# read the data again with the defined schema
# my_data = spark.read.csv('data/ind-ban-comment.csv', header= True, sep=",")

my_data = spark.read.format("csv").option("header", "true", ).option("delimiter", ",").schema(my_schema)\
    .load('data/ind-ban-comment.csv')
# option("inferSchema"=my_schema)

# see the default schema of the dataframe
my_data.printSchema()

# drop the columns that are not required
my_data = my_data.drop(*['Batsman', 'Bowler', 'Id'])
print(my_data.columns)

# check type and dimensions of data
print(type(my_data))

my_data.show()

test_list = [('1', 1), ('2', 2)]
result = spark.createDataFrame(test_list, ['string', 'value']).collect()
print(result)

"""
Things to learn:
- Machine Learning pipeline in pyspark
- SQL in spark --> goal is to practice both SQL and Spark 
- Spark Streaming (part of Apache Spark Core)
- JSON / TXT file loaden into dataframe
- Analyze performance of spark in SparkUI
"""

my_data.createOrReplaceTempView("table1")
sql_table = spark.sql("SELECT * FROM table1")
sql_table.show()

# sql_table.select("Batsman_Name", F.when(sql_table.Runs >=1, 1).otherwise(0)).show()
# sql_table.select("Batsman_Name").where(sql_table.Runs == 1).show()
# sql_table.select("Batsman_Name", "Bowler_Name").where(sql_table.Detail == "W").show()
# sql_table.select("Bowler_Name").where(sql_table.Batsman_Name == "Mohammed Shami").show()
# sql_table.select("Bowler_Name").where(sql_table.Bowler_Name == "Mohammed Shami").show()
# sql_table.select("Batsman_Name", "Bowler_name").where(sql_table.Bowler_Name == "Mohammed Shami").show()

# sql_table.select("Batsman_name",  )

print(type(my_data))
print(type(sql_table))
my_data.select("Isball", 'Isboundary', 'Runs').describe().show()
my_data.groupBy('Batsman_Name').count().show()


# encoding categorical variables
# https://www.analyticsvidhya.com/blog/2019/11/build-machine-learning-pipelines-pyspark/
# string indexing
# One hot encoding


#  VectorAssembler
""" The Vector Assembler converts feature columns into a single feature column in order to train the machine learning mode
It accepts numeric, boolean and vector type columns
"""
from pyspark.ml.feature import StringIndexer

SI_batsman = StringIndexer(inputCol='Batsman_Name',outputCol='Batsman_Index')
test = SI_batsman.fit(my_data).transform(my_data)


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=[
                                       'Over',
                                       'Runs',
                                       'Batsman_Index'],
                            outputCol="vector"
                            )

# fill the null values
test = test.na.fill(0)
test.show()
# transform the data
final_data = assembler.transform(test)
print(final_data.columns)

# view the transformed vector
final_data.select('Batsman_Index', 'vector').show()
# -------------------------
# # Read Json or txt file into dataframe
# json_df = spark.read.json('data/unconfirmed_bitcoin_transactions.json')
# json_df.show()




# MACHINE LEARNING PIPELINES
"""
Each stage in a pipeline is either a Transformer or an Estimator
Transformer: convert one dataframe into another by applying a transformation to some function
Estimator: implements the fit() method on the dataframe and produces a model.
"""
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression


df = my_data.select("*")
df = df.na.fill(0)

stage_1 = StringIndexer(inputCol='Batsman_Name',outputCol='Batsman_Index')

stage_2 = StringIndexer(inputCol='Bowler_Name',outputCol='Bowler_Index')

stage_3 = VectorAssembler(inputCols=[ 'Bowler_Index',
                                       'Over',
                                       'Batsman_Index'],
                            outputCol="features"
                            )

stage_4 = LogisticRegression(featuresCol='features', labelCol='Runs')


regression_pipeline = Pipeline(stages=[stage_1, stage_2, stage_3, stage_4])

# train test split
train_df, test_df = df.randomSplit([0.6, 0.3])


# Train
pipeline_model = regression_pipeline.fit(train_df)
train_result = pipeline_model.transform(train_df)
print('Pipeline runt als een gekkk')
train_result.show()

# Test
test_result = pipeline_model.transform(test_df)
test_result.show()


