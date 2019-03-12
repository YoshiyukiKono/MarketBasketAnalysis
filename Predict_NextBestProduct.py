from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql import SparkSession

sc = SparkSession.builder \
     .appName("PredictNextBestProduct") \
     .master("local[*]") \
     .enableHiveSupport() \
     .getOrCreate()
  
model = MatrixFactorizationModel.load(sc, "/tmp/amolenaar/model")

items = sc.sql('SELECT * FROM transaction_item')

def predict(args):
  user=args["user"]
  result=model.recommendProducts(user,1)
  result1=items.filter(items.item_id == result[0][1]).collect()
  return {"result" : result1}
