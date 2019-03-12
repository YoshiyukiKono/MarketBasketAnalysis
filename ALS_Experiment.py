from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql import SparkSession
import cdsw

sc = SparkSession.builder \
      .appName("ALSmodel") \
      .getOrCreate()
          
# Load and parse the data
data = sc.sparkContext.textFile("/tmp/transactions_andre_als")

ratings = data.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares

param_Rank=int(sys.argv[1])
param_numIterations=int(sys.argv[2])
model = ALS.trainImplicit(ratings, param_Rank, param_numIterations, alpha=0.01)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

cdsw.track_metric("Rank",param_Rank)
cdsw.track_metric("numIterations",param_numIterations)
cdsw.track_metric("MSE",MSE)
