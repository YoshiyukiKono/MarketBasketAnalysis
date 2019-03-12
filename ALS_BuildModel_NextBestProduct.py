from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql import SparkSession

sc = SparkSession.builder \
      .appName("ALSmodel") \
      .getOrCreate()
          
# Load and parse the data
data = sc.sparkContext.textFile("/tmp/transactions_andre_als")

ratings = data.map(lambda l: l.split(','))\
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 15
numIterations = 5
model = ALS.trainImplicit(ratings, rank, numIterations, alpha=0.01)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
model.save(sc.sparkContext, "/tmp/amolenaar/model")
